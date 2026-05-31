"""
In-process LIBERO evaluation for QwenPI.

Skips the websocket policy server entirely — loads the model via
``baseframework.from_pretrained`` in the same Python process as the LIBERO
simulator and calls ``predict_action`` directly.  Single GPU, single process.

Training used the agentview camera (rotated 180° to match training data
preprocessing).  Action chunk size = ``model.action_horizon``.

Usage:
    MUJOCO_GL=osmesa PYTHONPATH=/workspace/LIBERO:/workspace/starVLA \\
    CUDA_VISIBLE_DEVICES=0 python examples/LIBERO/eval_files/eval_libero_local_qwenpi.py \\
      --ckpt playground/Checkpoints/libero_h20_validation/checkpoints/steps_200_pytorch_model.pt \\
      --task-suite libero_goal --num-trials 3
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import json
import math
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# torch.load compat patch — LIBERO ships pickled numpy init_states that
# PyTorch >= 2.6 refuses under the default ``weights_only=True``.
# We trust LIBERO, so force the legacy path.
# ---------------------------------------------------------------------------
_original_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Force unbuffered logging to stderr
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
_handler.flush = lambda: sys.stderr.flush()  # type: ignore
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger("eval_libero_local_qwenpi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quat2axisangle(quat):
    """Copied from robosuite (used by LIBERO)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _binarize_gripper_open(open_val) -> np.ndarray:
    """Binarize gripper: >0.5 → -1 (closed), <=0.5 → 1 (open)."""
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def unnormalize_actions(
    normalized_actions: np.ndarray,
    action_norm_stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """Un-normalize actions from [-1, 1] back to raw space.

    Gripper dim (index 6) is binarized before un-normalization.
    """
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high = np.asarray(action_norm_stats["max"])
    action_low = np.asarray(action_norm_stats["min"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


def get_max_steps(task_suite: str) -> int:
    """Return the standard max-step limit for each LIBERO suite."""
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite]


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class EvalArgs:
    ckpt: str = ""
    task_suite: str = "libero_goal"
    num_trials: int = 3
    num_steps_wait: int = 10
    seed: int = 7
    video_out: Optional[str] = None
    unnorm_key: str = "franka"
    use_wrist: bool = False
    use_state: bool = False
    log_per_step: bool = False


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run(args: EvalArgs) -> None:
    np.random.seed(args.seed)

    # === LIBERO sim setup ===
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    max_steps = get_max_steps(args.task_suite)
    log.info(
        f"task_suite={args.task_suite}  num_tasks={num_tasks}  "
        f"max_steps={max_steps}  trials/task={args.num_trials}"
    )

    # === Model loading via generic from_pretrained ===
    from starVLA.model.framework.base_framework import baseframework

    log.info(f"loading checkpoint: {args.ckpt}")
    t0 = time.time()
    model = baseframework.from_pretrained(args.ckpt)
    log.info(f"from_pretrained ok in {time.time() - t0:.1f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if hasattr(torch.cuda, "memory_allocated") and torch.cuda.is_available():
        log.info(f"GPU memory after move: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    log.info(f"model on {device}")

    # === Action stats / chunk size ===
    # QwenPI stores chunk length as ``model.action_horizon``.
    # Legacy checkpoints may expose ``future_action_window_size`` instead
    # (action_horizon == future_action_window_size + 1).
    norm_stats = model.norm_stats[args.unnorm_key]["action"]
    chunk_size = getattr(model, "action_horizon", None)
    if chunk_size is None:
        chunk_size = getattr(model, "future_action_window_size", 7) + 1
    log.info(f"unnorm_key={args.unnorm_key}  action_chunk_size={chunk_size}")
    log.info(f"action min: {np.asarray(norm_stats['min'])}")
    log.info(f"action max: {np.asarray(norm_stats['max'])}")
    log.info(f"use_wrist={args.use_wrist}  use_state={args.use_state}")

    if args.video_out:
        pathlib.Path(args.video_out).mkdir(parents=True, exist_ok=True)

    # === Output paths ===
    progress_path = pathlib.Path("/workspace/starVLA/playground/eval_progress.jsonl")
    final_path = pathlib.Path("/workspace/starVLA/playground/eval_final_results.json")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_path.exists():
        progress_path.unlink()

    def _append_jsonl(record: dict) -> None:
        with open(progress_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # === Eval loop ===
    total_episodes = 0
    total_successes = 0
    per_task_results: Dict[str, Dict[str, int]] = {}
    step_times: List[float] = []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)

        task_episodes = 0
        task_successes = 0
        log.info(f"[task {task_id + 1}/{num_tasks}] {task_description}")

        for ep_idx in range(args.num_trials):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])

            t = 0
            step = 0
            done = False
            replay_imgs: List[np.ndarray] = []
            cached_unnorm_actions: Optional[np.ndarray] = None

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Match training preprocessing: rotate 180° (training data was rotated)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                replay_imgs.append(img)

                # Build image list — agentview (primary) always included
                images = [Image.fromarray(img.astype(np.uint8))]
                if args.use_wrist:
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    images.append(Image.fromarray(wrist_img.astype(np.uint8)))

                # Optional 7-dim proprioceptive state
                state_input = None
                if args.use_state:
                    gripper_q = np.asarray(
                        obs["robot0_gripper_qpos"], dtype=np.float32
                    ).reshape(-1)
                    state7 = np.concatenate(
                        (
                            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                            _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32).reshape(-1),
                            gripper_q[:1],
                        )
                    )
                    state_input = state7[None].astype(np.float32)  # (1, 7)

                if step % chunk_size == 0:
                    example: dict = {
                        "image": images,
                        "lang": str(task_description),
                    }
                    if state_input is not None:
                        example["state"] = state_input

                    t_step_start = time.time()
                    with torch.no_grad():
                        out = model.predict_action([example])
                    t_step_end = time.time()
                    step_times.append(t_step_end - t_step_start)

                    normed = out["normalized_actions"][0]  # (chunk, action_dim)
                    cached_unnorm_actions = unnormalize_actions(normed, norm_stats)

                    if args.log_per_step:
                        log.info(
                            f"  step {step}: predict_action took "
                            f"{(t_step_end - t_step_start) * 1000:.0f}ms"
                        )

                cur_action = cached_unnorm_actions[step % chunk_size]
                # Re-binarize gripper using training convention
                wv = cur_action[:3]
                rot = cur_action[3:6]
                grip = _binarize_gripper_open(cur_action[6:7])
                action7 = np.concatenate([wv, rot, grip], axis=0)
                obs, _, done, _ = env.step(action7.tolist())

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            log.info(
                f"  ep{ep_idx} {'SUCCESS' if done else 'fail'}  "
                f"task_sr={task_successes}/{task_episodes}  "
                f"total_sr={total_successes}/{total_episodes} "
                f"({100 * total_successes / total_episodes:.1f}%)"
            )

            _append_jsonl(
                {
                    "type": "episode",
                    "task_id": task_id,
                    "task_description": str(task_description),
                    "episode": ep_idx,
                    "success": bool(done),
                    "task_successes": int(task_successes),
                    "task_episodes": int(task_episodes),
                    "total_successes": int(total_successes),
                    "total_episodes": int(total_episodes),
                    "timestamp": time.time(),
                }
            )

            # Optional video recording
            if args.video_out and replay_imgs:
                import imageio

                tag = "success" if done else "failure"
                fname = pathlib.Path(args.video_out) / f"task{task_id}_ep{ep_idx}_{tag}.mp4"
                imageio.mimwrite(str(fname), replay_imgs, fps=10)

        per_task_results[task_description] = {
            "success": task_successes,
            "total": task_episodes,
        }
        log.info(
            f"[task {task_id + 1}] FINAL "
            f"sr={task_successes}/{task_episodes} "
            f"({100 * task_successes / task_episodes:.1f}%)"
        )

        _append_jsonl(
            {
                "type": "task_summary",
                "task_id": task_id,
                "task_description": str(task_description),
                "successes": int(task_successes),
                "episodes": int(task_episodes),
                "success_rate": float(100 * task_successes / task_episodes) if task_episodes > 0 else 0.0,
                "timestamp": time.time(),
            }
        )

        env.close()

    # === Summary ===
    log.info("=" * 60)
    if total_episodes > 0:
        log.info(
            f"FINAL TOTAL SR: {total_successes}/{total_episodes} "
            f"({100 * total_successes / total_episodes:.1f}%)"
        )
    log.info("Per task:")
    for k, v in per_task_results.items():
        sr = 100 * v["success"] / v["total"] if v["total"] else 0
        log.info(f"  {sr:5.1f}%  {v['success']:3d}/{v['total']:3d}  {k}")
    if step_times:
        avg_ms = np.mean(step_times) * 1000
        log.info(
            f"Inference perf: avg {avg_ms:.0f}ms/step "
            f"over {len(step_times)} predictions"
        )

    final_results = {
        "total_successes": int(total_successes),
        "total_episodes": int(total_episodes),
        "total_success_rate": float(100 * total_successes / total_episodes) if total_episodes > 0 else 0.0,
        "per_task": {
            k: {
                "successes": int(v["success"]),
                "episodes": int(v["total"]),
                "success_rate": float(100 * v["success"] / v["total"]) if v["total"] > 0 else 0.0,
            }
            for k, v in per_task_results.items()
        },
        "timestamp": time.time(),
    }
    with open(final_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    log.info(f"Final results written to {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Local LIBERO evaluation for QwenPI (no policy server)."
    )
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--task-suite",
        default="libero_goal",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-out", default=None)
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument(
        "--use-wrist",
        action="store_true",
        help="Include wrist camera image (2nd view) in model input",
    )
    parser.add_argument(
        "--use-state",
        action="store_true",
        help="Include 7-dim proprioceptive state in model input",
    )
    parser.add_argument(
        "--log-per-step",
        action="store_true",
        help="Log per-step inference latency",
    )
    args = parser.parse_args()

    run(
        EvalArgs(
            ckpt=args.ckpt,
            task_suite=args.task_suite,
            num_trials=args.num_trials,
            num_steps_wait=args.num_steps_wait,
            seed=args.seed,
            video_out=args.video_out,
            unnorm_key=args.unnorm_key,
            use_wrist=args.use_wrist,
            use_state=args.use_state,
            log_per_step=args.log_per_step,
        )
    )


if __name__ == "__main__":
    main()
