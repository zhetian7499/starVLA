"""Qwen × action-head profiling bench. See README in this directory."""
import argparse
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

HEAD_TO_FRAMEWORK = {
    "OFT": "QwenOFT",
    "PI": "QwenPI",
    "FAST": "QwenFast",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen × action-head profile bench")
    p.add_argument("--head", choices=list(HEAD_TO_FRAMEWORK), required=True,
                   help="Which action head to benchmark.")
    p.add_argument("--config_yaml", default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
                   help="Base LIBERO config; overridden by --head and CLI dotlist.")
    p.add_argument("--base_vlm", default="playground/Pretrained_models/Qwen3.5-0.8B",
                   help="Path to Qwen3.5-0.8B weights on disk.")
    p.add_argument("--data_root", default="playground/Datasets/LEROBOT_LIBERO_DATA",
                   help="Path to LIBERO LeRobot data on disk.")
    p.add_argument("--output_dir", default="./out", help="Where artifacts are written.")
    p.add_argument("--batch_path", default=None,
                   help="If set, load frozen batch from this .pt file instead of dataloader.")
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--active_steps", type=int, default=10)
    p.add_argument("--cooldown_steps", type=int, default=5)
    p.add_argument("--no_profile", action="store_true",
                   help="Run the loop without torch.profiler — useful for smoke testing.")
    p.add_argument("--selftest_hooks", action="store_true",
                   help="Run a local hook-firing self-test and exit.")
    return p.parse_args()


def load_config(args: argparse.Namespace):
    """Load LIBERO base yaml, merge bench-specific overrides."""
    cfg = OmegaConf.load(args.config_yaml)
    overrides = OmegaConf.from_dotlist([
        f"framework.name={HEAD_TO_FRAMEWORK[args.head]}",
        f"framework.qwenvl.base_vlm={args.base_vlm}",
        f"datasets.vla_data.data_root_dir={args.data_root}",
        # Minimal training fields the bench needs but doesn't actually use.
        "trainer.max_train_steps=999999",
        "trainer.logging_frequency=999999",
        "trainer.save_interval=999999",
        "trainer.eval_interval=999999",
        "trainer.freeze_modules=",
        "trainer.gradient_accumulation_steps=1",
        # Disable wandb side effects in this bench.
        "wandb_entity=bench",
        "wandb_project=bench",
        "is_debug=false",
        "run_id=profile_bench",
        "run_root_dir=./out",
    ])
    cfg = OmegaConf.merge(cfg, overrides)

    # starVLA's config-compat shim, same as train_starvla.py
    from starVLA.model.framework.share_tools import apply_config_compat
    cfg = apply_config_compat(cfg)
    return cfg


import torch
import random
import numpy as np
from starVLA.model.framework.base_framework import build_framework

def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(cfg) -> torch.nn.Module:
    """Build framework; assert the two submodules we want to hook exist."""
    model = build_framework(cfg)
    if not hasattr(model, "qwen_vl_interface"):
        attrs = [a for a in dir(model) if not a.startswith("_")]
        raise RuntimeError(
            f"framework missing `qwen_vl_interface`. Available top-level attrs: {attrs}"
        )
    if not hasattr(model, "action_model"):
        attrs = [a for a in dir(model) if not a.startswith("_")]
        raise RuntimeError(
            f"framework missing `action_model`. Available top-level attrs: {attrs}"
        )
    return model


from starVLA.dataloader import build_dataloader

def build_loader(cfg):
    """Wrap build_dataloader with bench-friendly overrides (single worker, no shuffle drift)."""
    # NOTE: per-device batch size is read from cfg.datasets.vla_data.per_device_batch_size
    return build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

def get_or_dump_batch(loader, batch_path: str) -> object:
    """Load a frozen batch from disk if present; otherwise fetch one and dump."""
    p = Path(batch_path)
    if p.exists():
        print(f"[bench] loading frozen batch from {p}")
        return torch.load(p, map_location="cpu")
    print(f"[bench] dumping one batch from dataloader to {p}")
    batch = next(iter(loader))
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, p)
    return batch


from contextlib import contextmanager

@contextmanager
def prof_range(name: str):
    """Push both NVTX range (for nsys/asys) and record_function (for torch.profiler).

    On PPU `torch.cuda.nvtx` becomes a no-op, which is fine — the PPU side
    relies on model_prof's libnvToolsExt loading instead.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    rf = torch.profiler.record_function(name)
    rf.__enter__()
    try:
        yield
    finally:
        rf.__exit__(None, None, None)
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def register_module_hooks(model: torch.nn.Module, name_to_module: dict) -> list:
    """Attach pre/post forward hooks that push NVTX + record_function around each module.

    Returns a list of hook handles so the caller can `.remove()` them later.
    """
    handles = []
    for label, module in name_to_module.items():
        def make_hooks(_label):
            def pre(mod, _inputs):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(_label)
                rf = torch.profiler.record_function(_label)
                rf.__enter__()
                mod._bench_prof_ctx = rf
            def post(mod, _inputs, _outputs):
                ctx = getattr(mod, "_bench_prof_ctx", None)
                if ctx is not None:
                    ctx.__exit__(None, None, None)
                    del mod._bench_prof_ctx
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
            return pre, post
        pre_h, post_h = make_hooks(label)
        handles.append(module.register_forward_pre_hook(pre_h))
        handles.append(module.register_forward_hook(post_h))
    return handles


def _selftest_hooks():
    """Local-only sanity check that hooks fire in the right order."""
    seen = []

    class FakeRF:
        def __init__(self, name): self.name = name
        def __enter__(self): seen.append(f"rf_enter:{self.name}"); return self
        def __exit__(self, *a): seen.append(f"rf_exit:{self.name}")

    # Monkey-patch record_function for this call only
    real_rf = torch.profiler.record_function
    torch.profiler.record_function = FakeRF
    try:
        m = torch.nn.Linear(4, 4)
        register_module_hooks(m, {"toy_forward": m})
        m(torch.zeros(1, 4))
    finally:
        torch.profiler.record_function = real_rf
    assert seen == ["rf_enter:toy_forward", "rf_exit:toy_forward"], seen
    print(f"[bench] selftest_hooks OK: {seen}")


import time
from accelerate import Accelerator, DeepSpeedPlugin


def setup_accelerator():
    """Match train_starvla.py's accelerator setup."""
    ds_plugin = DeepSpeedPlugin()
    return Accelerator(deepspeed_plugin=ds_plugin)


def setup_optimizer(model, lr: float = 1e-5):
    """Minimal AdamW. We don't need an LR scheduler for a 45-step bench."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )


def run_loop(model, optimizer, batch, total_steps: int, hooks_target: dict | None):
    """Single-process body of the training loop.

    `model` is the *prepared* (Accelerator-wrapped) model. `hooks_target` maps
    label -> submodule to hook for module-level NVTX/record_function ranges.
    """
    if hooks_target is not None:
        register_module_hooks(model, hooks_target)

    step_times = []
    for step in range(total_steps):
        t0 = time.perf_counter()
        with prof_range("step_total"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
                loss = out["action_loss"]
            with prof_range("backward"):
                loss.backward()
            with prof_range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        if step % 5 == 0:
            print(f"[bench] step {step:3d} loss={loss.item():.4f} dt={dt*1000:.1f}ms")
    return step_times


def main():
    args = parse_args()
    if args.selftest_hooks:
        _selftest_hooks()
        return 0
    cfg = load_config(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    set_global_seed(42)
    print("[bench] building framework ...")
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] framework class = {type(model).__name__}")
    print(f"[bench] total parameters = {n_params/1e9:.2f}B")
    print(f"[bench] qwen_vl_interface type = {type(model.qwen_vl_interface).__name__}")
    print(f"[bench] action_model type      = {type(model.action_model).__name__}")
    print("[bench] building dataloader ...")
    loader = build_loader(cfg)
    batch_path = args.batch_path or str(Path(args.output_dir) / "fixed_batch.pt")
    batch = get_or_dump_batch(loader, batch_path)
    print(f"[bench] batch type = {type(batch).__name__}, len = {len(batch) if hasattr(batch, '__len__') else '?'}")

    accelerator = setup_accelerator()
    optimizer = setup_optimizer(model)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Resolve hook targets after .prepare() may have wrapped the model
    inner = accelerator.unwrap_model(model)
    hooks_target = {
        "vlm_forward": inner.qwen_vl_interface,
        "action_head_forward": inner.action_model,
    }

    total_steps = args.warmup_steps + args.active_steps + args.cooldown_steps
    if args.no_profile:
        print(f"[bench] starting plain loop for {total_steps} steps")
        run_loop(model, optimizer, batch, total_steps=total_steps, hooks_target=hooks_target)
        print("[bench] loop done (no profile)")
        return 0

    # Profile-enabled path comes in Task 7.
    print("[bench] --no_profile not set; profile path not yet implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
