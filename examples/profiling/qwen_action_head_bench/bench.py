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


def main():
    args = parse_args()
    cfg = load_config(args)
    print(f"[bench] resolved framework.name = {cfg.framework.name}")
    print(f"[bench] resolved base_vlm       = {cfg.framework.qwenvl.base_vlm}")
    print(f"[bench] resolved data_root      = {cfg.datasets.vla_data.data_root_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
