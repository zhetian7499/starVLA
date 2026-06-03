"""LIBERO seq-len distribution profiler — Step 1 of the workload-by-length study.

For each sampled frame, measures the three components that drive LLM input length:
  - image_tokens:        sum over views of (grid_t*grid_h*grid_w) // merge_size**2
                         from the Qwen-VL image_processor (smart_resize-aware)
  - lang_tokens:         instruction length under the Qwen text tokenizer
  - fast_action_tokens:  FAST tokenizer output length on the [T, action_dim] chunk

Then reports two totals:
  - total_oftpi = image + lang
      OFT/PI heads consume hidden states; actions never enter the LLM stream.
  - total_fast  = image + lang + fast_action
      FAST head autoregressively predicts <robot_action_N> tokens; the action
      sequence is part of the LLM input on the training-time teacher-forcing pass.

CPU-only; the script never touches GPU. Mirrors bench.py's config loading so
the numbers match what the bench actually feeds the model.

Example:
  python examples/profiling/qwen_action_head_bench/dataset_seqlen_stats.py \\
      --base_vlm /mnt/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3.5-9B \\
      --num_samples 2000

To scan a single LIBERO suite, override --data_mix (libero_goal / libero_object /
libero_spatial / libero_10 / libero_franka — whatever's registered in
DATASET_NAMED_MIXTURES).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image


def _apply_tokenizer_file_patch():
    """Same patch as bench.py — FAST processor needs it on transformers 5.x."""
    from transformers import PreTrainedTokenizerFast
    if getattr(PreTrainedTokenizerFast.__init__, "_bench_patched", False):
        return
    _orig = PreTrainedTokenizerFast.__init__

    def _patched(self, *args, **kwargs):
        if not kwargs.get("tokenizer_file"):
            nop = kwargs.get("name_or_path")
            if nop:
                cand = os.path.join(nop, "tokenizer.json")
                if os.path.isfile(cand):
                    kwargs["tokenizer_file"] = cand
        return _orig(self, *args, **kwargs)
    _patched._bench_patched = True
    PreTrainedTokenizerFast.__init__ = _patched


def parse_args():
    p = argparse.ArgumentParser(description="LIBERO seq-len distribution profiler")
    p.add_argument("--config_yaml",
                   default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
                   help="Base LIBERO yaml — same one bench.py uses.")
    p.add_argument("--data_root",
                   default="playground/Datasets/LEROBOT_LIBERO_DATA")
    p.add_argument("--base_vlm",
                   default="/mnt/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3.5-9B",
                   help="VLM path. Must be a vision-language Qwen so AutoProcessor "
                        "returns both a text tokenizer and an image_processor.")
    p.add_argument("--fast_tokenizer",
                   default="playground/Pretrained_models/fast",
                   help="Path / HF id of the physical-intelligence/fast processor.")
    p.add_argument("--data_mix", default=None,
                   help="Override datasets.vla_data.data_mix to scan a single suite.")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--output", default="./out_profile/dataset_seqlen_stats.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(args):
    """Mirror bench.py's overrides; only the dataset side matters here."""
    cfg = OmegaConf.load(args.config_yaml)
    overrides = [
        # Force FAST framework so apply_config_compat picks the right action_model
        # branch — we read action_horizon / action_dim out of cfg below.
        "framework.name=QwenFast",
        f"framework.qwenvl.base_vlm={args.base_vlm}",
        f"datasets.vla_data.data_root_dir={args.data_root}",
        # Bench-mode minimal trainer fields (unused but the config schema demands them).
        "trainer.max_train_steps=999999",
        "trainer.logging_frequency=999999",
        "trainer.save_interval=999999",
        "trainer.eval_interval=999999",
        "trainer.freeze_modules=",
        "trainer.gradient_accumulation_steps=1",
        "wandb_entity=bench",
        "wandb_project=bench",
        "is_debug=false",
        "run_id=seqlen_stats",
        "run_root_dir=./out",
        # Iterate per-sample so we can stop exactly at --num_samples.
        "datasets.vla_data.per_device_batch_size=1",
    ]
    if args.data_mix:
        overrides.append(f"datasets.vla_data.data_mix={args.data_mix}")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    from starVLA.model.framework.share_tools import apply_config_compat
    cfg = apply_config_compat(cfg)
    return cfg


def _to_pil(im):
    """Best-effort conversion to PIL.Image — dataloaders return PIL / np / tensor."""
    if isinstance(im, Image.Image):
        return im
    if isinstance(im, np.ndarray):
        arr = im
    else:
        # torch.Tensor or similar
        arr = im.detach().cpu().numpy() if hasattr(im, "detach") else np.asarray(im)
    # CHW → HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        # assume [0, 1] float if not uint8
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if arr.dtype.kind == "f" else arr.astype(np.uint8)
    return Image.fromarray(arr)


def count_image_tokens(images, image_processor, merge_size):
    """Return (total_image_tokens, [per_view_tokens])."""
    pil = [_to_pil(im) for im in images]
    out = image_processor(images=pil, return_tensors="pt")
    # Qwen2.5-VL / Qwen3-VL processors both expose image_grid_thw.
    grid_thw = out.get("image_grid_thw")
    if grid_thw is None:
        # Fallback: some processors store it under a different key.
        for k in ("image_grid_hws", "grid_thw", "video_grid_thw"):
            if k in out:
                grid_thw = out[k]
                break
    if grid_thw is None:
        raise RuntimeError(f"image_processor output has no grid_thw; keys={list(out.keys())}")
    per_view = (grid_thw.prod(dim=-1) // (merge_size ** 2)).tolist()
    return int(sum(per_view)), per_view


def count_fast_action_tokens(action, fast_processor):
    """FAST processor returns variable-length token ids for the action chunk."""
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]   # [1, T, D]
    out = fast_processor(arr)
    seq = out[0]
    if isinstance(seq, (list, tuple)):
        return len(seq)
    if hasattr(seq, "__len__") and not isinstance(seq, str):
        return len(seq)
    if isinstance(seq, str):
        # token string like "<robot_action_12><robot_action_3>..." — count '<'
        return seq.count("<")
    return len(seq)


def stats_block(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
    }


def percentile_line(name, arr):
    a = np.asarray(arr)
    return (f"{name:>22} | min={a.min():6.0f} "
            f"p10={np.percentile(a,10):6.0f} p50={np.percentile(a,50):6.0f} "
            f"p90={np.percentile(a,90):6.0f} p99={np.percentile(a,99):6.0f} "
            f"max={a.max():6.0f} mean={a.mean():6.1f} std={a.std():5.1f}")


def ascii_histogram(values, bins=15, width=50, name=""):
    arr = np.asarray(values)
    if len(set(arr.tolist())) == 1:
        print(f"\n[hist] {name}: constant = {arr[0]} (n={len(arr)})")
        return
    counts, edges = np.histogram(arr, bins=bins)
    peak = counts.max() if counts.max() > 0 else 1
    print(f"\n[hist] {name} (n={len(arr)})")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(width * c / peak)
        print(f"  {lo:8.1f} - {hi:8.1f} | {c:6d} {bar}")


def _ensure_dist():
    """starVLA's dataloader calls dist.get_rank() unconditionally."""
    import torch.distributed as dist
    if dist.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo")


def main():
    args = parse_args()
    _apply_tokenizer_file_patch()
    np.random.seed(args.seed)

    print(f"[stats] base_vlm       = {args.base_vlm}")
    print(f"[stats] fast_tokenizer = {args.fast_tokenizer}")
    print(f"[stats] config_yaml    = {args.config_yaml}")
    cfg = load_config(args)
    print(f"[stats] data_mix       = {cfg.datasets.vla_data.data_mix}")
    print(f"[stats] action_horizon = {cfg.framework.action_model.action_horizon}, "
          f"action_dim = {cfg.framework.action_model.action_dim}")

    from transformers import AutoProcessor, AutoTokenizer
    print(f"[stats] loading Qwen processor ...")
    processor = AutoProcessor.from_pretrained(args.base_vlm, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and hasattr(processor, "image_processor"):
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
    else:
        # AutoProcessor returned just the image processor (rare); load tokenizer separately.
        tokenizer = AutoTokenizer.from_pretrained(args.base_vlm, trust_remote_code=True)
        image_processor = processor
    merge_size = int(getattr(image_processor, "merge_size", 2))
    print(f"[stats] merge_size     = {merge_size}")

    print(f"[stats] loading FAST processor ...")
    fast_processor = AutoProcessor.from_pretrained(args.fast_tokenizer, trust_remote_code=True)
    # QwenFast wires these at framework init; mirror it so encode behaves the same.
    fast_processor.time_horizon = int(cfg.framework.action_model.action_horizon)
    fast_processor.action_dim = int(cfg.framework.action_model.action_dim)

    _ensure_dist()
    # starVLA's dataloader writes dataset_statistics.json to cfg.output_dir on
    # rank 0 — same workaround bench.py uses (bench.py:515).
    cfg.output_dir = str(Path(args.output).parent)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[stats] building dataloader ...")
    from starVLA.dataloader import build_dataloader
    loader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    per_sample = []
    n_errors = 0
    first_dump_done = False
    for it, batch in enumerate(loader):
        if len(per_sample) >= args.num_samples:
            break
        samples = batch if isinstance(batch, (list, tuple)) else [batch]
        for sample in samples:
            if not first_dump_done:
                print(f"[stats] first sample keys: {list(sample.keys())}")
                if "image" in sample:
                    im0 = sample["image"]
                    if isinstance(im0, list):
                        v0 = im0[0]
                        print(f"[stats]   n_views={len(im0)} view0 type={type(v0).__name__} "
                              f"size={getattr(v0, 'size', None) or getattr(v0, 'shape', None)}")
                    else:
                        print(f"[stats]   image type={type(im0).__name__} "
                              f"size={getattr(im0, 'size', None) or getattr(im0, 'shape', None)}")
                if "action" in sample:
                    a = sample["action"]
                    print(f"[stats]   action type={type(a).__name__} "
                          f"shape={getattr(a, 'shape', None) or len(a)}")
                first_dump_done = True

            try:
                img = sample["image"]
                lang = sample["lang"]
                act = sample["action"]
                imgs_list = img if isinstance(img, list) else [img]

                lang_tok = len(tokenizer(lang, add_special_tokens=False).input_ids)
                img_tok_total, img_tok_per_view = count_image_tokens(
                    imgs_list, image_processor, merge_size)
                fast_tok = count_fast_action_tokens(act, fast_processor)

                per_sample.append({
                    "image_tokens": img_tok_total,
                    "image_tokens_per_view": img_tok_per_view,
                    "lang_tokens": lang_tok,
                    "fast_action_tokens": fast_tok,
                    "n_views": len(imgs_list),
                })
            except Exception as e:
                n_errors += 1
                if n_errors <= 5:
                    print(f"[stats] sample error #{n_errors}: {e!r}")

            if len(per_sample) >= args.num_samples:
                break

        if (it + 1) % 100 == 0:
            print(f"[stats] iter {it+1}, collected={len(per_sample)} errors={n_errors}")

    print(f"\n[stats] done: {len(per_sample)} samples, {n_errors} errors")
    if not per_sample:
        sys.exit("[stats] no samples collected — check config/data paths")

    img_toks = [s["image_tokens"] for s in per_sample]
    lang_toks = [s["lang_tokens"] for s in per_sample]
    fast_toks = [s["fast_action_tokens"] for s in per_sample]
    total_oftpi = [a + b for a, b in zip(img_toks, lang_toks)]
    total_fast = [a + b + c for a, b, c in zip(img_toks, lang_toks, fast_toks)]

    summary = {
        "n_samples": len(per_sample),
        "n_errors": n_errors,
        "base_vlm": args.base_vlm,
        "data_mix": str(cfg.datasets.vla_data.data_mix),
        "action_horizon": int(cfg.framework.action_model.action_horizon),
        "action_dim": int(cfg.framework.action_model.action_dim),
        "merge_size": merge_size,
        "components": {
            "image_tokens": stats_block(img_toks),
            "lang_tokens": stats_block(lang_toks),
            "fast_action_tokens": stats_block(fast_toks),
            "total_oftpi": stats_block(total_oftpi),
            "total_fast": stats_block(total_fast),
        },
    }

    print("\n=== PERCENTILES ===")
    for name, arr in [
        ("image_tokens", img_toks),
        ("lang_tokens", lang_toks),
        ("fast_action_tokens", fast_toks),
        ("total_oftpi", total_oftpi),
        ("total_fast", total_fast),
    ]:
        print(percentile_line(name, arr))

    for name, arr in [
        ("image_tokens", img_toks),
        ("lang_tokens", lang_toks),
        ("fast_action_tokens", fast_toks),
        ("total_oftpi", total_oftpi),
        ("total_fast", total_fast),
    ]:
        ascii_histogram(arr, bins=15, name=name)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "summary": summary,
        "per_sample": per_sample,
    }, indent=2))
    print(f"\n[stats] wrote -> {args.output}")


if __name__ == "__main__":
    main()
