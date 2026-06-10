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
    p.add_argument("--base_vlm", default="/mnt/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3.5-0.8B",
                   help="Path to Qwen3.5-0.8B weights on disk.")
    p.add_argument("--data_root", default="playground/Datasets/LEROBOT_LIBERO_DATA",
                   help="Path to LIBERO LeRobot data on disk.")
    p.add_argument("--output_dir", default="./out", help="Where artifacts are written.")
    p.add_argument("--batch_path", default=None,
                   help="If set, load frozen batch from this .pt file instead of dataloader.")
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--active_steps", type=int, default=10)
    p.add_argument("--cooldown_steps", type=int, default=5)
    p.add_argument("--profiler", choices=["nsys", "torch", "none"], default="torch",
                   help="Which profiler stack to enable. 'nsys' uses model_prof's "
                        "iter-range (assumes the process is wrapped by prof.sh / nsys / asys); "
                        "'torch' uses torch.profiler with tensorboard handler; "
                        "'none' is a smoke-test loop with no profiler. Running both "
                        "nsys and torch in one execution pollutes per-step timing — "
                        "use two separate runs instead.")
    p.add_argument("--selftest_hooks", action="store_true",
                   help="Run a local hook-firing self-test and exit.")
    p.add_argument("--detailed_module_hooks", action="store_true",
                   help="In addition to vlm_forward/action_head_forward, register "
                        "NVTX + record_function on every leaf module (nn.Linear, "
                        "RMSNorm, Attention, …). Annotates trace with 'mod:<path>:<type>' "
                        "so per-module kernel attribution is possible. Skip-list keeps "
                        "trace size in check by ignoring Dropout/Embedding/activations.")
    p.add_argument("--with_stack", action="store_true",
                   help="Pass with_stack=True to torch.profiler.profile so "
                        "each event carries a Python call stack. Inflates "
                        "chrome trace size ~5-10x; only enable when you need "
                        "callsite attribution. nsys pass is unaffected.")
    p.add_argument("--batch_size_override", type=int, default=None,
                   help="If set, slice frozen batch to this size after loading. "
                        "Use when the full frozen batch OOMs (e.g. torch >=2.7 "
                        "uses more memory than 2.6 for the same model).")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VAL",
                   help="OmegaConf dotlist override (repeatable). Applied AFTER "
                        "head/path overrides so --set wins on conflicts. "
                        "Example: --set datasets.vla_data.per_device_batch_size=1")
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

    # User-supplied --set k=v overrides win over the head/path defaults above.
    if getattr(args, "overrides", None):
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))

    # starVLA's config-compat shim, same as train_starvla.py
    from starVLA.model.framework.share_tools import apply_config_compat
    cfg = apply_config_compat(cfg)
    return cfg


import torch
import random
import numpy as np
from starVLA.model.framework.base_framework import build_framework


def _apply_tokenizer_file_patch():
    """Work around transformers 5.x dropping `tokenizer_file` in nested processor
    loading. Needed by the QwenFast head to load physical-intelligence/fast.
    Idempotent and harmless to other tokenizers (only fires when tokenizer.json
    exists next to the model path and no explicit tokenizer_file was passed)."""
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
        return torch.load(p, map_location="cpu", weights_only=False)
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


def register_module_hooks(model: torch.nn.Module, name_to_module: dict,
                          also_backward: bool = False) -> list:
    """Attach pre/post forward hooks that push NVTX + record_function around each module.

    If also_backward=True, additionally attach register_full_backward_{pre,post}_hook
    so backward-time kernels can also be attributed. The backward annotation uses
    the label prefix 'bwd:' to disambiguate from forward in the resulting trace.

    Returns a list of hook handles so the caller can `.remove()` them later.
    """
    handles = []
    for label, module in name_to_module.items():
        def make_fwd_hooks(_label):
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
        pre_h, post_h = make_fwd_hooks(label)
        handles.append(module.register_forward_pre_hook(pre_h))
        handles.append(module.register_forward_hook(post_h))
        if also_backward:
            bwd_label = f"bwd:{label}"
            def make_bwd_hooks(_label):
                def bpre(mod, _grad_output):
                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_push(_label)
                    rf = torch.profiler.record_function(_label)
                    rf.__enter__()
                    mod._bench_bwd_prof_ctx = rf
                def bpost(mod, _grad_input, _grad_output):
                    ctx = getattr(mod, "_bench_bwd_prof_ctx", None)
                    if ctx is not None:
                        ctx.__exit__(None, None, None)
                        del mod._bench_bwd_prof_ctx
                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()
                return bpre, bpost
            bpre_h, bpost_h = make_bwd_hooks(bwd_label)
            handles.append(module.register_full_backward_pre_hook(bpre_h))
            handles.append(module.register_full_backward_hook(bpost_h))
    return handles


_SKIP_HOOK_TYPES = (
    torch.nn.Dropout, torch.nn.Identity, torch.nn.Embedding,
    torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Tanh, torch.nn.Sigmoid,
)


def collect_leaf_hook_targets(root: torch.nn.Module) -> dict:
    """Walk named_modules(), pick leafs (no children), skip cheap activation/dropout/embed.

    Returns {label: module} where label is `mod:<path>:<ClassName>`. <path> uses
    `named_modules()` convention (dot-separated, indices for ModuleList children).
    Empty <path> = the root module itself; we skip that — root is the framework
    wrapper which we already hook as `vlm_forward`/`action_head_forward`.
    """
    targets = {}
    for name, m in root.named_modules():
        if not name:  # root
            continue
        if list(m.children()):  # not a leaf
            continue
        if isinstance(m, _SKIP_HOOK_TYPES):
            continue
        label = f"mod:{name}:{type(m).__name__}"
        targets[label] = m
    return targets


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


def setup_accelerator(per_device_batch_size: int):
    """Build Accelerator. Use DeepSpeed only when launched via `accelerate launch
    --config_file ...deepspeed_zero2.yaml` (which sets ACCELERATE_USE_DEEPSPEED=true).
    Single-process `python bench.py ...` runs use plain Accelerator — DeepSpeed
    is overkill for a smoke test and would require a dataloader passed to .prepare()
    just to read its batch size.
    """
    use_ds = os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() == "true"
    if not use_ds:
        return Accelerator()
    ds_plugin = DeepSpeedPlugin()
    acc = Accelerator(deepspeed_plugin=ds_plugin)
    # DeepSpeed needs train_micro_batch_size_per_gpu to construct ZeRO. We don't
    # pass a dataloader to .prepare() (the bench manages its own batch), so set it
    # explicitly here.
    acc.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = per_device_batch_size
    return acc


def setup_optimizer(model, lr: float = 1e-5):
    """Minimal AdamW. We don't need an LR scheduler for a 45-step bench."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )


def run_loop(model, optimizer, batch, total_steps: int, hooks_target: dict | None, accelerator):
    """Single-process body of the training loop.

    `model` is the *prepared* (Accelerator-wrapped) model. `hooks_target` maps
    label -> submodule to hook for module-level NVTX/record_function ranges.
    `accelerator` is required to route backward through DeepSpeed engine when
    ZeRO is enabled — calling `loss.backward()` directly trips ZeRO-2's
    "parameter already reduced" assertion.
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
                accelerator.backward(loss)  
            with prof_range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        if step % 5 == 0:
            print(f"[bench] step {step:3d} loss={loss.item():.4f} dt={dt*1000:.1f}ms")
    return step_times


def run_loop_nsys(model, optimizer, batch, args, hooks_target, accelerator):
    """Loop with only model_prof iter-range (nsys / asys captures externally via prof.sh).

    No torch.profiler wrapping — running both at once doubles CUPTI subscribers
    and adds end-of-active trace-export stalls that pollute per-step timing.

    `accelerator` required for the same reason as `run_loop` — backward
    must route through DeepSpeed engine via `accelerator.backward(loss)`.
    """
    import model_prof as mp

    if hooks_target is not None:
        register_module_hooks(model, hooks_target,
                              also_backward=getattr(args, "detailed_module_hooks", False))

    warmup, active, cooldown = args.warmup_steps, args.active_steps, args.cooldown_steps
    total = warmup + active + cooldown

    # Extend mp's iter range past `active` to cover cooldown too. Why: prof.sh
    # runs `nsys profile --kill 9 -c cudaProfilerApi`, which SIGKILLs python the
    # instant cudaProfilerStop fires. mp.prof_iter(step==stop_iter+1) auto-fires
    # prof_stop, so if stop_iter ends inside the loop we get killed before
    # write_summary runs. By setting stop_iter = total-1, the in-loop auto-stop
    # never triggers; main() will call mp.prof_stop() *after* writing the JSON.
    mp.set_iter_range(warmup, total - 1)

    step_times = []
    for step in range(total):
        mp.prof_iter(step)
        t0 = time.perf_counter()
        with prof_range("step_total"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
                loss = out["action_loss"]
            with prof_range("backward"):
                accelerator.backward(loss)
            with prof_range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        if step % 5 == 0:
            print(f"[bench] step {step:3d} loss={loss.item():.4f} "
                  f"dt={step_times[-1]*1000:.1f}ms")

    # NOTE: do NOT call mp.prof_stop() here. The caller (main) must write
    # summary.json first; only then is it safe to stop, since cudaProfilerStop
    # under nsys --kill 9 will immediately SIGKILL this process.
    return step_times


def run_loop_torch(model, optimizer, batch, args, hooks_target, accelerator):
    """Loop with only torch.profiler — no model_prof / no outer nsys wrap.

    `accelerator` required for the same reason as `run_loop` — backward
    must route through DeepSpeed engine via `accelerator.backward(loss)`.
    """
    if hooks_target is not None:
        register_module_hooks(model, hooks_target,
                              also_backward=getattr(args, "detailed_module_hooks", False))

    warmup, active, cooldown = args.warmup_steps, args.active_steps, args.cooldown_steps
    total = warmup + active + cooldown

    tb_dir = Path(args.output_dir) / f"tb_trace_{args.head}_torch"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Keep the trace small: a few active steps, no stack frames, no memory events.
    # Shape info stays on; both CPU + CUDA activities are explicit so we don't
    # silently lose either if PyTorch changes its defaults.
    sched = torch.profiler.schedule(
        wait=0, warmup=warmup, active=active, repeat=1,
    )

    step_times = []
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=sched,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(tb_dir)),
        record_shapes=True,
        profile_memory=False,
        with_stack=args.with_stack,
        with_modules=args.with_stack,
    ) as tp_prof:
        for step in range(total):
            t0 = time.perf_counter()
            with prof_range("step_total"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch)
                    loss = out["action_loss"]
                with prof_range("backward"):
                    accelerator.backward(loss)
                with prof_range("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            tp_prof.step()
            if step % 5 == 0:
                print(f"[bench] step {step:3d} loss={loss.item():.4f} "
                      f"dt={step_times[-1]*1000:.1f}ms")
    return step_times


def compute_input_stats(batch) -> dict:
    """Best-effort inspection of the frozen batch to record sequence shapes."""
    stats = {}
    try:
        # batch is list[dict] per starVLA convention
        first = batch[0] if isinstance(batch, (list, tuple)) else batch
        if isinstance(first, dict):
            if "image" in first:
                img = first["image"]
                if hasattr(img, "__len__"):
                    stats["n_images_per_sample"] = len(img)
            if "lang" in first:
                stats["lang_chars_per_sample"] = len(first["lang"])
            if "action" in first:
                a = first["action"]
                stats["action_shape"] = list(getattr(a, "shape", []))
        stats["batch_len"] = len(batch) if hasattr(batch, "__len__") else None
    except Exception as e:
        stats["error"] = repr(e)
    return stats


def write_summary(args, cfg, step_times, batch, out_path: Path):
    """Roll the run into a JSON file for cross-hardware comparison."""
    warmup, active = args.warmup_steps, args.active_steps
    traced = step_times[warmup:warmup + active]
    mean_total_ms = (sum(traced) / max(len(traced), 1)) * 1000

    peak_alloc = peak_reserved = 0.0
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    summary = {
        "head": args.head,
        "framework_name": cfg.framework.name,
        "backbone": cfg.framework.qwenvl.base_vlm,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "per_device_batch_size": int(cfg.datasets.vla_data.per_device_batch_size),
        "step_count": {
            "warmup": args.warmup_steps,
            "traced": args.active_steps,
            "cooldown": args.cooldown_steps,
        },
        "input_stats": compute_input_stats(batch),
        "timings_ms": {
            "step_total_mean_traced": mean_total_ms,
            # Per-range means are computed offline from the torch.profiler trace;
            # this JSON is the high-level wall-clock summary.
            "per_step_ms_all_steps": [t * 1000 for t in step_times],
        },
        "memory_gb": {
            "peak_allocated": peak_alloc,
            "peak_reserved": peak_reserved,
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[bench] wrote summary -> {out_path}")


def ensure_dist_initialized():
    """Init a 1-process dist group when bench is launched as plain `python`.

    starVLA's dataloader calls dist.get_rank() unconditionally. Under
    `accelerate launch`, dist is already initialized — this function is a no-op.
    Under plain `python bench.py`, we synthesize a single-rank group so the
    dataloader can proceed.
    """
    import torch.distributed as dist
    if dist.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def main():
    args = parse_args()
    if args.selftest_hooks:
        _selftest_hooks()
        return 0
    cfg = load_config(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # starVLA's dataloader writes dataset_statistics.json to cfg.output_dir on rank 0.
    cfg.output_dir = args.output_dir

    set_global_seed(42)
    _apply_tokenizer_file_patch()
    ensure_dist_initialized()
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
    if args.batch_size_override and hasattr(batch, '__len__'):
        batch = batch[:args.batch_size_override]
        print(f"[bench] sliced batch to {args.batch_size_override} (--batch_size_override)")
    print(f"[bench] batch type = {type(batch).__name__}, len = {len(batch) if hasattr(batch, '__len__') else '?'}")

    accelerator = setup_accelerator(int(cfg.datasets.vla_data.per_device_batch_size))
    optimizer = setup_optimizer(model)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Resolve hook targets after .prepare() may have wrapped the model
    inner = accelerator.unwrap_model(model)
    hooks_target = {
        "vlm_forward": inner.qwen_vl_interface,
        "action_head_forward": inner.action_model,
    }
    if args.detailed_module_hooks:
        # Add per-leaf-module hooks for fine-grained kernel attribution.
        # Walk under qwen_vl_interface (where ~all the FLOPs are) so we don't
        # double-count the wrappers we already hook above.
        leaf_targets = collect_leaf_hook_targets(inner.qwen_vl_interface)
        # Namespace under qwen_vl_interface in the label so paths are reproducible.
        hooks_target.update({f"vlm.{k}": v for k, v in leaf_targets.items()})
        if accelerator.is_main_process:
            print(f"[bench] detailed_module_hooks: registering {len(leaf_targets)} extra hooks "
                  f"on qwen_vl_interface leaves")

    total_steps = args.warmup_steps + args.active_steps + args.cooldown_steps
    if args.profiler == "none":
        if accelerator.is_main_process:
            print(f"[bench] starting plain loop for {total_steps} steps")
        run_loop(model, optimizer, batch, total_steps=total_steps, hooks_target=None, accelerator=accelerator)
        if accelerator.is_main_process:
            print("[bench] loop done (no profile)")
        accelerator.wait_for_everyone()
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0

    if accelerator.is_main_process:
        print(f"[bench] starting {args.profiler}-profiled loop for {total_steps} steps "
              f"(warmup={args.warmup_steps}, active={args.active_steps}, cooldown={args.cooldown_steps})")
    if args.profiler == "nsys":
        step_times = run_loop_nsys(model, optimizer, batch, args, hooks_target, accelerator)
    else:  # "torch"
        step_times = run_loop_torch(model, optimizer, batch, args, hooks_target, accelerator)
    mean_traced = sum(step_times[args.warmup_steps:args.warmup_steps+args.active_steps]) / max(args.active_steps, 1)

    if accelerator.is_main_process:
        print(f"[bench] loop done. mean traced step = {mean_traced*1000:.1f}ms")
        summary_path = Path(args.output_dir) / f"bench_{args.head}_{args.profiler}_summary.json"
        write_summary(args, cfg, step_times, batch, summary_path)

    # Sync, tear down NCCL, *then* stop the profiler. After mp.prof_stop()
    # nsys --kill 9 will SIGKILL this process, so anything we want to do
    # cleanly must happen before this line.
    accelerator.wait_for_everyone()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

    if args.profiler == "nsys" and accelerator.is_main_process:
        import model_prof as mp
        mp.prof_stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())