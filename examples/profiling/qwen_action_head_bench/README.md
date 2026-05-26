# Qwen × Action Head Profiling Bench

Self-contained micro-benchmark for comparing how Qwen3.5-0.8B × {OFT, PI, FAST}
action heads spend compute on real LIBERO data. Produces both `torch.profiler`
and `nsys`/`asys` traces for cross-hardware (GPU vs PPU) comparison.

See design doc: `docs/superpowers/specs/2026-05-26-qwen-action-head-profiling-design.md`

## Quick reference

Three files in this directory:

- `bench.py` — single-process / accelerate-launched benchmark runner
- `run.sh`   — loops OFT/PI/FAST via `model_prof`'s `prof.sh` (GPU/PPU agnostic)
- `README.md` — this file

## End-to-end workflow

### 1. Local edit

Edit `bench.py` / `run.sh` on your laptop. No code execution needed locally
(except `python bench.py --selftest_hooks` for the hook self-test).

### 2. Sync to server

```bash
rsync -av examples/profiling/qwen_action_head_bench/ \
    user@server:/path/to/starVLA/examples/profiling/qwen_action_head_bench/
```

### 3. One-time server setup

```bash
ssh user@server
cd /path/to/starVLA

# Install model_prof (provides prof.sh wrapper for nsys/asys auto-routing)
pip install -e /mnt/ssd/alilab/Model_Center_Pipeline/model_prof

# Confirm nsys is on PATH (GPU node) or asys (PPU node)
which nsys || which asys
```

### 4. Run

Sweep all three heads:

```bash
cd /path/to/starVLA
bash examples/profiling/qwen_action_head_bench/run.sh
```

Or just one head:

```bash
HEADS=OFT bash examples/profiling/qwen_action_head_bench/run.sh
```

### 5. Fetch results back

```bash
rsync -av user@server:/path/to/starVLA/out_profile/ ./out_profile/
```

## Outputs (per head)

```
out_profile/
├── bench_OFT.nsys-rep              # nsys (GPU) or .asysrep (PPU)
├── bench_OFT_gputrace.csv          # nsys kernel list (or _ppu.csv on PPU)
├── bench_OFT_gputrace_sum.csv      # aggregated kernel summary
├── bench_OFT_summary.json          # bench.py rollup: per-step time, peak mem
├── tb_trace_OFT/                   # torch.profiler chrome trace
└── fixed_batch.pt                  # frozen LIBERO batch, identical across heads
```

> nsys/asys trace and CSVs are captured on rank 0 only. NCCL kernels participated in by rank 0 are still visible in the trace, so the cross-hardware comparison is still meaningful.

### Reading the JSON summary

```bash
jq '.timings_ms.step_total_mean_traced, .memory_gb' out_profile/bench_OFT_summary.json
```

### Viewing the torch.profiler trace

```bash
# Locally, after rsync:
pip install tensorboard torch-tb-profiler
tensorboard --logdir ./out_profile/
# Open the PYTORCH_PROFILER tab in the browser
```

### Viewing the nsys trace

Open `.nsys-rep` in Nsight Systems UI (also runs on macOS).

## Environment knobs (override before `run.sh`)

| Variable | Default | Purpose |
|---|---|---|
| `PROF_DIR` | `/mnt/ssd/alilab/Model_Center_Pipeline/model_prof` | Where `model_prof` is installed |
| `BASE_VLM` | `playground/Pretrained_models/Qwen3.5-0.8B` | Qwen3.5 weight directory |
| `DATA_ROOT` | `playground/Datasets/LEROBOT_LIBERO_DATA` | LIBERO data root |
| `OUT_DIR` | `./out_profile` | Output directory |
| `NUM_GPUS` | `8` | Number of processes for accelerate launch |
| `HEADS` | `OFT PI FAST` | Space-separated list of heads to bench |
| `WARMUP` | `30` | Untraced warmup steps |
| `ACTIVE` | `10` | Steps captured by torch.profiler + model_prof |
| `COOLDOWN` | `5` | Trailing untraced steps |

## Reproducibility on PPU

Copy the entire directory + `fixed_batch.pt` to the PPU box. `run.sh` will
auto-detect `asys` instead of `nsys`. Use the exact same `WARMUP/ACTIVE/COOLDOWN`,
`NUM_GPUS`, and `--batch_path` pointing at the rsync'd `fixed_batch.pt` to
ensure the workload is byte-identical.

## Troubleshooting

- **`AttributeError: 'QwenXXX' object has no attribute 'qwen_vl_interface'`**
  The framework's attribute name diverged. Open
  `starVLA/model/framework/VLM4A/Qwen<head>.py`, find the actual name, update
  `build_model()` in `bench.py`.
- **`prof.sh: Neither asys nor nsys found`**
  Profiler not on PATH. On GPU: `module load cuda` or check Nsight Systems
  install. On PPU: source the PPU SDK environment.
- **NCCL hang on accelerate launch**
  Same env vars as `examples/LIBERO/train_files/run_libero_train.sh`:
  ```bash
  export NCCL_SOCKET_IFNAME=bond0
  export NCCL_IB_HCA=mlx5_2,mlx5_3
  ```
- **`Failed to connect to Agent` warnings spam the log on multi-GPU runs**
  This means nsys is trying to attach to multiple ranks. Confirm `run.sh` is
  pointing accelerate at `bench_wrap.py` (not `bench.py` directly) — the
  wrapper ensures only rank 0 goes through `prof.sh`.
- **`ERROR: Report 'gputrace' could not be found`**
  Same root cause as the warning above: the `.nsys-rep` is missing or corrupt
  because multiple ranks raced to write it. Fix is the same — ensure
  `bench_wrap.py` is in the launch path.
