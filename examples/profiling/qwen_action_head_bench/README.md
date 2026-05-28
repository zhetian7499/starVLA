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

**Bench Python code is GPU/PPU agnostic — no source edits needed.** All
compatibility workarounds (DeepSpeed conditional init, `accelerator.backward`
for ZeRO, deferred `mp.prof_stop`, `tokenizer_file` patch for transformers 5.x)
are already baked in. The PPU operator only needs to: (a) set 4 paths,
(b) make one symlink for the FAST tokenizer, (c) bring the GPU-side
`fixed_batch.pt` over to guarantee identical input.

### Step 1 — Sync this directory and the frozen batch from the GPU side

```bash
# On the operator's laptop / GPU box (whoever has both endpoints)
rsync -av <gpu_host>:/path/to/starVLA/examples/profiling/qwen_action_head_bench/ \
    <ppu_host>:/path/to/starVLA/examples/profiling/qwen_action_head_bench/

# CRITICAL: bring the exact same frozen batch over — otherwise PPU is
# benching a different input distribution and the numbers are not comparable.
rsync -av <gpu_host>:/path/to/starVLA/out_profile/fixed_batch.pt \
    <ppu_host>:/path/to/starVLA/out_profile/fixed_batch.pt
```

### Step 2 — Resolve four paths on the PPU box

Find these four locations on PPU and remember them:

| What | Used as | Default in `run.sh` |
|---|---|---|
| `model_prof` install root | `PROF_DIR` env | `/mnt/ssd/alilab/Model_Center_Pipeline/model_prof` |
| Qwen3.5-0.8B weights dir | `BASE_VLM` env | `playground/Pretrained_models/Qwen3.5-0.8B` |
| LIBERO LeRobot data dir | `DATA_ROOT` env | `playground/Datasets/LEROBOT_LIBERO_DATA` |
| `physical-intelligence/fast` tokenizer dir | symlink target (see Step 3) | — |

### Step 3 — Symlink the FAST tokenizer to where starVLA's `fast_ActionHeader` hard-codes

starVLA's `fast_ActionHeader.py` hard-codes the path `playground/Pretrained_models/fast`.
Don't fight it; symlink:

```bash
cd /path/to/starVLA
mkdir -p playground/Pretrained_models
ln -sfn /your/ppu/path/to/physical-intelligence/fast \
        playground/Pretrained_models/fast
ls playground/Pretrained_models/fast/   # sanity: should list tokenizer.json etc.
```

If `physical-intelligence/fast` isn't already on the PPU box, download it from
HuggingFace (model id `physical-intelligence/fast`) — it's small (~700 KB).

### Step 4 — (FAST head only) Augment the Qwen vocab with action tokens

starVLA's QwenFast head maps fast-tokenizer ids to `<robot_action_N>` tokens in
the LLM vocab. The base Qwen3.5-0.8B doesn't have these tokens; without them
labels get masked to -100, CE loss is `None`, fallback is a non-grad
`tensor(0.0)`, and backward fails with "element 0 of tensors does not require
grad". This step adds the 2048 `<robot_action_N>` tokens and resizes the
embedding matrix.

starVLA ships `add_special_tokens_to_qwen.py` for this, but it hard-codes
`Qwen3VLForConditionalGeneration` (fails on Qwen3.5) and saves the processor
after the augmented tokenizer (clobbers it). We provide a small replacement
in this directory:

```bash
cd /path/to/starVLA
python examples/profiling/qwen_action_head_bench/preprocess_qwen_action_tokens.py \
    --source <ppu_qwen_path> \
    --dest playground/Pretrained_models/Qwen3.5-0.8B-Action \
    --tokens-file starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt
```

Output is ~3 GB. Then point the FAST head's `BASE_VLM` at the augmented dir
(OFT and PI continue to use the original Qwen — they don't need action tokens).

### Step 5 — Confirm `asys` is on PATH

```bash
which asys && asys --version | head -1
```

If absent, source the PPU SDK env (PPU vendor specific) before running.

### Step 6 — Run

OFT and PI use the original Qwen; FAST uses the action-augmented Qwen from Step 4:

```bash
cd /path/to/starVLA
rm -rf out_profile/bench_*   # clean any prior outputs

# Round 1: OFT + PI on original Qwen
HEADS="OFT PI" \
PROF_DIR=<ppu_model_prof_path> \
BASE_VLM=<ppu_qwen_path> \
DATA_ROOT=<ppu_libero_path> \
NUM_GPUS=<ppu_card_count> \
    bash examples/profiling/qwen_action_head_bench/run.sh

# Round 2: FAST on the augmented Qwen
HEADS=FAST \
PROF_DIR=<ppu_model_prof_path> \
BASE_VLM=playground/Pretrained_models/Qwen3.5-0.8B-Action \
DATA_ROOT=<ppu_libero_path> \
NUM_GPUS=<ppu_card_count> \
    bash examples/profiling/qwen_action_head_bench/run.sh
```

Defaults are 30 warmup + 10 active + 5 cooldown. Each head ~5-8 min on H20-8; budget similar on PPU.

### Step 7 — Send results back

The outputs mirror the GPU side, just with `.asysrep` instead of `.nsys-rep`
and `_ppu*.csv` instead of `_gputrace*.csv`:

```
out_profile/
├── bench_OFT.asysrep           ⟷  bench_OFT.nsys-rep on GPU
├── bench_OFT_ppu.csv           ⟷  bench_OFT_gputrace.csv
├── bench_OFT_ppu_sum.csv       ⟷  bench_OFT_gputrace_sum.csv
├── bench_OFT_summary.json      same schema both sides
├── tb_trace_OFT/               same (torch.profiler chrome trace)
├── (same for PI and FAST)
└── fixed_batch.pt              must match GPU's byte-for-byte
```

Send the whole `out_profile/` directory back. The GPU-side operator does
side-by-side comparison from `*_summary.json` and the trace files.

### Things that do NOT need to change for PPU

- `bench.py`, `bench_wrap.py`, `run.sh` — **all unmodified**
- `WARMUP/ACTIVE/COOLDOWN` step counts — keep identical for valid comparison
- `seed` (hardcoded `42` in bench.py)
- `per_device_batch_size` (from LIBERO config, 16)
- The PreTrainedTokenizerFast monkey-patch in `bench.py` is idempotent and only
  fires when `tokenizer.json` is present next to the model dir — harmless on
  PPU even if their transformers version doesn't need it.

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
