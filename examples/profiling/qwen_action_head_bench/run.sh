#!/bin/bash
# examples/profiling/qwen_action_head_bench/run.sh
# Loops OFT/PI/FAST, invokes model_prof's prof.sh on rank 0 only (via bench_wrap.py).
# Auto-routes to nsys (GPU) or asys (PPU).
set -euo pipefail

# === Modify these to your environment ===
PROF_DIR="${PROF_DIR:-/mnt/ssd/alilab/Model_Center_Pipeline/model_prof}"
BASE_VLM="${BASE_VLM:-/mnt/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3.5-0.8B}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
OUT_DIR="${OUT_DIR:-./out_profile}"
NUM_GPUS="${NUM_GPUS:-8}"
HEADS="${HEADS:-OFT PI}"
WARMUP="${WARMUP:-30}"
ACTIVE="${ACTIVE:-10}"
COOLDOWN="${COOLDOWN:-5}"
# torch.profiler trace size grows linearly with active. Senior review: keep it
# to 1-2 steps. 2 lets us check whether the first traced step is anomalously
# slow without blowing up the .pt.trace.json.gz size.
ACTIVE_TORCH="${ACTIVE_TORCH:-2}"
# Per-run sub-directory so re-runs don't clobber prior artifacts.
# Override with RUN_TAG=foo to land in out_profile/foo/ instead.
RUN_TAG="${RUN_TAG:-run_$(date +%Y%m%d_%H%M%S)}"
# === End of environment ===

RUN_DIR="${OUT_DIR}/${RUN_TAG}"
mkdir -p "${RUN_DIR}"
# Cache the frozen batch outside RUN_DIR so it's reused across runs (it's a
# dataloader snapshot, not a profile artifact).
BATCH_CACHE="${OUT_DIR}/fixed_batch.pt"

# Snapshot run-level metadata that's NOT already in bench_*_summary.json or
# bench_*.log (head/world_size/batch/step_count/timings/memory live in summary;
# per-step loss + framework class + accelerate launcher output live in the log).
# Best-effort: missing tools (nvidia-smi on PPU, git outside a repo) -> "unknown".
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
# Treat untracked files as dirty too — `git diff --quiet HEAD` ignores them.
if [ -z "$(git status --porcelain 2>/dev/null)" ]; then
    GIT_DIRTY="false"
else
    GIT_DIRTY="true"
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1 || echo unknown)"

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
HOSTNAME_="$(hostname)" \
GPU_NAME="${GPU_NAME}" \
GIT_SHA="${GIT_SHA}" \
GIT_DIRTY="${GIT_DIRTY}" \
HEADS_PLANNED="${HEADS}" \
NUM_GPUS_="${NUM_GPUS}" \
BASE_VLM_="${BASE_VLM}" \
DATA_ROOT_="${DATA_ROOT}" \
PROF_DIR_="${PROF_DIR}" \
BATCH_CACHE_="${BATCH_CACHE}" \
python3 -c '
import json, os, sys
meta = {
    "started_utc": os.environ["STARTED_UTC"],
    "hostname": os.environ["HOSTNAME_"],
    "gpu": os.environ["GPU_NAME"],
    "git_sha": os.environ["GIT_SHA"],
    "git_dirty": os.environ["GIT_DIRTY"] == "true",
    "heads_planned": os.environ["HEADS_PLANNED"].split(),
    "num_gpus": int(os.environ["NUM_GPUS_"]),
    "base_vlm": os.environ["BASE_VLM_"],
    "data_root": os.environ["DATA_ROOT_"],
    "prof_dir": os.environ["PROF_DIR_"],
    "batch_cache": os.environ["BATCH_CACHE_"],
}
json.dump(meta, sys.stdout, indent=2)
sys.stdout.write("\n")
' > "${RUN_DIR}/run_meta.json"

export MODEL_PROFILE=1
export MODEL_PROFILE_RANGE="${WARMUP},$((WARMUP + ACTIVE - 1))"
# Skip prof.sh's auto CSV generation. Its `nsys stats --report gputrace` invocation
# is incompatible with newer nsys (CUDA 12.x renamed the report). The .nsys-rep
# is still produced and openable in Nsight Systems. Set MODEL_PROFILE_SKIP_GENERATE_TRACE_REPORT=0
# to opt back in if your model_prof / nsys versions are aligned.
export MODEL_PROFILE_SKIP_GENERATE_TRACE_REPORT="${MODEL_PROFILE_SKIP_GENERATE_TRACE_REPORT:-1}"

if [ ! -x "${PROF_DIR}/model_prof/tool/prof.sh" ]; then
    echo "ERROR: prof.sh not found at ${PROF_DIR}/model_prof/tool/prof.sh"
    echo "       Install model_prof first: pip install -e <model_prof repo>"
    exit 1
fi

for HEAD in ${HEADS}; do
    echo "============================================================"
    echo "[run.sh] head=${HEAD}: two passes (nsys, then torch.profiler)"
    echo "============================================================"

    # Pass A: nsys / asys via prof.sh (rank-0 only). bench.py runs in --profiler nsys
    # mode so torch.profiler is OFF — running both at once doubles CUPTI
    # subscribers and pollutes per-step timing.
    REPORT_PREFIX_NSYS="${RUN_DIR}/bench_${HEAD}_nsys"
    echo "[run.sh]  -> pass A (nsys)  out=${REPORT_PREFIX_NSYS}"
    PROF_DIR="${PROF_DIR}" REPORT_PREFIX="${REPORT_PREFIX_NSYS}" \
        accelerate launch \
            --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
            --num_processes "${NUM_GPUS}" \
            examples/profiling/qwen_action_head_bench/bench_wrap.py \
            --head "${HEAD}" \
            --base_vlm "${BASE_VLM}" \
            --data_root "${DATA_ROOT}" \
            --output_dir "${RUN_DIR}" \
            --batch_path "${BATCH_CACHE}" \
            --warmup_steps "${WARMUP}" \
            --active_steps "${ACTIVE}" \
            --cooldown_steps "${COOLDOWN}" \
            --profiler nsys \
        2>&1 | tee "${REPORT_PREFIX_NSYS}.log"

    # Pass B: torch.profiler only. Skip bench_wrap entirely — no prof.sh / nsys
    # on rank 0 — so torch.profiler sees a clean run. Fewer active steps to keep
    # the trace small per senior review.
    REPORT_PREFIX_TORCH="${RUN_DIR}/bench_${HEAD}_torch"
    echo "[run.sh]  -> pass B (torch) out=${REPORT_PREFIX_TORCH} (active=${ACTIVE_TORCH})"
    accelerate launch \
            --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
            --num_processes "${NUM_GPUS}" \
            examples/profiling/qwen_action_head_bench/bench.py \
            --head "${HEAD}" \
            --base_vlm "${BASE_VLM}" \
            --data_root "${DATA_ROOT}" \
            --output_dir "${RUN_DIR}" \
            --batch_path "${BATCH_CACHE}" \
            --warmup_steps "${WARMUP}" \
            --active_steps "${ACTIVE_TORCH}" \
            --cooldown_steps "${COOLDOWN}" \
            --profiler torch \
        2>&1 | tee "${REPORT_PREFIX_TORCH}.log"
done

echo "[run.sh] all done. Outputs in ${RUN_DIR}/"
ls -lh "${RUN_DIR}"
