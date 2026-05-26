#!/bin/bash
# examples/profiling/qwen_action_head_bench/run.sh
# Loops OFT/PI/FAST, invokes model_prof's prof.sh on rank 0 only (via bench_wrap.py).
# Auto-routes to nsys (GPU) or asys (PPU).
set -euo pipefail

# === Modify these to your environment ===
PROF_DIR="${PROF_DIR:-/mnt/ssd/alilab/Model_Center_Pipeline/model_prof}"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/Qwen3.5-0.8B}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
OUT_DIR="${OUT_DIR:-./out_profile}"
NUM_GPUS="${NUM_GPUS:-8}"
HEADS="${HEADS:-OFT PI FAST}"
WARMUP="${WARMUP:-30}"
ACTIVE="${ACTIVE:-10}"
COOLDOWN="${COOLDOWN:-5}"
# === End of environment ===

mkdir -p "${OUT_DIR}"

export MODEL_PROFILE=1
export MODEL_PROFILE_RANGE="${WARMUP},$((WARMUP + ACTIVE - 1))"

if [ ! -x "${PROF_DIR}/model_prof/tool/prof.sh" ]; then
    echo "ERROR: prof.sh not found at ${PROF_DIR}/model_prof/tool/prof.sh"
    echo "       Install model_prof first: pip install -e <model_prof repo>"
    exit 1
fi

for HEAD in ${HEADS}; do
    REPORT_PREFIX="${OUT_DIR}/bench_${HEAD}"
    echo "============================================================"
    echo "[run.sh] profiling head=${HEAD} (rank-0 only) -> ${REPORT_PREFIX}"
    echo "============================================================"

    PROF_DIR="${PROF_DIR}" REPORT_PREFIX="${REPORT_PREFIX}" \
        accelerate launch \
            --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
            --num_processes "${NUM_GPUS}" \
            examples/profiling/qwen_action_head_bench/bench_wrap.py \
            --head "${HEAD}" \
            --base_vlm "${BASE_VLM}" \
            --data_root "${DATA_ROOT}" \
            --output_dir "${OUT_DIR}" \
            --warmup_steps "${WARMUP}" \
            --active_steps "${ACTIVE}" \
            --cooldown_steps "${COOLDOWN}"
done

echo "[run.sh] all done. Outputs in ${OUT_DIR}/"
ls -lh "${OUT_DIR}"
