#!/bin/bash
set -euo pipefail

source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate repa-sd3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export DATADIR="${DATADIR:-/inspire/hdd/project/chineseculture/public/yuxuan/datasets/}"
export MODEL_DIR="${MODEL_DIR:-/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3}"
export INCEPTION_WEIGHTS_PATH="${INCEPTION_WEIGHTS_PATH:-/inspire/hdd/project/chineseculture/public/yuxuan/base_models/torchvision/inception_v3/inception_v3_google-0cc3c7bd.pth}"

export NUM_PROMPTS="${NUM_PROMPTS:-100}"
export NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-8}"
export HEIGHT="${HEIGHT:-1024}"
export WIDTH="${WIDTH:-1024}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
export NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
export FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"

unset RESIDUAL_TARGET_LAYERS
unset RESIDUAL_ORIGIN_LAYER
unset RESIDUAL_ORIGIN_LAYERS
unset RESIDUAL_WEIGHTS
unset RESIDUAL_WEIGHTS_PATH
unset RESIDUAL_PROCRUSTES_PATH

export RESIDUAL_USE_LAYERNORM="${RESIDUAL_USE_LAYERNORM:-1}"
export TIMESTEP_RESIDUAL_WEIGHT_FN="${TIMESTEP_RESIDUAL_WEIGHT_FN:-constant}"
export TIMESTEP_RESIDUAL_WEIGHT_POWER="${TIMESTEP_RESIDUAL_WEIGHT_POWER:-1.0}"
export TIMESTEP_RESIDUAL_WEIGHT_EXP_ALPHA="${TIMESTEP_RESIDUAL_WEIGHT_EXP_ALPHA:-1.5}"
export TIMESTEP_STAGE="${TIMESTEP_STAGE:-0}"

BASE_OUTPUT_PREFIX="${BASE_OUTPUT_PREFIX:-$SCRIPT_DIR/logs/conditional_vendi_coco5k_base}"

RUN_LABELS=(
    "3"
    "4"
)

RUN_SEED_GROUPS=(
    "72 73 74 75 76 77 78 79"
    "82 83 84 85 86 87 88 89"
)

if (( ${#RUN_LABELS[@]} != ${#RUN_SEED_GROUPS[@]} )); then
    echo "ERROR: RUN_LABELS and RUN_SEED_GROUPS must have the same length." >&2
    exit 1
fi

for run_idx in "${!RUN_LABELS[@]}"; do
    run_label="${RUN_LABELS[$run_idx]}"
    run_seeds="${RUN_SEED_GROUPS[$run_idx]}"
    run_output_dir="${BASE_OUTPUT_PREFIX}-${run_label}"

    read -r -a seed_array <<< "$run_seeds"
    if (( ${#seed_array[@]} < NUM_IMAGES_PER_PROMPT )); then
        echo "ERROR: seed group ${run_label} only has ${#seed_array[@]} seeds, fewer than NUM_IMAGES_PER_PROMPT=${NUM_IMAGES_PER_PROMPT}." >&2
        exit 1
    fi

    echo "====================================================="
    echo "Running conditional Vendi evaluation without residual"
    echo "  RUN_LABEL             : ${run_label}"
    echo "  DATADIR               : ${DATADIR}"
    echo "  OUTPUT_DIR            : ${run_output_dir}"
    echo "  INCEPTION_WEIGHTS_PATH: ${INCEPTION_WEIGHTS_PATH}"
    echo "  NUM_PROMPTS           : ${NUM_PROMPTS}"
    echo "  NUM_IMAGES_PER_PROMPT : ${NUM_IMAGES_PER_PROMPT}"
    echo "  SEEDS                 : ${run_seeds}"
    echo "  CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES}"
    echo "====================================================="

    OUTPUT_DIR="${run_output_dir}" \
    SEEDS="${run_seeds}" \
    bash "$SCRIPT_DIR/evaluate_sd3_conditional_vendi.sh"
done
