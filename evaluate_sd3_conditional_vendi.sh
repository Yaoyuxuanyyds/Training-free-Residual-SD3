#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0,1 bash evaluate_sd3_conditional_vendi.sh

Environment overrides:
  DATADIR                   COCO root directory. Must contain coco/annotations/captions_val2017.json
  MODEL_DIR                 Base SD3 model directory
  LOAD_CKPT_PATH            Optional finetuned transformer checkpoint
  OUTPUT_DIR                Output directory for generated images and Vendi results
  NUM_PROMPTS               Number of COCO prompts to evaluate
  NUM_IMAGES_PER_PROMPT     Number of seeds/images per prompt
  SEEDS                     Space-separated seed list, e.g. "42 43 44 45 46"
  HEIGHT, WIDTH             Image size
  NUM_INFERENCE_STEPS       Sampling steps
  GUIDANCE_SCALE            CFG scale
  NEGATIVE_PROMPT           Negative prompt string
  GENERATION_BATCH_SIZE     Batch size for per-prompt seed generation
  FEATURE_BATCH_SIZE        Batch size for Inception feature extraction
  INCEPTION_WEIGHTS_PATH    Local Inception-V3 weights path for offline feature extraction
  RESIDUAL_TARGET_LAYERS    Space-separated residual target layers
  RESIDUAL_ORIGIN_LAYER     Single residual origin layer
  RESIDUAL_ORIGIN_LAYERS    Space-separated residual origin layers
  RESIDUAL_WEIGHTS          Space-separated residual weights
  RESIDUAL_WEIGHTS_PATH     Path to saved residual weights
  RESIDUAL_PROCRUSTES_PATH  Path to Procrustes rotations
EOF
  exit 0
fi

GPUS_CSV="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS_CSV"
WORLD_SIZE="${#GPU_ARRAY[@]}"

DATADIR="${DATADIR:-/inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench/examples/dataset}"
MODEL_DIR="${MODEL_DIR:-/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3}"
LOAD_CKPT_PATH="${LOAD_CKPT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/logs/conditional_vendi_coco5k}"

NUM_PROMPTS="${NUM_PROMPTS:-100}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-5}"
SEEDS="${SEEDS:-42 43 44 45 46}"

HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
INCEPTION_WEIGHTS_PATH="${INCEPTION_WEIGHTS_PATH:-/inspire/hdd/project/chineseculture/public/yuxuan/base_models/torchvision/inception_v3/inception_v3_google-0cc3c7bd.pth}"

RESIDUAL_TARGET_LAYERS="${RESIDUAL_TARGET_LAYERS:-}"
RESIDUAL_ORIGIN_LAYER="${RESIDUAL_ORIGIN_LAYER:-}"
RESIDUAL_ORIGIN_LAYERS="${RESIDUAL_ORIGIN_LAYERS:-}"
RESIDUAL_WEIGHTS="${RESIDUAL_WEIGHTS:-}"
RESIDUAL_WEIGHTS_PATH="${RESIDUAL_WEIGHTS_PATH:-}"
RESIDUAL_PROCRUSTES_PATH="${RESIDUAL_PROCRUSTES_PATH:-}"
RESIDUAL_USE_LAYERNORM="${RESIDUAL_USE_LAYERNORM:-1}"
TIMESTEP_RESIDUAL_WEIGHT_FN="${TIMESTEP_RESIDUAL_WEIGHT_FN:-constant}"
TIMESTEP_RESIDUAL_WEIGHT_POWER="${TIMESTEP_RESIDUAL_WEIGHT_POWER:-1.0}"
TIMESTEP_RESIDUAL_WEIGHT_EXP_ALPHA="${TIMESTEP_RESIDUAL_WEIGHT_EXP_ALPHA:-1.5}"
TIMESTEP_STAGE="${TIMESTEP_STAGE:-0}"

mkdir -p "$OUTPUT_DIR"

if [[ "$DATADIR" == "/path/to/your/data" ]]; then
  echo "Please set DATADIR before running this script."
  exit 1
fi

COMMON_ARGS=(
  --datadir "$DATADIR"
  --model_dir "$MODEL_DIR"
  --output_dir "$OUTPUT_DIR"
  --num_prompts "$NUM_PROMPTS"
  --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
  --seeds $SEEDS
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --negative_prompt "$NEGATIVE_PROMPT"
  --generation_batch_size "$GENERATION_BATCH_SIZE"
  --feature_batch_size "$FEATURE_BATCH_SIZE"
  --inception_weights_path "$INCEPTION_WEIGHTS_PATH"
  --world_size "$WORLD_SIZE"
  --timestep_residual_weight_fn "$TIMESTEP_RESIDUAL_WEIGHT_FN"
  --timestep_residual_weight_power "$TIMESTEP_RESIDUAL_WEIGHT_POWER"
  --timestep_residual_weight_exp_alpha "$TIMESTEP_RESIDUAL_WEIGHT_EXP_ALPHA"
  --timestep_stage "$TIMESTEP_STAGE"
  --residual_use_layernorm "$RESIDUAL_USE_LAYERNORM"
)

if [[ -n "$LOAD_CKPT_PATH" ]]; then
  COMMON_ARGS+=(--load_ckpt_path "$LOAD_CKPT_PATH")
fi

if [[ -n "$RESIDUAL_TARGET_LAYERS" ]]; then
  COMMON_ARGS+=(--residual_target_layers $RESIDUAL_TARGET_LAYERS)
fi

if [[ -n "$RESIDUAL_ORIGIN_LAYER" ]]; then
  COMMON_ARGS+=(--residual_origin_layer "$RESIDUAL_ORIGIN_LAYER")
fi

if [[ -n "$RESIDUAL_ORIGIN_LAYERS" ]]; then
  COMMON_ARGS+=(--residual_origin_layers $RESIDUAL_ORIGIN_LAYERS)
fi

if [[ -n "$RESIDUAL_WEIGHTS" ]]; then
  COMMON_ARGS+=(--residual_weights $RESIDUAL_WEIGHTS)
fi

if [[ -n "$RESIDUAL_WEIGHTS_PATH" ]]; then
  COMMON_ARGS+=(--residual_weights_path "$RESIDUAL_WEIGHTS_PATH")
fi

if [[ -n "$RESIDUAL_PROCRUSTES_PATH" ]]; then
  COMMON_ARGS+=(--residual_procrustes_path "$RESIDUAL_PROCRUSTES_PATH")
fi

echo "Output dir: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $GPUS_CSV"
echo "World size: $WORLD_SIZE"

pids=()
for rank in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$rank]}"
  echo "Launching rank $rank on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_sd3_conditional_vendi.py" \
    "${COMMON_ARGS[@]}" \
    --rank "$rank" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one worker failed."
  exit "$status"
fi

echo "All workers finished. Aggregating summary..."
"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_sd3_conditional_vendi.py" \
  "${COMMON_ARGS[@]}" \
  --rank 0 \
  --aggregate_only
