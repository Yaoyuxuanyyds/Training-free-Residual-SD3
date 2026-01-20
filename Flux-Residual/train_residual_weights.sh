#!/usr/bin/env bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/.venv_diffusers/bin/activate
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual


export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_KEY="/inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev"
DATASET="blip3o60k"
DATADIR="/inspire/hdd/project/chineseculture/public/yuxuan/datasets"
CACHE_DIRS=(
  "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/cache/basic_content/blip3o60k-0-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/cache/basic_content/blip3o60k-1-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/cache/basic_content/blip3o60k-2-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/cache/basic_content/blip3o60k-3-l333"
)

LOGDIR="./logs/learnable_residual"

torchrun \
  --nproc_per_node=1 \
  train_residual_weights.py \
  --model_dir "${MODEL_KEY}" \
  --output_dir "${LOGDIR}" \
  --datadir "${DATADIR}" \
  --dataset "${DATASET}" \
  --precompute_dir "${CACHE_DIRS[@]}" \
  --img_size 512 \
  --steps 2000 \
  --batch_size 4 \
  --lr 1e-3 \
  --residual_origin_layer 1 \
  --residual_init 0.05 \
  --residual_rotation_path /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/procrustes_rotations/procrustes_rotations_coco5k_ln_t5-full.pt \

