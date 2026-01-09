#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=1



MODEL_KEY="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3"
DATASET="blip3o60k"
DATADIR="/inspire/hdd/project/chineseculture/public/yuxuan/datasets"
CACHE_DIRS=(
  "/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/logs/cache/basic_content/blip3o60k-0-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/logs/cache/basic_content/blip3o60k-1-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/logs/cache/basic_content/blip3o60k-2-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/logs/cache/basic_content/blip3o60k-3-l333"
)

LOGDIR="./logs/learnable_residual"

python train_residual_weights.py \
  --model_key "${MODEL_KEY}" \
  --datadir "${DATADIR}" \
  --precompute_dir "${CACHE_DIRS[@]}" \
  --logdir "${LOGDIR}" \
  --dataset "${DATASET}" \
  --img_size 512 \
  --epochs 1 \
  --batch_size 64 \
  --lr 1e-3 \
  --wd 0.0 \
  --dtype float16 \
  --time_mode logitnorm \
  --time_shift 0.0 \
  --warmup_steps 100 \
  --eval_interval 500 \
  --save_interval 500 \
  --grad_clip 1.0 \
  --residual_origin_layer 1 \
  --residual_init 0.1 \
  --residual_use_layernorm 1 \
  --residual_rotation_path /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/procrustes_rotations/procrustes_rotations_coco5k_ln.pt \
