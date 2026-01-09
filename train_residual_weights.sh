#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="/path/to/SD3"
DATADIR="/path/to/dataset"
PRECOMPUTE_DIR="/path/to/precompute"
LOGDIR="./logs"

python train_residual_weights.py \
  --model_key "${MODEL_KEY}" \
  --datadir "${DATADIR}" \
  --precompute_dir "${PRECOMPUTE_DIR}" \
  --logdir "${LOGDIR}" \
  --dataset coco \
  --img_size 512 \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --wd 0.0 \
  --dtype float16 \
  --time_mode logitnorm \
  --time_shift 0.0 \
  --warmup_steps 100 \
  --eval_interval 500 \
  --save_interval 500 \
  --grad_clip 1.0 \
  --residual_origin_layer 1 \
  --residual_init 0.0 \
  --residual_use_layernorm 1
