#!/usr/bin/env bash
set -euo pipefail

# =============== 环境 ===============
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate qwen-image
cd /path/to/Training-free-Residual-SD3/Qwen-Image-Residual

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="/path/to/Qwen-Image"
DATASET="coco"
DATADIR="/path/to/datasets"

LOGDIR="./logs/learnable_residual"

torchrun \
  --nproc_per_node=4 \
  train_residual_weights.py \
  --model_dir "${MODEL_DIR}" \
  --dataset "${DATASET}" \
  --datadir "${DATADIR}" \
  --logdir "${LOGDIR}" \
  --img_size 1024 \
  --steps 5000 \
  --batch_size 64 \
  --lr 1e-3 \
  --wd 0.0 \
  --dtype bfloat16 \
  --time_mode logitnorm \
  --time_shift 0.0 \
  --warmup_steps 100 \
  --save_interval 500 \
  --grad_clip 1.0 \
  --residual_origin_layer 1 \
  --residual_init 0.05 \
  --init_mode "constant" \
  --residual_use_layernorm 1 \
  --residual_stop_grad 1 \
  --residual_rotation_path /path/to/procrustes_rotations_qwen.pt \
  --residual_smoothness_weight 0.1
