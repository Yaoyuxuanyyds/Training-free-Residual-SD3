#!/bin/bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate qwen-image
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Qwen-Image-Residual


python compute_qwen_text_procrustes.py \
  --model /inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --num_samples 5000 \
  --origin_layer 2 \
  --target_layer_start 3 \
  --output /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Qwen-Image-Residual/output/procrustes_rotations/qwen_procrustes_rotations-ln-o2.pt \
  --col-center \
  --timestep-buckets 4 \

