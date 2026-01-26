#!/usr/bin/env bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/.venv_diffusers/bin/activate
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual


CUDA_VISIBLE_DEVICES=0 python compute_flux_text_procrustes.py \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --model /inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev \
  --num-samples 5000 \
  --origin-layer 16 \
  --target-layer-start 17 \
  --output /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/procrustes_rotations/procrustes_rotations_coco5k_ln_t1-full-o16.pt \
  --col-center \
  --timestep-buckets 1 \




