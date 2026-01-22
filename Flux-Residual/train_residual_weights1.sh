#!/usr/bin/env bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/.venv_diffusers/bin/activate
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# =============== 阶段 1：配置参数 ===============
MODEL_KEY="/inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev"

DATASET="blip3o60k"
DATADIR="/inspire/hdd/project/chineseculture/public/yuxuan/datasets"

CACHE_DIRS=(
  "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/cache/basic_content/blip3o60k-0-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/cache/basic_content/blip3o60k-1-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/cache/basic_content/blip3o60k-2-l333"
  "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/cache/basic_content/blip3o60k-3-l333"
)

# ===== 核心实验超参数 =====
NPROC=8
IMG_SIZE=1024
STEPS=2000
BATCH_SIZE=2        # per-GPU
LR=1e-3
RES_LAYER=2
RES_INIT=0.05

GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NPROC))


ROTATION_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/procrustes_rotations/procrustes_rotations_coco5k_ln_t1-full-o2.pt"


# =============== 阶段 2：自动构造 LOGDIR ===============
LOGDIR="./logs/learnable_residual/lr${LR}_layer${RES_LAYER}_init${RES_INIT}_img${IMG_SIZE}_steps${STEPS}_gbs${GLOBAL_BATCH_SIZE}-Rotation"

mkdir -p "${LOGDIR}"

echo "======================================"
echo "Launching training with config:"
echo "  lr                    = ${LR}"
echo "  residual_origin_layer = ${RES_LAYER}"
echo "  residual_init         = ${RES_INIT}"
echo "  img_size              = ${IMG_SIZE}"
echo "  steps                 = ${STEPS}"
echo "  batch_size (per GPU)  = ${BATCH_SIZE}"
echo "  global_batch_size     = ${GLOBAL_BATCH_SIZE}"
echo "  logdir                = ${LOGDIR}"
echo "======================================"

# =============== 阶段 3：启动训练 ===============
torchrun \
  --nproc_per_node=${NPROC} \
  train_residual_weights.py \
  --model_dir "${MODEL_KEY}" \
  --output_dir "${LOGDIR}" \
  --datadir "${DATADIR}" \
  --dataset "${DATASET}" \
  --precompute_dir "${CACHE_DIRS[@]}" \
  --img_size ${IMG_SIZE} \
  --steps ${STEPS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --residual_origin_layer ${RES_LAYER} \
  --residual_init ${RES_INIT} \
  --residual_rotation_path ${ROTATION_PATH}
