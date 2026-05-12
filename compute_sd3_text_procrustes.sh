#!/bin/bash

### -----------------------------
### Basic Settings
### -----------------------------
export CUDA_VISIBLE_DEVICES=0

MODEL="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3"
DATADIR="/inspire/hdd/project/chineseculture/public/yuxuan/datasets"
DATASET="blip3o60k"
NUM_SAMPLES=5000
HEIGHT=1024
WIDTH=1024
SEED=0
TIMESTEP_BUCKETS=1
RES_USE_LAYERNORM=0

OUTDIR="/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/procrustes_rotations"


### -----------------------------
### Procrustes Settings
### -----------------------------
# 单层 origin layer，和 target layers 做 Procrustes 对齐
ORIGIN_LAYER=1
TARGET_LAYER_START=2

# 输出文件名
OUTNAME="procrustes_rotations_${DATASET}_ln_t${TARGET_LAYER_START}_o${ORIGIN_LAYER}_single_origin_noLN.pt"
OUTPUT_PATH="${OUTDIR}/${OUTNAME}"
mkdir -p "$OUTDIR"


### -----------------------------
### Run Procrustes Computation
### -----------------------------
python compute_sd3_text_procrustes.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --datadir "$DATADIR" \
    --num-samples $NUM_SAMPLES \
    --height $HEIGHT \
    --width $WIDTH \
    --seed $SEED \
    --origin-layer $ORIGIN_LAYER \
    --target-layer-start $TARGET_LAYER_START \
    --residual_use_layernorm $RES_USE_LAYERNORM \
    --timestep-buckets $TIMESTEP_BUCKETS \
    --output "$OUTPUT_PATH" \
    --col-center
