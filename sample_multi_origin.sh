#!/bin/bash

### -----------------------------
### Basic Settings
### -----------------------------
export CUDA_VISIBLE_DEVICES=0

MODEL="sd3"
NFE=28
CFG=7.0
IMGSIZE=1024
BATCHSIZE=1

SAVEDIR="/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/generate/test_multi_origin"


### -----------------------------
### Residual Experiment Settings
### -----------------------------
# 多层 origin layers，推理时会先求均值再 residual 到 target layers
RES_ORIGINS="1 2"
RES_USE_LAYERNORM=0
RES_TARGET="$(seq -s ' ' 3 21)"
RES_WEIGHT="$(printf '0.05 %.0s' $(seq 3 21))"

PROMPT="A photo of a green traffic light."


# 自动压缩显示形式 
FIRST_ORIGIN=$(echo "$RES_ORIGINS" | awk '{print $1}')
LAST_ORIGIN=$(echo "$RES_ORIGINS" | awk '{print $NF}')
ORIGIN_COUNT=$(echo "$RES_ORIGINS" | awk '{print NF}')
if [ "$ORIGIN_COUNT" -eq 1 ]; then
    EXP_ORIGIN_SHORT="${FIRST_ORIGIN}"
else
    EXP_ORIGIN_SHORT="${FIRST_ORIGIN}to${LAST_ORIGIN}"
fi

FIRST_TARGET=$(echo "$RES_TARGET" | awk '{print $1}')
LAST_TARGET=$(echo "$RES_TARGET" | awk '{print $NF}')
EXP_TARGET_SHORT="${FIRST_TARGET}to${LAST_TARGET}"

FIRST_WEIGHT=$(echo "$RES_WEIGHT" | awk '{print $1}')
EXP_WEIGHT_SHORT="${FIRST_WEIGHT}"

SAVENAME="target-${EXP_TARGET_SHORT}__origin-${EXP_ORIGIN_SHORT}__w-${EXP_WEIGHT_SHORT}-LayerNorm-Procruste-multi-origin"
FULL_SAVE_DIR="${SAVEDIR}/${SAVENAME}"
mkdir -p "$FULL_SAVE_DIR"


### -----------------------------
### Run sampling
### -----------------------------
python sample.py \
    --cfg_scale $CFG \
    --NFE $NFE \
    --model $MODEL \
    --img_size $IMGSIZE \
    --batch_size $BATCHSIZE \
    --save_dir $FULL_SAVE_DIR \
    --save_name $SAVENAME \
    --prompt "$PROMPT" \
    --timestep_residual_weight_fn "constant" \
    --timestep_residual_weight_exp_alpha 0.0 \
    --residual_target_layers $RES_TARGET \
    --residual_origin_layers $RES_ORIGINS \
    --residual_use_layernorm $RES_USE_LAYERNORM \
    --residual_weights $RES_WEIGHT
    # --residual_procrustes_path /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/procrustes_rotations/procrustes_rotations_multi_origin.pt
