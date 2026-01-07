#!/bin/bash

### -----------------------------
### Basic Settings
### -----------------------------
export CUDA_VISIBLE_DEVICES=1

MODEL="sd3"
NFE=28
CFG=7.0
IMGSIZE=1024
BATCHSIZE=1


SAVEDIR="/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/generate/test3"


### -----------------------------
### Residual Experiment Settings
### -----------------------------
# 支持任意 residual 参组合
RES_ORIGIN=1

RES_TARGET="$(seq -s ' ' 2 21)"

RES_WEIGHT="$(printf '0.05 %.0s' $(seq 2 21))"





PROMPT="a photo of a green traffic light."


# 自动压缩 target layers 显示形式
FIRST_LAYER=$(echo "$RES_TARGET" | awk '{print $1}')
LAST_LAYER=$(echo "$RES_TARGET" | awk '{print $NF}')
EXP_TARGET_SHORT="${FIRST_LAYER}to${LAST_LAYER}"
# 权重统一就取第一个即可
FIRST_WEIGHT=$(echo "$RES_WEIGHT" | awk '{print $1}')
EXP_WEIGHT_SHORT="${FIRST_WEIGHT}"
# SAVENAME="target-${EXP_TARGET_SHORT}__origin-${RES_ORIGIN}__w-${EXP_WEIGHT_SHORT}-LayerNorm-Procruste"
SAVENAME="target-${EXP_TARGET_SHORT}__origin-${RES_ORIGIN}__w-${EXP_WEIGHT_SHORT}-LayerNorm-Procruste-exp-pro"
# SAVENAME="target-${EXP_TARGET_SHORT}__origin-${RES_ORIGIN}__w-${EXP_WEIGHT_SHORT}-LayerNorm"
# SAVENAME="target-${EXP_TARGET_SHORT}__origin-${RES_ORIGIN}__w-${EXP_WEIGHT_SHORT}-LayerNorm-exp"

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
    --timestep_residual_weight_fn "exp" \
    --residual_target_layers $RES_TARGET \
    --residual_origin_layer $RES_ORIGIN \
    --residual_weights $RES_WEIGHT \
    --residual_procrustes_path /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/procrustes_rotations/procrustes_rotations_blip3o10k_ln.pt



# A woman holding a Hello Kitty phone on her hands.

# the word'START'written inchalk on asidewalk

# A mirror that tracks your daily health metrics.

# three black cats standing next to two orange cats.

# a spaceship that looks like the Sydney OperaHouse.

# Generate an image of an animal with (3 + 6) lives.

# A cat holding a sign that says hello world

# A picture of a bookshelf with some books on it. The bottom shelf is empty.

# A man holds a letter stamped “Accepted” from his dream university, with his emotional response clearly visible.
