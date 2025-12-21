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


SAVEDIR="/inspire/hdd/project/chineseculture/public/yuxuan/SD3-Residual/logs/generate/lora/test"


### -----------------------------
### Residual Experiment Settings
### -----------------------------
# 支持任意 residual 参组合
RES_ORIGIN=1

RES_TARGET="$(seq -s ' ' 4 13)"

RES_WEIGHT="$(printf '0.1 %.0s' $(seq 4 13))"



LORA_CKPT="/inspire/hdd/project/chineseculture/public/yuxuan/SD3-Residual/logs/sd3-lora/ds-qwen-l333_epochs-5/bs-256_lrLoRA-1e-5_warmup-200-time-logitnorm/lora-r256-a256-d0.05-t-to_q-to_k-to_v-to_out.0/Residual--target-4to13__origin-1__w-0.1-LayerNorm-StopGrad/20251206-134312/sd3_lora_dp_train_cached/lora_step2000.pth"

LORA_TG="to_q,to_k,to_v,to_out.0"
LORA_RANK=256
LORA_ALPHA=256


PROMPT="A truck and a microwave."


# 自动压缩 target layers 显示形式
FIRST_LAYER=$(echo "$RES_TARGET" | awk '{print $1}')
LAST_LAYER=$(echo "$RES_TARGET" | awk '{print $NF}')
EXP_TARGET_SHORT="${FIRST_LAYER}to${LAST_LAYER}"
# 权重统一就取第一个即可
FIRST_WEIGHT=$(echo "$RES_WEIGHT" | awk '{print $1}')
EXP_WEIGHT_SHORT="${FIRST_WEIGHT}"
SAVENAME="target-${EXP_TARGET_SHORT}__origin-${RES_ORIGIN}__w-${EXP_WEIGHT_SHORT}-LayerNorm"


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
    --residual_target_layers $RES_TARGET \
    --residual_origin_layer $RES_ORIGIN \
    --residual_weights $RES_WEIGHT \
    --lora_ckpt $LORA_CKPT --lora_target $LORA_TG --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA \





# A woman holding a Hello Kitty phone on her hands.

# the word'START'written inchalk on asidewalk

# A mirror that tracks your daily health metrics.

# three black cats standing next to two orange cats.

# a spaceship that looks like the Sydney OperaHouse.

# Generate an image of an animal with (3 + 6) lives.

# A cat holding a sign that says hello world

# A picture of a bookshelf with some books on it. The bottom shelf is empty.

# A man holds a letter stamped “Accepted” from his dream university, with his emotional response clearly visible.
