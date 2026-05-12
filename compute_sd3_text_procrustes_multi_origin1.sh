#!/bin/bash
#!/bin/bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate repa-sd3
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3


### -----------------------------
### Basic Settings
### -----------------------------
export CUDA_VISIBLE_DEVICES=1

MODEL="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3"
DATADIR="/inspire/hdd/project/chineseculture/public/yuxuan/datasets"
DATASET="blip3o60k"
NUM_SAMPLES=5000
HEIGHT=1024
WIDTH=1024
SEED=0
TIMESTEP_BUCKETS=1
RES_USE_LAYERNORM=1

OUTDIR="/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/procrustes_rotations"


### -----------------------------
### Procrustes Settings
### -----------------------------
# 多层 origin layers，先求均值，再和 target layers 做 Procrustes 对齐
ORIGIN_LAYERS="1 2 3 4"
TARGET_LAYER_START=5

# 输出文件名
FIRST_ORIGIN=$(echo "$ORIGIN_LAYERS" | awk '{print $1}')
LAST_ORIGIN=$(echo "$ORIGIN_LAYERS" | awk '{print $NF}')
ORIGIN_COUNT=$(echo "$ORIGIN_LAYERS" | awk '{print NF}')
if [ "$ORIGIN_COUNT" -eq 1 ]; then
    ORIGIN_TAG="${FIRST_ORIGIN}"
else
    ORIGIN_TAG="${FIRST_ORIGIN}to${LAST_ORIGIN}"
fi

OUTNAME="procrustes_rotations_${DATASET}_ln_t${TARGET_LAYER_START}_o${ORIGIN_TAG}_multi_origin.pt"
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
    --origin-layers $ORIGIN_LAYERS \
    --target-layer-start $TARGET_LAYER_START \
    --residual_use_layernorm $RES_USE_LAYERNORM \
    --timestep-buckets $TIMESTEP_BUCKETS \
    --output "$OUTPUT_PATH" \
    --col-center
