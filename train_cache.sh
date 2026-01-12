#!/bin/bash
set -euo pipefail

# =============== 环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate repa-sd3
cd /inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3




# ========= 训练配置 =========
DATASET="coco"
DATADIR="/inspire/hdd/project/chineseculture/public/yuxuan/datasets"
MODEL_KEY="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3"
BATCH_SIZE=256
DTYPE="float16"






# ========= 预存特征 =========
python train_cache_subset.py \
  --datadir "${DATADIR}" \
  --dataset "${DATASET}" \
  --model_key "${MODEL_KEY}" \
  --pc_batch_size ${BATCH_SIZE} \
  --dtype "${DTYPE}" \
  --precompute_dir "/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/logs/cache/basic_content/${DATASET}-l333" \
  --do_precompute \
  --subset_number 0






