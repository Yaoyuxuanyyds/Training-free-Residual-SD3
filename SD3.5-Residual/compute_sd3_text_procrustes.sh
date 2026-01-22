set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate repa-sd3
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/SD3.5-Residual


CUDA_VISIBLE_DEVICES=0 python compute_sd3_text_procrustes.py \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --num-samples 5000 \
  --origin-layer 2 \
  --target-layer-start 3 \
  --output procrustes_rotations_coco5k_ln_t5_o2.pt \
  --timestep-buckets 5 \
  --col_center