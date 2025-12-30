set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate repa-sd3
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3

# Dataset-level evaluation example
python compute_sd3_text_grad_sensitivity.py \
    --dataset blip3o \
    --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
    --num-samples 100 \
    --timestep-idx 500 \
    --num-seeds 1 \
    --num-timesteps 10 \
    --ignore-padding \
    --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/vis/grad-metrics

# Single prompt example
# python compute_sd3_text_grad_sensitivity.py \
#     --prompt "The photo of a green traffic light." \
#     --image /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/image0.png \
#     --timestep-idx 945 \
#     --num-seeds 2 \
#     --num-timesteps 10 \
#     --ignore-padding \
#     --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/vis/grad-metrics
