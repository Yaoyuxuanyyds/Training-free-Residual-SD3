set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate repa-sd3
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3



python compute_sd3_text_exp.py \
    --dataset blip3o \
    --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
    --num-samples 50 \
    --timestep-idx 500 \
    --vis-sample-size 5000 \
    --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/results/txt_feat \
    --dataset-train \
    --ignore-padding

# # 单条 prompt 可视化（带 token idx+内容标注）
# python compute_sd3_text_cknna_pro.py \
#     --prompt "A cozy living room with a wooden coffee table, a white mug, a small potted plant, and stacked books, warm natural light from a window." \
#     --image /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/image0.png \
#     --timestep-idx 945 \
#     --num-samples 1 \
#     --ignore-padding \
#     --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/vis/test3

# A cozy living room with a wooden coffee table, a white mug, a small potted plant, and stacked books, warm natural light from a window.

# LoRA 示例
# python compute_sd3_text_cknna_pro.py \
#     --dataset blip3o \
#     --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
#     --num-samples 25 \
#     --timestep-idx 500 \
#     --vis-sample-size 100000 \
#     --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/logs/results/results-data-rm1/base-normalize-LN \
#     --dataset-train \
#     --ignore-padding \
#     --lora-rank 256 \
#     --lora-alpha 256 \
#     --lora-target "to_q,to_k,to_v,to_out.0" \
#     --lora-dropout 0.05 \
#     --lora-ckpt "/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/logs/sd3-lora-Self-Forcing/ds-blip3o60k&echo4o-l333/epochs-15_bs-256_lrLoRA-1e-5_warmup-500/layers-16-gm0.02-cw1.0-sw0.0/lora-r256-a256-d0.05/20251110-054217/sd3_lora_dp_train_cached/lora_step7000.pth"

# --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/logs/results/results-dataset-rm1/t500_l16-l0_r256_gm0.02_lr1e-5-test \
