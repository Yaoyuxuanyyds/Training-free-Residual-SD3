set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate repa-sd3
cd /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3

# 使用 compute_sd3_text_exp.py 的 PCA dump 进行可视化
python compute_sd3_text_pca.py \
    --npz-path /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/results/txt_feat/text_emb_layers.npz \
    --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/vis/pca_layers-sd3 \
    --cmap viridis \
    --point-size 0.5
