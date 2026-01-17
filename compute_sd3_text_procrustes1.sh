CUDA_VISIBLE_DEVICES=1 python compute_sd3_text_procrustes.py \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --num-samples 5000 \
  --origin-layer 1 \
  --target-layer-start 2 \
  --num-timesteps 1 \
  --output /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/procrustes_rotations/procrustes_rotations_coco5k_rn_pro.pt \