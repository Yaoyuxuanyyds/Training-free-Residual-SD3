CUDA_VISIBLE_DEVICES=0 python compute_sd3_text_procrustes.py \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --num-samples 500 \
  --origin-layer 1 \
  --target-layer-start 2 \
  --output /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/procrustes_rotations/procrustes_rotations_coco0.5k_rn_pro.pt \