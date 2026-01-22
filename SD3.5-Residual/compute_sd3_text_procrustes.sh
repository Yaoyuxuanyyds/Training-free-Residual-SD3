CUDA_VISIBLE_DEVICES=0 python compute_sd3_text_procrustes.py \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --num-samples 5000 \
  --origin-layer 1 \
  --target-layer-start 2 \
  --output procrustes_rotations_coco5k_ln_t4.pt \
  --timestep-buckets 4 \