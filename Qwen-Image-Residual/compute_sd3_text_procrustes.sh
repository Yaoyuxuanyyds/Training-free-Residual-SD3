python Qwen-Image-Residual/compute_sd3_text_procrustes.py \
  --model /inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --num-samples 50000 \
  --origin-layer 1 \
  --target-layer-start 2 \
  --output /inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual/output/procrustes_rotations/qwen_procrustes_rotations.pt
