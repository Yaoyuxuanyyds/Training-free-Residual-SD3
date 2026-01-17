CUDA_VISIBLE_DEVICES=0 python compute_qwen_text_procrustes.py \
  --model /inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --num_samples 5000 \
  --origin_layer 1 \
  --target_layer_start 2 \
  --output /inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual/output/procrustes_rotations/qwen_procrustes_rotations-rn.pt
