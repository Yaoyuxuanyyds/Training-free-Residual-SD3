CUDA_VISIBLE_DEVICES=0 python compute_flux_text_procrustes.py \
  --dataset blip3o60k \
  --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets \
  --model /inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev \
  --num-samples 5000 \
  --origin-layer 1 \
  --target-layer-start 2 \
  --output /inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/Flux-Residual/logs/procrustes_rotations/procrustes_rotations_coco5k_ln_t1.pt \
  --col-center \
  --timestep-buckets 1 \