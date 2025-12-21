
# --------------- User Config -----------------
MODEL_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3"

IMAGE_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/SD3-Residual/test_resized.jpg"
OUTPUT_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/SD3-Residual/logs/results/attn_vis_complex_prompt/attn_vis_out-residual"


# Diffusion & Visualization
TIMESTEP=900
HEIGHT=1024
WIDTH=1024
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
# PROMPT="A cozy living room with a wooden coffee table, a white mug, a small potted plant, and stacked books, warm natural light from a window."
PROMPT="a photo of a green traffic light."
TOKEN_WORDS="photo of green traffic light"



RES_ORIGIN=1

RES_TARGET="$(seq -s ' ' 4 13)"

RES_WEIGHT="$(printf '0.0 %.0s' $(seq 4 13))"



DEVICE="cuda"
SEED=42
# ---------------------------------------------

echo "[INFO] Running visualization"

OUTPUT_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/SD3-Residual/logs/results/attn_vis_full/test2/attn_vis_out-BASE"

TIMESTEPS="0 2 4 6 8 9 10 11 12 14 16 18 20 22 24 27"

CUDA_VISIBLE_DEVICES=0 python vis_pro.py \
  --model "${MODEL_PATH}" \
  --prompt "${PROMPT}" \
  --layers ${LAYERS} \
  --token-words ${TOKEN_WORDS} \
  --height ${HEIGHT} --width ${WIDTH} \
  --output "${OUTPUT_DIR}" \
  --device ${DEVICE} \
  --seed ${SEED} \
  --dump-timesteps $TIMESTEPS \
  --residual_target_layers $RES_TARGET \
  --residual_origin_layer $RES_ORIGIN \
  --residual_weights $RES_WEIGHT \





# python visualize_sd3_cross_attention_pro.py \
#   --model "${MODEL_PATH}" \
#   --prompt "${PROMPT}" \
#   --image "${IMAGE_PATH}" \
#   --timestep-idx ${TIMESTEP} \
#   --layers ${LAYERS} \
#   --token-words ${TOKEN_WORDS} \
#   --height ${HEIGHT} --width ${WIDTH} \
#   --output "${OUTPUT_DIR}" \
#   --device ${DEVICE} \
#   --seed ${SEED} \
#   --residual_target_layers $RES_TARGET \
#   --residual_origin_layer $RES_ORIGIN \
#   --residual_weights $RES_WEIGHT \


# echo "[DONE] All attention maps saved under: ${OUTPUT_DIR}"
  