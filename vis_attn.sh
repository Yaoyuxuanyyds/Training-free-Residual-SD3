
# --------------- User Config -----------------
MODEL_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3"

# Diffusion & Visualization
HEIGHT=1024
WIDTH=1024
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"

# PROMPT="A cozy living room with a wooden coffee table, a white mug, a small potted plant, and stacked books, warm natural light from a window."
# TOKEN_WORDS="A cozy living room with a wooden coffee table , a white mug , a small pot ted plant , and stacked books , warm natural light from a window"

# PROMPT="The photo of a green traffic light."
# TOKEN_WORDS="The photo of a green traffic light ."

PROMPT="A red dog."
TOKEN_WORDS="A red dog ."



RES_ORIGIN=1

RES_TARGET="$(seq -s ' ' 2 13)"

RES_WEIGHT="$(printf '0.0 %.0s' $(seq 2 13))"



DEVICE="cuda"
SEED=42
# ---------------------------------------------

echo "[INFO] Running visualization"

OUTPUT_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/attn_vis/test6/attn_vis_out-BASE"
# joint attention maps saved as: ${OUTPUT_DIR}/tXXXX/joint_attn_layer-<layer>.png

TIMESTEPS="0 2 4 6 8 10 12 16 20 24 27"

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
  --cfg-scale 4.0 \





PROMPT="A bench and a ski."
TOKEN_WORDS="A bench and a ski ."
OUTPUT_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/logs/attn_vis/test7/attn_vis_out-BASE"


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
  --cfg-scale 4.0 \


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
  
