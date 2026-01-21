#!/bin/bash

source /inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/.venv_diffusers/bin/activate
cd /inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3-1/flux

python generate_res_geneval.py \
    --residual_origin_layer 2 \
    --residual_target_layers  7 8 9 10 11 12\
    --residual_weight 0.25\
    --outdir geneval_outputs/res_2__7_8_9_10_11_12_0.25