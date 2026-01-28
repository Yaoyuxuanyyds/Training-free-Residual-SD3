import torch
from pipeline_taca_flux import FluxPipeline

pipe = FluxPipeline.from_pretrained("/inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

prompt = "A beautiful landscape with mountains and a river."

# Comment the following line if you just want training-free inference
pipe.load_lora_weights('/inspire/hdd/project/chineseculture/public/yuxuan/TACA/TACA/flux-dev-lora-rank-64.safetensors')

image = pipe(
    prompt,
    num_inference_steps=30,
).images[0]

image.save("out.png")