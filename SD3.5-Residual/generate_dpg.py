import argparse
import numpy as np
import random
import os
import glob
import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms
from PIL import Image

from generate_image_res import SD35PipelineWithRES
from sd35_transformer_res import SD35Transformer2DModel_RES
from util import load_residual_procrustes, select_residual_rotations, load_residual_weights
from lora_utils import inject_lora, load_lora_state_dict

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

DEFAULT_SD35_MODEL = "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5-large"

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5]),
    ])
    return transform


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def make_grid_2x2(imgs):
    assert len(imgs) == 4
    pil_imgs = [(torch.clamp(img * 0.5 + 0.5, 0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy() for img in imgs]
    pil_imgs = [Image.fromarray(p) for p in pil_imgs]

    w, h = pil_imgs[0].size
    grid = Image.new("RGB", (w * 2, h * 2))

    grid.paste(pil_imgs[0], (0, 0))
    grid.paste(pil_imgs[1], (w, 0))
    grid.paste(pil_imgs[2], (0, h))
    grid.paste(pil_imgs[3], (w, h))

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--NFE", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=1024)

    parser.add_argument("--model", type=str, default="sd3.5")
    parser.add_argument('--load_dir', type=str, default=None, help="replace it with your checkpoint")
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--prompt_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/prompts")

    parser.add_argument('--lora_ckpt', type=str, default=None, help='Path to LoRA-only checkpoint (.pth)')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target', type=str, default='all_linear',
                        help="all_linear 或模块名片段，如: to_q,to_k,to_v,to_out")
    parser.add_argument('--lora_dropout', type=float, default=0.0)

    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model not in ("sd3.5", "sd3.5-large", "sd35", "sd35-large"):
        print("[WARN] model flag is ignored for SD3.5 pipeline; use --load_dir to specify checkpoint.")

    model_path = args.load_dir or DEFAULT_SD35_MODEL
    pipe = SD35PipelineWithRES.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    pipe.transformer = SD35Transformer2DModel_RES(pipe.transformer)

    if args.lora_ckpt is not None:
        print(f"[LoRA] injecting & loading LoRA from: {args.lora_ckpt}")
        target = "all_linear" if args.lora_target == "all_linear" else tuple(args.lora_target.split(","))
        inject_lora(
            pipe.transformer,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target=target,
            dropout=args.lora_dropout,
            is_train=False,
        )
        lora_sd = torch.load(args.lora_ckpt, map_location="cpu")
        load_lora_state_dict(pipe.transformer, lora_sd, strict=True)
        pipe.transformer.eval()
        print("[LoRA] loaded and ready.")

    residual_rotation_matrices = None
    residual_rotation_meta = None
    if args.residual_procrustes_path is not None:
        residual_rotation_matrices, target_layers, meta = load_residual_procrustes(
            args.residual_procrustes_path
        )
        residual_rotation_matrices, args.residual_target_layers = select_residual_rotations(
            residual_rotation_matrices, target_layers, args.residual_target_layers
        )
        if args.residual_origin_layer is None and isinstance(meta, dict):
            args.residual_origin_layer = meta.get("origin_layer")
        residual_rotation_meta = meta

    if args.residual_weights is None and args.residual_weights_path is not None:
        args.residual_weights = load_residual_weights(args.residual_weights_path)

    os.makedirs(args.save_dir, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(args.prompt_dir, "*.txt")))
    total_prompts = len(txt_files)

    if args.world_size > 1:
        txt_files = [f for i, f in enumerate(txt_files) if i % args.world_size == args.rank]

    print(f"[DPG] World size = {args.world_size}, rank = {args.rank}")
    print(f"[DPG] Total prompts: {total_prompts}, this rank will handle: {len(txt_files)}")

    for txt_path in txt_files:
        base = os.path.basename(txt_path)
        name = os.path.splitext(base)[0]
        out_path = os.path.join(args.save_dir, f"{name}.png")

        if os.path.exists(out_path):
            print(f"[Skip] {name}.png already exists.")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"\n[DPG] Generating for: {name}")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                height=args.img_size,
                width=args.img_size,
                num_inference_steps=args.NFE,
                guidance_scale=args.cfg_scale,
                num_images_per_prompt=4,
                output_type="pt",
                residual_target_layers=args.residual_target_layers,
                residual_origin_layer=args.residual_origin_layer,
                residual_weights=args.residual_weights,
                residual_rotation_matrices=residual_rotation_matrices,
                residual_rotation_meta=residual_rotation_meta,
            )

        imgs = result.images
        if torch.is_tensor(imgs) and imgs.dim() == 4:
            imgs_list = [imgs[i] for i in range(4)]
        else:
            imgs_list = [img for img in imgs]

        grid = make_grid_2x2(imgs_list)
        grid.save(out_path)
        print(f"[Saved] {out_path}")

    print("\n[DPG] All done.")
