import argparse
import numpy as np
import random, os, glob
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms
from PIL import Image

from sampler import SD3Euler, build_timestep_residual_weight_fn
from util import load_residual_procrustes, select_residual_rotations, load_residual_weights
from lora_utils import *

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def make_grid_2x2(imgs):
    """imgs: list of 4 tensors, each (3,1024,1024)"""
    assert len(imgs) == 4

    # 转 PIL 更简单
    pil_imgs = [(torch.clamp(img * 0.5 + 0.5, 0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy() for img in imgs]
    pil_imgs = [Image.fromarray(p) for p in pil_imgs]

    w, h = pil_imgs[0].size
    grid = Image.new("RGB", (w * 2, h * 2))

    grid.paste(pil_imgs[0], (0, 0))
    grid.paste(pil_imgs[1], (w, 0))
    grid.paste(pil_imgs[2], (0, h))
    grid.paste(pil_imgs[3], (w, h))

    return grid


# ================================================================
#                       主脚本
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--NFE", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=1024)
    
    parser.add_argument("--model", type=str, default="sd3")
    parser.add_argument('--load_dir', type=str, default=None, help="replace it with your checkpoint")
    parser.add_argument("--save_dir", type=str, required=True)

    # dpg bench prompt path
    parser.add_argument("--prompt_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/prompts")

    # ---------- LoRA 采样支持 ---------- #
    parser.add_argument('--lora_ckpt', type=str, default=None, help='Path to LoRA-only checkpoint (.pth)')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target', type=str, default='all_linear',
                        help="all_linear 或模块名片段，如: to_q,to_k,to_v,to_out")
    parser.add_argument('--lora_dropout', type=float, default=0.0)


    # residual
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument(
        "--timestep_residual_weight_fn",
        type=str,
        default="constant",
        help="Mapping from timestep (0-1000) to residual weight multiplier.",
    )
    parser.add_argument(
        "--timestep_residual_weight_power",
        type=float,
        default=1.0,
        help="Optional power for timestep residual weight mapping.",
    )
    parser.add_argument(
        "--timestep_residual_weight_exp_alpha",
        type=float,
        default=1.5,
        help="Exponent alpha for exponential timestep residual weight mapping.",
    )
    # 多 GPU 分片参数
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of workers for sharded prompts (e.g., 4 GPUs).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="This worker's rank in [0, world_size-1].",
    )


    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    if args.model != "sd3":
        raise ValueError("Only sd3 is supported for this benchmark.")
    sampler = SD3Euler(use_8bit=False, load_ckpt_path=args.load_dir)

    # ---------- 如果提供了 LoRA ckpt，注入 + 加载 ----------
    if args.lora_ckpt is not None:
        print(f"[LoRA] injecting & loading LoRA from: {args.lora_ckpt}")
        target = "all_linear" if args.lora_target == "all_linear" else tuple(args.lora_target.split(","))
        # 对 sampler.denoiser（SD3Transformer2DModel_Vanilla）里的 transformer 注入
        denoiser = sampler.denoiser
        inject_lora(denoiser, rank=args.lora_rank, alpha=args.lora_alpha,
                    target=target, dropout=args.lora_dropout)
        denoiser.to(device=device, dtype=torch.float32)   # 就地转换
        lora_sd = torch.load(args.lora_ckpt, map_location="cpu")
        load_lora_state_dict(denoiser, lora_sd, strict=True)
        
        sampler.denoiser.eval()
        print("[LoRA] loaded and ready.")
        

    sampler.denoiser.to(torch.float32)
    torch.set_default_dtype(torch.float32)

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

    # prepare dirs
    os.makedirs(args.save_dir, exist_ok=True)

    # # 扫描 prompts
    txt_files = sorted(glob.glob(os.path.join(args.prompt_dir, "*.txt")))
    total_prompts = len(txt_files)

    # 多 GPU 分片：按下标对 world_size 取模
    if args.world_size > 1:
        txt_files = [f for i, f in enumerate(txt_files) if i % args.world_size == args.rank]

    print(f"[DPG] World size = {args.world_size}, rank = {args.rank}")
    print(f"[DPG] Total prompts: {total_prompts}, this rank will handle: {len(txt_files)}")


    # ================================================================
    #                        遍历每个 prompt
    # ================================================================
    for txt_path in txt_files:
        base = os.path.basename(txt_path)
        name = os.path.splitext(base)[0]
        out_path = os.path.join(args.save_dir, f"{name}.png")

        # 若已生成则跳过
        if os.path.exists(out_path):
            print(f"[Skip] {name}.png already exists.")
            continue

        # 读取 prompt
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"\n[DPG] Generating for: {name}")

        # ------------------------------------------------
        #    一次输入 prompt，输出 4 张图像
        # ------------------------------------------------
        prompts = [prompt] * 4

        with torch.inference_mode():
            if args.residual_origin_layer is None:
                imgs = sampler.sample(
                    prompts,
                    NFE=args.NFE,
                    img_shape=(args.img_size, args.img_size),
                    cfg_scale=args.cfg_scale,
                    batch_size=4,
                )
            else:
                imgs = sampler.sample_residual(
                    prompts,
                    NFE=args.NFE,
                    img_shape=(args.img_size, args.img_size),
                    cfg_scale=args.cfg_scale,
                    batch_size=4,
                    residual_target_layers=args.residual_target_layers,
                    residual_origin_layer=args.residual_origin_layer,
                    residual_weights=args.residual_weights,
                    residual_rotation_matrices=residual_rotation_matrices,
                    residual_rotation_meta=residual_rotation_meta,
                    residual_timestep_weight_fn=build_timestep_residual_weight_fn(
                        args.timestep_residual_weight_fn,
                        power=args.timestep_residual_weight_power,
                        exp_alpha=args.timestep_residual_weight_exp_alpha,
                    ),
                )

        # imgs shape: [4, 3, 1024, 1024]
        # 拼接成 2×2 = 2048×2048
        grid = make_grid_2x2([imgs[i] for i in range(4)])

        # 保存
        grid.save(out_path)
        print(f"[Saved] {out_path}")

    print("\n[DPG] All done.")
