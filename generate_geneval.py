

import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import save_image

from sampler import SD3Euler, build_timestep_residual_weight_fn
from util import load_residual_procrustes, select_residual_rotations, set_seed, load_residual_weights
from lora_utils import *

torch.set_grad_enabled(False)


# ============================================================
# SD3 Image Generator（保持原样）
# ============================================================
class SD3ImageGenerator:
    def __init__(
        self,
        model='sd3',
        load_dir=None,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_use_layernorm: bool = True,
        residual_rotation_matrices=None,
        residual_rotation_meta=None,
        residual_timestep_weight_fn=None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model == 'sd3':
            self.sampler = SD3Euler(use_8bit=False, load_ckpt_path=load_dir)
        else:
            raise ValueError('model should be sd3 only')


        # 保存 residual 参数
        self.residual_target_layers = residual_target_layers
        self.residual_origin_layer = residual_origin_layer
        self.residual_weights = residual_weights
        self.residual_use_layernorm = residual_use_layernorm
        self.residual_rotation_matrices = residual_rotation_matrices
        self.residual_rotation_meta = residual_rotation_meta
        self.residual_timestep_weight_fn = residual_timestep_weight_fn


    def generate_image(
        self,
        prompt,
        seed=0,
        img_size=1024,
        num_inference_steps=28,
        guidance_scale=7.0,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_use_layernorm: bool = True,
        residual_rotation_matrices=None,
        residual_rotation_meta=None,
        residual_timestep_weight_fn=None,
    ):
        # 优先级：函数参数 > 初始化参数
        rt = residual_target_layers if residual_target_layers is not None else self.residual_target_layers
        ro = residual_origin_layer if residual_origin_layer is not None else self.residual_origin_layer
        rw = residual_weights if residual_weights is not None else self.residual_weights
        rln = residual_use_layernorm if residual_use_layernorm is not None else self.residual_use_layernorm
        rr = residual_rotation_matrices if residual_rotation_matrices is not None else self.residual_rotation_matrices
        rr_meta = (
            residual_rotation_meta
            if residual_rotation_meta is not None
            else self.residual_rotation_meta
        )
        rtw = (
            residual_timestep_weight_fn
            if residual_timestep_weight_fn is not None
            else self.residual_timestep_weight_fn
        )

        set_seed(seed)
        prompts = [prompt]

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            if ro is None:
                img = self.sampler.sample(
                    prompts,
                    NFE=num_inference_steps,
                    img_shape=(img_size, img_size),
                    cfg_scale=guidance_scale,
                    batch_size=1
                )
            else:
                img = self.sampler.sample_residual(
                    prompts,
                    NFE=num_inference_steps,
                    img_shape=(img_size, img_size),
                    cfg_scale=guidance_scale,
                    batch_size=1,
                    residual_target_layers=rt,
                    residual_origin_layer=ro,
                    residual_weights=rw,
                    residual_use_layernorm=rln,
                    residual_rotation_matrices=rr,
                    residual_rotation_meta=rr_meta,
                    residual_timestep_weight_fn=rtw,
                )
        return img


# ============================================================
# 参数解析（加入 world_size / rank）
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)

    # residual
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument("--residual_use_layernorm", type=int, default=1)
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

    # ---------- LoRA 采样支持 ---------- #
    parser.add_argument('--lora_ckpt', type=str, default=None, help='Path to LoRA-only checkpoint (.pth)')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target', type=str, default='all_linear',
                        help="all_linear 或模块名片段，如: to_q,to_k,to_v,to_out")
    parser.add_argument('--lora_dropout', type=float, default=0.0)


    # ===== 多 GPU 分片 =====
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    args.residual_use_layernorm = bool(args.residual_use_layernorm)
    return args


# ============================================================
# 主流程
# ============================================================
def main(args):
    # ===== 加载 metadata =====
    if not os.path.exists(args.metadata_file):
        raise FileNotFoundError(f"metadata 文件不存在：{args.metadata_file}")
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    total_items = len(metadatas)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ===== 手动分片 =====
    if args.world_size > 1:
        local_indices = [i for i in range(total_items) if i % args.world_size == args.rank]
    else:
        local_indices = list(range(total_items))

    print(f"[Rank {args.rank}] 总任务: {total_items}，本卡处理: {len(local_indices)}")

    # ===== 初始化生成器（每张卡一次）=====
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

    generator = SD3ImageGenerator(
        model='sd3',
        load_dir=None,
        residual_target_layers=args.residual_target_layers,
        residual_origin_layer=args.residual_origin_layer,
        residual_weights=args.residual_weights,
        residual_use_layernorm=args.residual_use_layernorm,
        residual_rotation_matrices=residual_rotation_matrices,
        residual_rotation_meta=residual_rotation_meta,
        residual_timestep_weight_fn=build_timestep_residual_weight_fn(
            args.timestep_residual_weight_fn,
            power=args.timestep_residual_weight_power,
            exp_alpha=args.timestep_residual_weight_exp_alpha,
        ),
    )

    # ---------- 如果提供了 LoRA ckpt，注入 + 加载 ----------
    if args.lora_ckpt is not None:
        print(f"[LoRA] injecting & loading LoRA from: {args.lora_ckpt}")
        target = "all_linear" if args.lora_target == "all_linear" else tuple(args.lora_target.split(","))
        # 对 sampler.denoiser（SD3Transformer2DModel_Vanilla）里的 transformer 注入
        inject_lora(generator.sampler.denoiser, rank=args.lora_rank, alpha=args.lora_alpha,
                    target=target, dropout=args.lora_dropout)
        generator.sampler.denoiser.to(device=device, dtype=torch.float32)   # 就地转换
        lora_sd = torch.load(args.lora_ckpt, map_location="cpu")
        load_lora_state_dict(generator.sampler.denoiser, lora_sd, strict=True)
        
        generator.sampler.denoiser.eval()
        print("[LoRA] loaded and ready.")
        




    # ===== 遍历当前 rank 的 metadata =====
    for index in local_indices:
        metadata = metadatas[index]
        prompt = metadata["prompt"]

        outpath = os.path.join(args.outdir, f"{index:05d}")
        sample_path = os.path.join(outpath, "samples")

        os.makedirs(sample_path, exist_ok=True)

        print(f"[Rank {args.rank}] Prompt {index:05d}/{total_items}: {prompt}")

        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        all_samples = []

        # 一个 prompt 多次采样
        with torch.no_grad():
            for _ in trange(
                (args.n_samples + args.batch_size - 1) // args.batch_size,
                desc=f"[Rank {args.rank}] Sampling {index:05d}",
            ):
                current_bs = min(args.batch_size, args.n_samples - sample_count)

                for _ in range(current_bs):
                    img_path = os.path.join(sample_path, f"{sample_count:05d}.png")

                    if os.path.exists(img_path):
                        print(f"[Rank {args.rank}] 跳过已存在: {img_path}")
                        sample_count += 1
                        continue

                    image = generator.generate_image(
                        prompt=prompt,
                        seed=args.seed + sample_count,
                        residual_target_layers=args.residual_target_layers,
                        residual_origin_layer=args.residual_origin_layer,
                        residual_weights=args.residual_weights,
                        residual_use_layernorm=args.residual_use_layernorm,
                        residual_rotation_matrices=residual_rotation_matrices,
                        residual_timestep_weight_fn=generator.residual_timestep_weight_fn,
                    )

                    # ====== 新增：统一成 [C, H, W] ======
                    if image.dim() == 4:
                        # [B, C, H, W] -> 取第一个
                        image = image[0]

                    if image.dim() == 3:
                        # 如果是 [H, W, 3]，转为 [3, H, W]
                        if image.shape[0] != 3 and image.shape[-1] == 3:
                            image = image.permute(2, 0, 1)
                    else:
                        raise ValueError(f"Unexpected image shape: {image.shape}")
                    # ===================================

                    save_image(image, img_path, normalize=True)



                    sample_count += 1




        del all_samples

    print(f"[Rank {args.rank}] 完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
