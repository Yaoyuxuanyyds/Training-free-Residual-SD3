#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute orthogonal Procrustes rotations for SD3 text residual alignment."""

import argparse
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from dataset.datasets import get_target_dataset
from sampler import StableDiffusion3Base


def load_and_resize_pil(image_source, height: int, width: int) -> Image.Image:
    if isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    elif torch.is_tensor(image_source):
        tensor = image_source.detach().cpu()
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.dim() != 3:
            raise ValueError(f"Unexpected tensor shape for image: {tensor.shape}")
        if tensor.min() < 0:
            tensor = (tensor + 1.0) / 2.0
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif tensor.is_floating_point() and tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0)
        import torchvision.transforms as T
        to_pil = T.ToPILImage()
        img = to_pil(tensor)
    else:
        img = Image.open(image_source).convert("RGB")
    img = img.resize((width, height), Image.BICUBIC)
    return img


def pil_to_tensor(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    import torchvision.transforms as T
    t = T.ToTensor()
    return t(pil_img).unsqueeze(0).to(device)


def encode_image_to_latent(base: StableDiffusion3Base, img_tensor: torch.Tensor) -> torch.Tensor:
    vae = base.vae
    scaling = vae.config.scaling_factor
    shift = vae.config.shift_factor
    with torch.no_grad():
        img_tensor = img_tensor.to(dtype=vae.dtype)
        posterior = vae.encode(img_tensor * 2 - 1)
        latent_pre = posterior.latent_dist.sample()
        z0 = (latent_pre - shift) * scaling
    return z0


def build_noisy_latent_like_training(
    scheduler,
    clean_latent: torch.Tensor,
    timestep_idx: int,
    generator: Optional[torch.Generator] = None,
):
    device = clean_latent.device
    total = int(scheduler.config.num_train_timesteps)
    bsz = clean_latent.shape[0]
    t_tensor = torch.full((bsz,), timestep_idx, device=device, dtype=torch.long)
    s = (t_tensor.float() / float(total)).view(bsz, 1, 1, 1)
    if generator is None:
        x1 = torch.randn_like(clean_latent)
    else:
        x1 = torch.randn(
            clean_latent.shape,
            device=clean_latent.device,
            dtype=clean_latent.dtype,
            generator=generator,
        )
    noisy = (1.0 - s) * clean_latent + s * x1
    return noisy, t_tensor


def _normalize_prompt(prompt_value) -> str:
    if isinstance(prompt_value, (list, tuple)):
        if not prompt_value:
            raise ValueError("Prompt list is empty.")
        prompt_value = prompt_value[0]
    return str(prompt_value)


def _extract_pair(sample) -> Tuple[object, str]:
    if isinstance(sample, dict):
        image = sample.get("image") or sample.get("img") or sample.get("pixel_values")
        prompt_val = sample.get("prompt") or sample.get("caption") or sample.get("text")
        if image is None or prompt_val is None:
            raise ValueError("Unable to extract image/prompt from dataset dictionary sample.")
        return image, prompt_val
    if isinstance(sample, (list, tuple)):
        if len(sample) < 2:
            raise ValueError("Dataset sample tuple does not contain both image and prompt.")
        return sample[0], sample[1]
    raise TypeError(f"Unsupported dataset sample type: {type(sample)}")


def _build_dataset(args: argparse.Namespace):
    if args.dataset is None or len(args.dataset) == 0:
        return None
    if not args.datadir:
        raise ValueError("--datadir must be provided when using --dataset.")
    datasets = [
        get_target_dataset(name, args.datadir, train=args.dataset_train, transform=None)
        for name in args.dataset
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    total_available = len(dataset)
    if total_available == 0:
        raise ValueError("Loaded dataset is empty.")
    if args.num_samples > 0 and args.num_samples < total_available:
        indices = list(range(args.num_samples))
        dataset = Subset(dataset, indices)
    return dataset


def _iterate_pairs(args: argparse.Namespace, dataset):
    if dataset is None:
        if args.prompt is None or args.image is None:
            raise ValueError("--prompt and --image must be provided when not using --dataset.")
        yield 0, args.image, args.prompt
        return
    for idx in range(len(dataset)):
        image_data, prompt_value = _extract_pair(dataset[idx])
        yield idx, image_data, prompt_value




def simulate_step_rmsnorm(chunks: List[torch.Tensor]) -> List[torch.Tensor]:
    """模拟推理时的 RMSNorm (行归一化)."""
    rms_chunks = []
    for x in chunks:
        # x shape: [num_tokens, d]
        if x.shape[0] == 0:
            rms_chunks.append(x)
            continue
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        x_norm = x / rms
        rms_chunks.append(x_norm)
    return rms_chunks


def run(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    precision = args.precision.lower()
    if precision == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported precision: {args.precision}")

    # 初始化 SD3 基础模型
    base = StableDiffusion3Base(
        model_key=args.model,
        device=args.device,
        dtype=dtype,
        use_8bit=False,
        load_ckpt_path=args.load_ckpt,
        load_transformer_only=False,
    )

    denoiser = base.denoiser
    denoiser.eval().requires_grad_(False)
    denoiser_base = getattr(denoiser, "module", denoiser)
    denoiser_base = getattr(denoiser_base, "base_model", denoiser_base)
    
    if getattr(denoiser_base, "gradient_checkpointing", False):
        denoiser_base.gradient_checkpointing = False

    num_layers = len(denoiser_base.transformer_blocks)
    if args.target_layers:
        target_layers = sorted(set(args.target_layers))
    else:
        target_layers = list(range(args.target_layer_start, num_layers))
    
    if not target_layers:
        raise ValueError("No target layers specified for Procrustes computation.")

    # 加载数据集
    dataset = _build_dataset(args)
    total_pairs = len(dataset) if dataset is not None else 1
    
    # 存储原始特征的列表
    origin_chunks: List[torch.Tensor] = []
    target_chunks: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}

    pair_iter: Iterable = _iterate_pairs(args, dataset)
    if dataset is not None:
        pair_iter = tqdm(pair_iter, total=total_pairs, desc="Collecting states")

    # --- 阶段 1: 收集原始隐藏状态 ---
    if args.timesteps:
        timesteps = [int(t) for t in args.timesteps]
    else:
        timesteps = None
    if timesteps is not None:
        print(f"[INFO] Using fixed timesteps for feature collection: {timesteps}")
    else:
        print(
            "[INFO] Using random timesteps for feature collection: "
            f"{args.num_timesteps} per sample"
        )

    for pair_idx, image_data, prompt_value in pair_iter:
        prompt = _normalize_prompt(prompt_value)
        gt_pil = load_and_resize_pil(image_data, args.height, args.width)
        gt_tensor = pil_to_tensor(gt_pil, device=device)
        z0 = encode_image_to_latent(base, gt_tensor)

        prompt_emb, pooled_emb, token_mask = base.encode_prompt([prompt], batch_size=1)
        token_mask = token_mask[0].to(torch.bool)
        if not args.use_padding_mask or token_mask.sum() == 0:
            token_mask = None

        if timesteps is None:
            gen_cpu = torch.Generator(device="cpu")
            gen_cpu.manual_seed(int(args.seed or 0) + pair_idx)
            sample_timesteps = torch.randint(
                0, 1000, (args.num_timesteps,), generator=gen_cpu
            ).tolist()
        else:
            sample_timesteps = timesteps

        for t_offset, timestep_idx in enumerate(sample_timesteps):
            gen_cuda = torch.Generator(device=device)
            gen_cuda.manual_seed(
                int(args.seed or 0) + pair_idx * len(sample_timesteps) + t_offset
            )
            z_t, t_tensor = build_noisy_latent_like_training(
                base.scheduler, z0, timestep_idx, generator=gen_cuda
            )

            with torch.no_grad():
                outputs = denoiser(
                    hidden_states=z_t.to(dtype=denoiser.dtype),
                    timestep=t_tensor,
                    encoder_hidden_states=prompt_emb.to(dtype=denoiser.dtype),
                    pooled_projections=pooled_emb.to(dtype=denoiser.dtype),
                    return_dict=False,
                    output_text_inputs=True,
                )
            
            txt_input_states = outputs.get("txt_input_states")
            
            # 提取并掩码 Origin Layer
            origin_state = txt_input_states[args.origin_layer][0].float().cpu()
            if token_mask is not None:
                origin_state = origin_state[token_mask.cpu()]
            origin_chunks.append(origin_state)

            # 提取并掩码 Target Layers
            for layer in target_layers:
                target_state = txt_input_states[layer][0].float().cpu()
                if token_mask is not None:
                    target_state = target_state[token_mask.cpu()]
                target_chunks[layer].append(target_state)

    # --- 阶段 2: 模拟推理分布（行归一化） ---
    def apply_simulated_rmsnorm(chunks_list: List[torch.Tensor]) -> torch.Tensor:
        processed = []
        for x in chunks_list:
            # x: [num_tokens, d]
            if x.shape[0] == 0: continue
            # 行归一化：模拟推理时的 RMSNorm
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            processed.append(x / rms)
        return torch.cat(processed, dim=0)

    print("[PROCESS] Applying Row-wise RMSNorm and Column-wise Centering...")
    # 得到模拟推理分布后的 X
    X_ln = apply_simulated_rmsnorm(origin_chunks)
    # Token-wise 中心化
    X_final = X_ln - X_ln.mean(dim=-1, keepdim=True)

    rotations: List[torch.Tensor] = []
    # --- 阶段 3: 计算各层的正交旋转矩阵 ---
    for layer in target_layers:
        Y_ln = apply_simulated_rmsnorm(target_chunks[layer])
        # Token-wise 中心化
        Y_final = Y_ln - Y_ln.mean(dim=-1, keepdim=True)

        # 计算相关矩阵 C (使用 float32 保证 SVD 精度)
        C = X_final.t().matmul(Y_final).to(torch.float32)
        
        # SVD 分解求解正交普氏问题
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        R = U.matmul(Vh)
        rotations.append(R)
        
        # 打印拟合质量（F-范数残差参考）
        res = torch.norm(X_final.matmul(R) - Y_final, p='fro')
        print(f"Layer {layer} alignment residual (Frobenius): {res:.4f}")

    # --- 阶段 4: 保存结果 ---
    rotation_stack = torch.stack(rotations, dim=0)
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    payload = {
        "origin_layer": args.origin_layer,
        "target_layers": target_layers,
        "rotation_matrices": rotation_stack,
        "feature_dim": X_final.shape[1],
        "num_valid_tokens": X_final.shape[0],
        "strategy": "row_rmsnorm_then_token_center_random_timestep",
        "timesteps": timesteps,
        "num_timesteps": args.num_timesteps,
    }
    torch.save(payload, args.output)
    print(f"[DONE] Saved Procrustes rotations to {args.output}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3")
    parser.add_argument("--load-ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])

    parser.add_argument("--dataset", type=str, nargs="+", default=None)
    parser.add_argument("--datadir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--dataset-train", action="store_true")

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--origin-layer", type=int, default=1)
    parser.add_argument("--target-layer-start", type=int, default=2)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    parser.add_argument("--no-padding-mask", action="store_false", dest="use_padding_mask", default=True)
    parser.add_argument("--output", type=str, default="procrustes_rotations.pt")
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Fixed timesteps to sample per sample for Procrustes (overrides random sampling).",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=5,
        help="Number of random timesteps to sample per sample when --timesteps is not set.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
