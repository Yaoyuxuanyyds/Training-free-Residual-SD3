#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute orthogonal Procrustes rotations for SD3.5 text residual alignment."""

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
from generate_image_res import SD35PipelineWithRES
from sd35_transformer_res import SD35Transformer2DModel_RES

DEFAULT_SD35_MODEL = "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5-large"

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


def encode_image_to_latent(pipe: SD35PipelineWithRES, img_tensor: torch.Tensor) -> torch.Tensor:
    vae = pipe.vae
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


def simulate_step_ln(chunks: List[torch.Tensor]) -> List[torch.Tensor]:
    """模拟推理时的 LayerNorm (行归一化)."""
    ln_chunks = []
    for x in chunks:
        if x.shape[0] == 0:
            ln_chunks.append(x)
            continue
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x_ln = (x - mean) / std
        ln_chunks.append(x_ln)
    return ln_chunks


def _build_bucket_edges(num_train_timesteps: int, num_buckets: int) -> List[int]:
    if num_buckets <= 1:
        return [0, num_train_timesteps]
    base = num_train_timesteps // num_buckets
    remainder = num_train_timesteps % num_buckets
    edges = [0]
    for bucket_idx in range(num_buckets):
        size = base + (1 if bucket_idx < remainder else 0)
        edges.append(edges[-1] + size)
    edges[-1] = num_train_timesteps
    return edges

def vae_latent_to_flux_tokens(z):
    """
    z: [B, 16, 128, 128]
    return: [B, 4096, 64]
    """
    B, C, H, W = z.shape
    assert C == 16
    assert H % 2 == 0 and W % 2 == 0

    # 2×2 patch
    z = z.reshape(B, C, H // 2, 2, W // 2, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)          # [B, H/2, W/2, C, 2, 2]
    z = z.reshape(B, (H // 2) * (W // 2), C * 4)

    return z


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

    model_path = args.load_ckpt or args.model or DEFAULT_SD35_MODEL
    pipe = SD35PipelineWithRES.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    pipe.transformer = SD35Transformer2DModel_RES(pipe.transformer)

    denoiser = pipe.transformer
    denoiser.eval().requires_grad_(False)

    if getattr(denoiser, "gradient_checkpointing", False):
        denoiser.gradient_checkpointing = False

    num_layers = len(denoiser.transformer_blocks)
    if args.target_layers:
        target_layers = sorted(set(args.target_layers))
    else:
        target_layers = list(range(args.target_layer_start, num_layers))

    if not target_layers:
        raise ValueError("No target layers specified for Procrustes computation.")

    dataset = _build_dataset(args)
    total_pairs = len(dataset) if dataset is not None else 1

    num_train_timesteps = int(pipe.scheduler.config.num_train_timesteps)
    num_buckets = max(1, int(args.timestep_buckets))
    bucket_edges = _build_bucket_edges(num_train_timesteps, num_buckets)

    origin_chunks: List[List[torch.Tensor]] = [[] for _ in range(num_buckets)]
    target_chunks: Dict[int, List[List[torch.Tensor]]] = {
        layer: [[] for _ in range(num_buckets)] for layer in target_layers
    }

    pair_iter: Iterable = _iterate_pairs(args, dataset)
    if dataset is not None:
        pair_iter = tqdm(pair_iter, total=total_pairs, desc="Collecting states")

    for pair_idx, image_data, prompt_value in pair_iter:
        prompt = _normalize_prompt(prompt_value)
        gt_pil = load_and_resize_pil(image_data, args.height, args.width)
        gt_tensor = pil_to_tensor(gt_pil, device=device)
        z0 = encode_image_to_latent(pipe, gt_tensor)

        prompt_outputs = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=False,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
        )
        if len(prompt_outputs) == 5:
            prompt_embeds, _, pooled_prompt_embeds, _, token_mask = prompt_outputs
        else:
            prompt_embeds, _, pooled_prompt_embeds, _ = prompt_outputs
            token_mask = None
        if token_mask is None and args.use_padding_mask:
            raise ValueError("encode_prompt did not return token_mask, but padding mask is enabled.")
        if token_mask is not None:
            token_mask = token_mask[0].to(torch.bool)
            if not args.use_padding_mask or token_mask.sum() == 0:
                token_mask = None

        for bucket_idx in range(num_buckets):
            start = bucket_edges[bucket_idx]
            end = bucket_edges[bucket_idx + 1]
            gen_cpu = torch.Generator(device="cpu")
            gen_cpu.manual_seed(int(args.seed or 0) + pair_idx * num_buckets + bucket_idx)
            if end <= start:
                timestep_idx = min(start, num_train_timesteps - 1)
            else:
                timestep_idx = int(torch.randint(start, end, (1,), generator=gen_cpu).item())

            gen_cuda = torch.Generator(device=device)
            gen_cuda.manual_seed(int(args.seed or 0) + pair_idx * num_buckets + bucket_idx)
            z_t, t_tensor = build_noisy_latent_like_training(
                pipe.scheduler, z0, timestep_idx, generator=gen_cuda
            )

            z_t = vae_latent_to_flux_tokens(z_t)              # [1,4096,64]


            with torch.no_grad():
                outputs = denoiser(
                    hidden_states=z_t.to(dtype=denoiser.dtype),
                    timestep=t_tensor,
                    encoder_hidden_states=prompt_embeds.to(dtype=denoiser.dtype),
                    pooled_projections=pooled_prompt_embeds.to(dtype=denoiser.dtype),
                    return_dict=False,
                    output_text_inputs=True,
                )

            txt_input_states = outputs.get("txt_input_states")

            origin_state = txt_input_states[args.origin_layer][0].float().cpu()
            if token_mask is not None:
                origin_state = origin_state[token_mask.cpu()]
            origin_chunks[bucket_idx].append(origin_state)

            for layer in target_layers:
                target_state = txt_input_states[layer][0].float().cpu()
                if token_mask is not None:
                    target_state = target_state[token_mask.cpu()]
                target_chunks[layer][bucket_idx].append(target_state)

    def apply_simulated_ln(chunks_list: List[torch.Tensor]) -> torch.Tensor:
        processed = []
        for x in chunks_list:
            if x.shape[0] == 0:
                continue
            mu = x.mean(dim=-1, keepdim=True)
            st = x.std(dim=-1, keepdim=True) + 1e-6
            processed.append((x - mu) / st)
        return torch.cat(processed, dim=0)

    print("[PROCESS] Applying Row-wise LN and optional Column-wise Centering...")

    rotations_by_bucket: List[torch.Tensor] = []
    num_valid_tokens: List[int] = []

    for bucket_idx in range(num_buckets):
        X_ln = apply_simulated_ln(origin_chunks[bucket_idx])
        X_final = (
            X_ln - X_ln.mean(dim=0, keepdim=True)
            if args.col_center
            else X_ln
        )
        num_valid_tokens.append(X_final.shape[0])

        rotations: List[torch.Tensor] = []
        for layer in target_layers:
            Y_ln = apply_simulated_ln(target_chunks[layer][bucket_idx])
            Y_final = (
                Y_ln - Y_ln.mean(dim=0, keepdim=True)
                if args.col_center
                else Y_ln
            )

            C = X_final.t().matmul(Y_final).to(torch.float32)
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
            R = U.matmul(Vh)
            rotations.append(R)

            res = torch.norm(X_final.matmul(R) - Y_final, p="fro")
            print(
                f"[Bucket {bucket_idx}] Layer {layer} alignment residual (Frobenius): {res:.4f}"
            )

        rotations_by_bucket.append(torch.stack(rotations, dim=0))

    rotations_tensor = torch.stack(rotations_by_bucket, dim=0)
    meta = {
        "origin_layer": args.origin_layer,
        "target_layers": target_layers,
        "timestep_bucket_edges": bucket_edges,
        "num_valid_tokens": num_valid_tokens,
        "num_samples": args.num_samples,
    }

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, args.output)
    torch.save(
        {
            "rotation_matrices": rotations_tensor,
            "origin_layer": args.origin_layer,
            "target_layers": target_layers,
            "meta": meta,
        },
        output_path,
    )
    print(f"Saved Procrustes rotations to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_SD35_MODEL)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--dataset", type=str, nargs="+", default=["blip3o60k"])
    parser.add_argument("--datadir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/datasets")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--dataset-train", action="store_true")

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--origin_layer", type=int, default=1)
    parser.add_argument("--target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--target_layer_start", type=int, default=2)
    parser.add_argument("--max_sequence_length", type=int, default=256)

    parser.add_argument("--timestep_buckets", type=int, default=1)
    parser.add_argument("--col_center", action="store_true")
    parser.add_argument("--no-padding-mask", action="store_false", dest="use_padding_mask", default=True)

    parser.add_argument("--outdir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/SD3.5-Residual/logs/procrustes_rotations/")
    parser.add_argument("--output", type=str, default="procrustes_rotations_coco5k_ln_t1.pt")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
