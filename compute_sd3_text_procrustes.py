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
    if not hasattr(denoiser_base, "transformer_blocks"):
        raise AttributeError("Denoiser base model missing transformer_blocks.")

    num_layers = len(denoiser_base.transformer_blocks)
    if args.target_layers:
        target_layers = sorted(set(args.target_layers))
    else:
        target_layers = list(range(args.target_layer_start, num_layers))
    if not target_layers:
        raise ValueError("No target layers specified for Procrustes computation.")
    if args.origin_layer < 0 or args.origin_layer >= num_layers:
        raise ValueError(f"origin_layer must be in [0, {num_layers - 1}]")
    if max(target_layers) >= num_layers:
        raise ValueError(f"target layer exceeds max layer index {num_layers - 1}")

    dataset = _build_dataset(args)
    total_pairs = len(dataset) if dataset is not None else 1
    if dataset is not None:
        print(f"[DATASET] Using {total_pairs} image-text pairs.")

    origin_chunks: List[torch.Tensor] = []
    target_chunks: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}

    pair_iter: Iterable = _iterate_pairs(args, dataset)
    if dataset is not None:
        pair_iter = tqdm(pair_iter, total=total_pairs, desc="Collecting states")

    for pair_idx, image_data, prompt_value in pair_iter:
        prompt = _normalize_prompt(prompt_value)
        gt_pil = load_and_resize_pil(image_data, args.height, args.width)
        gt_tensor = pil_to_tensor(gt_pil, device=device)
        z0 = encode_image_to_latent(base, gt_tensor)

        prompt_emb, pooled_emb, token_mask = base.encode_prompt([prompt], batch_size=1)
        token_mask = token_mask[0].to(torch.bool)
        if not args.use_padding_mask or token_mask.sum() == 0:
            token_mask = None

        prompt_emb = prompt_emb.to(dtype=denoiser.dtype)
        pooled_emb = pooled_emb.to(dtype=denoiser.dtype)

        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed or 0) + pair_idx)
        timestep_idx = int(
            torch.randint(
                0,
                int(base.scheduler.config.num_train_timesteps),
                (1,),
                generator=gen,
            ).item()
        )
        z_t, t_tensor = build_noisy_latent_like_training(
            base.scheduler, z0, timestep_idx, generator=gen
        )
        z_t = z_t.to(dtype=denoiser.dtype)

        with torch.no_grad():
            outputs = denoiser(
                hidden_states=z_t,
                timestep=t_tensor,
                encoder_hidden_states=prompt_emb,
                pooled_projections=pooled_emb,
                return_dict=False,
                output_text_inputs=True,
                output_hidden_states=False,
                residual_stop_grad=True,
            )
        txt_input_states = outputs.get("txt_input_states")
        if txt_input_states is None:
            raise KeyError("Missing txt_input_states in denoiser outputs.")
        if max(target_layers) >= len(txt_input_states):
            raise ValueError(
                f"Requested layer {max(target_layers)} but only {len(txt_input_states)} layers recorded."
            )

        origin_state = txt_input_states[args.origin_layer][0].float().cpu()
        if token_mask is not None:
            origin_state = origin_state[token_mask]
        origin_chunks.append(origin_state)

        for layer in target_layers:
            target_state = txt_input_states[layer][0].float().cpu()
            if token_mask is not None:
                target_state = target_state[token_mask]
            target_chunks[layer].append(target_state)

    if not origin_chunks:
        raise RuntimeError("No samples collected; cannot compute Procrustes matrices.")

    X = torch.cat(origin_chunks, dim=0)
    if X.numel() == 0:
        raise RuntimeError("Origin features are empty after masking.")

    rotations: List[torch.Tensor] = []
    for layer in target_layers:
        Y = torch.cat(target_chunks[layer], dim=0)
        if Y.shape[0] != X.shape[0]:
            raise RuntimeError(
                f"Token count mismatch for layer {layer}: X={X.shape[0]}, Y={Y.shape[0]}"
            )
        X_center = X - X.mean(dim=0, keepdim=True)
        Y_center = Y - Y.mean(dim=0, keepdim=True)
        C = X_center.t().matmul(Y_center)
        U, _, Vh = torch.linalg.svd(C, full_matrices=False)
        R = U.matmul(Vh)
        rotations.append(R)

    rotation_stack = torch.stack(rotations, dim=0)
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None

    payload = {
        "origin_layer": args.origin_layer,
        "target_layers": target_layers,
        "rotation_matrices": rotation_stack,
        "feature_dim": X.shape[1],
        "num_samples": total_pairs,
        "num_valid_tokens": X.shape[0],
        "timestep_sampling": "per-sample-random",
        "use_padding_mask": args.use_padding_mask,
    }
    torch.save(payload, args.output)
    print(f"[DONE] Saved Procrustes rotations to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sd3")
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

    parser.add_argument("--origin-layer", type=int, default=0)
    parser.add_argument("--target-layer-start", type=int, default=2)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    parser.add_argument("--no-padding-mask", action="store_false", dest="use_padding_mask", default=True)
    parser.add_argument("--output", type=str, default="procrustes_rotations.pt")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
