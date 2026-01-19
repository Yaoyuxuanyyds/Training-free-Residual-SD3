#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute orthogonal Procrustes rotations for Qwen-Image residual alignment."""

import argparse
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from datasets import get_target_dataset
from sampler import MyQwenImagePipeline, calculate_shift, retrieve_timesteps


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


def _add_noise(scheduler, latents: torch.Tensor, timestep: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(latents, noise, timestep)

    total = int(scheduler.config.num_train_timesteps)
    s = (timestep.float() / float(total)).view(-1, *([1] * (latents.dim() - 1)))
    return (1.0 - s) * latents + s * noise


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


def _bucket_timesteps(timesteps: torch.Tensor, bucket_edges: List[int]) -> List[List[int]]:
    timesteps_list = timesteps.detach().cpu().tolist()
    bucket_indices: List[List[int]] = []
    for bucket_idx in range(len(bucket_edges) - 1):
        start = bucket_edges[bucket_idx]
        end = bucket_edges[bucket_idx + 1]
        indices = [i for i, t in enumerate(timesteps_list) if start <= t < end]
        if not indices:
            center = (start + end) / 2.0
            closest_idx = min(
                range(len(timesteps_list)),
                key=lambda i: abs(timesteps_list[i] - center),
            )
            indices = [closest_idx]
        bucket_indices.append(indices)
    return bucket_indices


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

    pipe = MyQwenImagePipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    denoiser = pipe.transformer
    denoiser.eval().requires_grad_(False)

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

        prompt_emb, prompt_emb_mask = pipe.encode_prompt(
            prompt=[prompt],
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
        )

        if args.image is not None or dataset is not None:
            _ = load_and_resize_pil(image_data, args.height, args.width)

        num_channels_latents = denoiser.config.in_channels // 4
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=args.height,
            width=args.width,
            dtype=prompt_emb.dtype,
            device=device,
            generator=None,
            latents=None,
        )

        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            args.num_inference_steps,
            device,
            sigmas=None,
            mu=mu,
        )

        bucket_indices = _bucket_timesteps(timesteps, bucket_edges)

        for bucket_idx, candidates in enumerate(bucket_indices):
            gen_cpu = torch.Generator(device="cpu")
            gen_cpu.manual_seed(int(args.seed or 0) + pair_idx * num_buckets + bucket_idx)
            if len(candidates) == 1:
                t_index = candidates[0]
            else:
                t_index = int(
                    torch.randint(0, len(candidates), (1,), generator=gen_cpu).item()
                )
                t_index = candidates[t_index]
            t = timesteps[t_index]
            t_tensor = t.expand(latents.shape[0]).to(latents.dtype)

            gen_cuda = torch.Generator(device=device)
            gen_cuda.manual_seed(int(args.seed or 0) + pair_idx * num_buckets + bucket_idx)
            z_t = _add_noise(pipe.scheduler, latents, t_tensor, generator=gen_cuda)

            img_shapes = [[(1, args.height // pipe.vae_scale_factor // 2, args.width // pipe.vae_scale_factor // 2)]]
            txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist() if prompt_emb_mask is not None else None

            with torch.no_grad():
                outputs = denoiser(
                    hidden_states=z_t.to(dtype=denoiser.dtype),
                    timestep=t_tensor / 1000,
                    encoder_hidden_states=prompt_emb.to(dtype=denoiser.dtype),
                    encoder_hidden_states_mask=prompt_emb_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                    output_text_inputs=True,
                )

            txt_input_states = outputs.get("txt_input_states")
            token_mask = prompt_emb_mask[0].to(torch.bool) if prompt_emb_mask is not None else None

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
    X_final = None

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
            U, _, Vh = torch.linalg.svd(C, full_matrices=False)
            R = U.matmul(Vh)
            rotations.append(R)

            res = torch.norm(X_final.matmul(R) - Y_final, p="fro")
            print(
                f"[Bucket {bucket_idx}] Layer {layer} alignment residual (Frobenius): {res:.4f}"
            )

        rotations_by_bucket.append(torch.stack(rotations, dim=0))

    if num_buckets == 1:
        rotation_stack = rotations_by_bucket[0]
        num_valid_tokens_payload = num_valid_tokens[0]
    else:
        rotation_stack = torch.stack(rotations_by_bucket, dim=0)
        num_valid_tokens_payload = num_valid_tokens
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    payload = {
        "origin_layer": args.origin_layer,
        "target_layers": target_layers,
        "rotation_matrices": rotation_stack,
        "feature_dim": X_final.shape[1] if X_final is not None else None,
        "num_valid_tokens": num_valid_tokens_payload,
        "strategy": "row_ln_then_col_center" if args.col_center else "row_ln",
        "column_center": args.col_center,
        "timestep_buckets": num_buckets,
        "timestep_bucket_edges": bucket_edges,
    }
    torch.save(payload, args.output)
    print(f"[DONE] Saved Procrustes rotations to {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image")
    parser.add_argument("--output", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual/output/procrustes_rotations/qwen_procrustes_rotations.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, nargs="*", default=["blip3o60k"])
    parser.add_argument("--datadir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/datasets")
    parser.add_argument("--dataset_train", action="store_true")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument("--origin_layer", type=int, default=1)
    parser.add_argument("--target_layers", type=int, nargs="*", default=None)
    parser.add_argument("--target_layer_start", type=int, default=2)
    parser.add_argument("--col-center", action="store_true", help="Enable column-wise centering.")
    parser.add_argument("--timestep-buckets", type=int, default=1)

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
