#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute gradient-based text token influence metrics across SD3 layers.

This script measures token influence via gradient norms of the denoiser output
with respect to text hidden states, then aggregates dataset-level statistics:
- Mean influence strength
- Top-k mass
- Entropy
It saves three line plots for the metrics across layers.
"""

import argparse
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from sampler import StableDiffusion3Base
from lora_utils import inject_lora, load_lora_state_dict
from dataset.datasets import get_target_dataset
from torch.utils.data import ConcatDataset, Subset


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# LoRA helpers
# -----------------------------------------------------------------------------

def _parse_lora_target(target: str):
    if target is None or target == "all_linear":
        return "all_linear"
    parts = [t.strip() for t in target.split(",") if t.strip()]
    return tuple(parts)


def _maybe_apply_lora(
    denoiser,
    device: torch.device,
    dtype: torch.dtype,
    ckpt: Optional[str],
    rank: int,
    alpha: int,
    target: str,
    dropout: float,
    strict: bool,
):
    if not ckpt:
        return
    target = _parse_lora_target(target)
    print(f"[LoRA] injecting adapters (target={target})")
    inject_lora(
        denoiser,
        rank=rank,
        alpha=alpha,
        target=target,
        dropout=dropout,
        is_train=False,
    )
    denoiser.to(device=device, dtype=dtype)

    print(f"[LoRA] loading weights from {ckpt}")
    lora_sd = torch.load(ckpt, map_location="cpu")
    load_lora_state_dict(denoiser, lora_sd, strict=strict)
    denoiser.to(device=device, dtype=dtype)
    print("[LoRA] ready.")


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def compute_metrics(scores: torch.Tensor, topk: int, warn_prefix: str = ""):
    scores = scores.float()
    if scores.numel() == 0:
        if warn_prefix:
            print(f"[WARN] {warn_prefix} empty score tensor; returning zeros.")
        return 0.0, 0.0, 0.0
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    total = scores.sum()
    if not torch.isfinite(total) or total <= 0:
        if warn_prefix:
            print(f"[WARN] {warn_prefix} non-positive or non-finite score sum; returning zeros.")
        return 0.0, 0.0, 0.0
    mean_strength = scores.mean().item()
    k = max(1, min(topk, scores.numel()))
    topk_mass = scores.topk(k).values.sum().div(total).item()
    p = scores / total
    eps = 1e-12
    entropy = -(p * (p + eps).log()).sum().item()
    return mean_strength, topk_mass, entropy


def build_timesteps(args, scheduler) -> Sequence[int]:
    if args.timestep_idxs:
        return list(dict.fromkeys(args.timestep_idxs))
    if args.num_timesteps > 1:
        max_t = int(scheduler.config.num_train_timesteps) - 1
        steps = np.linspace(0, max_t, num=args.num_timesteps)
        return [int(s) for s in steps]
    return [args.timestep_idx]


def build_seed_list(args) -> Sequence[int]:
    if args.num_seeds <= 1:
        return [args.seed] if args.seed is not None else [0]
    base = args.seed if args.seed is not None else 0
    return [base + i for i in range(args.num_seeds)]


class TextStateCollector:
    """Collect encoder_hidden_states inputs or outputs from transformer blocks."""

    def __init__(self, blocks, capture: str = "input"):
        if capture not in {"input", "output"}:
            raise ValueError(f"Unsupported capture mode: {capture}")
        self._handles = []
        self.states = []
        self.capture = capture

        if capture == "input":
            def _hook(_module, inputs, kwargs):
                if kwargs and "encoder_hidden_states" in kwargs:
                    self.states.append(kwargs["encoder_hidden_states"])
                    return
                if len(inputs) >= 2:
                    self.states.append(inputs[1])
                    return
                raise RuntimeError("Transformer block inputs missing encoder_hidden_states.")

            for block in blocks:
                self._handles.append(block.register_forward_pre_hook(_hook, with_kwargs=True))
        else:
            def _hook(_module, _inputs, outputs):
                if isinstance(outputs, (tuple, list)) and outputs:
                    self.states.append(outputs[0])
                else:
                    raise RuntimeError("Unexpected transformer block output; cannot collect text states.")

            for block in blocks:
                self._handles.append(block.register_forward_hook(_hook))

    def clear(self):
        self.states = []

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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
    _maybe_apply_lora(
        denoiser=denoiser,
        device=device,
        dtype=dtype,
        ckpt=args.lora_ckpt,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target=args.lora_target,
        dropout=args.lora_dropout,
        strict=args.lora_strict,
    )
    denoiser.eval().requires_grad_(False)
    denoiser_base = getattr(denoiser, "module", denoiser)
    denoiser_base = getattr(denoiser_base, "base_model", denoiser_base)
    if getattr(denoiser_base, "gradient_checkpointing", False):
        print("[INFO] Disabling gradient checkpointing to capture intermediate text gradients.")
        denoiser_base.gradient_checkpointing = False
    if not hasattr(denoiser_base, "transformer_blocks"):
        raise AttributeError("Denoiser base model missing transformer_blocks; cannot collect text states.")
    target_layers = sorted(set(args.layers))
    if not target_layers:
        raise ValueError("No transformer layers specified for evaluation.")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_mode = args.dataset is not None and len(args.dataset) > 0
    dataset = None
    total_pairs = 1
    if dataset_mode:
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
        total_pairs = len(dataset)
        print(f"[DATASET] Using {total_pairs} image-text pairs.")
    else:
        if args.prompt is None or args.image is None:
            raise ValueError("--prompt and --image must be provided when not using --dataset.")
        print("[SINGLE] Evaluating a single image-text pair.")

    def normalize_prompt(prompt_value):
        if isinstance(prompt_value, (list, tuple)):
            if not prompt_value:
                raise ValueError("Prompt list is empty.")
            prompt_value = prompt_value[0]
        return str(prompt_value)

    def extract_pair(sample):
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

    def iterate_pairs():
        if dataset_mode:
            for idx in range(total_pairs):
                image_data, prompt_value = extract_pair(dataset[idx])
                yield idx, image_data, prompt_value
        else:
            yield 0, args.image, args.prompt

    timesteps = build_timesteps(args, base.scheduler)
    seeds = build_seed_list(args)
    total_inner = len(timesteps) * len(seeds)

    layer_strength_records = {layer: [] for layer in target_layers}
    layer_topk_records = {layer: [] for layer in target_layers}
    layer_entropy_records = {layer: [] for layer in target_layers}

    processed_pairs = 0

    for pair_idx, image_data, prompt_value in iterate_pairs():
        prompt = normalize_prompt(prompt_value)
        prompt_preview = prompt if len(prompt) <= 60 else prompt[:57] + "..."
        gt_pil = load_and_resize_pil(image_data, args.height, args.width)
        gt_tensor = pil_to_tensor(gt_pil, device=device)
        if gt_tensor.min() < 0 or gt_tensor.max() > 1:
            raise ValueError(
                f"gt_tensor values out of [0,1] after pil_to_tensor: "
                f"min={gt_tensor.min().item():.4f}, max={gt_tensor.max().item():.4f}"
            )
        z0 = encode_image_to_latent(base, gt_tensor)

        prompt_emb, pooled_emb, token_mask = base.encode_prompt([prompt], batch_size=1)
        token_mask = token_mask[0].to(torch.bool)
        if not args.ignore_padding:
            token_mask = None
        elif token_mask.sum() == 0:
            print("[WARN] Token mask has no valid tokens; disabling padding mask.")
            token_mask = None

        prompt_emb = prompt_emb.to(dtype=denoiser.dtype).detach().requires_grad_(True)
        pooled_emb = pooled_emb.to(dtype=denoiser.dtype).detach().requires_grad_(True)

        layer_sum_scores = {layer: None for layer in target_layers}
        effective_counts = {layer: 0 for layer in target_layers}

        for t_idx, timestep_idx in enumerate(timesteps):
            for s_idx, seed in enumerate(seeds):
                gen = torch.Generator(device=device)
                gen.manual_seed(int(seed) + pair_idx * 1000 + t_idx * 100 + s_idx)

                z_t, t_tensor = build_noisy_latent_like_training(
                    base.scheduler, z0, timestep_idx, generator=gen
                )
                z_t = z_t.to(dtype=denoiser.dtype)

                with torch.enable_grad():
                    outputs = denoiser(
                        hidden_states=z_t,
                        timestep=t_tensor,
                        encoder_hidden_states=prompt_emb,
                        pooled_projections=pooled_emb,
                        return_dict=False,
                        output_hidden_states=True,
                        force_txt_grad=args.force_txt_grad,
                    )

                    pred = outputs["sample"]
                    if not torch.isfinite(pred).all():
                        print(f"[WARN] Non-finite denoiser output at timestep={timestep_idx}, seed={seed}.")
                    y = 0.5 * (pred.float() ** 2).sum()

                    txt_hidden_states_list = outputs.get("txt_hidden_states")
                    if txt_hidden_states_list is None:
                        raise KeyError(
                            "Missing txt_hidden_states in denoiser outputs; "
                            "ensure output_hidden_states=True in the denoiser forward."
                        )
                    max_layer = max(target_layers)
                    if max_layer >= len(txt_hidden_states_list):
                        raise ValueError(
                            f"Requested layer {max_layer} but only {len(txt_hidden_states_list)} layers were recorded."
                        )
                    target_states = [txt_hidden_states_list[layer][0] for layer in target_layers]
                    for layer, state in zip(target_layers, target_states):
                        if not state.requires_grad:
                            raise RuntimeError(
                                f"Text hidden state at layer {layer} does not require grad. "
                                "Ensure force_txt_grad=True and run the denoiser under torch.enable_grad()."
                            )

                    grads = torch.autograd.grad(
                        y,
                        target_states,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )

                for layer, grad, state in zip(target_layers, grads, target_states):
                    scores = grad.float().norm(dim=-1)
                    if token_mask is not None:
                        scores = scores[token_mask]
                    scores = scores.detach().cpu()

                    if layer_sum_scores[layer] is None:
                        layer_sum_scores[layer] = scores.clone()
                    else:
                        layer_sum_scores[layer] += scores
                    effective_counts[layer] += 1

                if prompt_emb.grad is not None:
                    prompt_emb.grad = None
                if pooled_emb.grad is not None:
                    pooled_emb.grad = None

        for layer in target_layers:
            if layer_sum_scores[layer] is None or effective_counts[layer] == 0:
                continue
            avg_scores = layer_sum_scores[layer] / float(effective_counts[layer])
            mean_strength, topk_mass, entropy = compute_metrics(
                avg_scores, args.topk, warn_prefix=f"Layer {layer:02d}"
            )
            layer_strength_records[layer].append(mean_strength)
            layer_topk_records[layer].append(topk_mass)
            layer_entropy_records[layer].append(entropy)

        processed_pairs += 1
        print(
            f"[PAIR {processed_pairs}/{total_pairs}] prompt='{prompt_preview}' | "
            f"timesteps={len(timesteps)} seeds={len(seeds)}"
        )

    if processed_pairs == 0:
        print("[WARN] No valid pairs processed.")
        return

    print(f"[STATS] Processed {processed_pairs} pairs. Aggregation: {total_inner} samples/pair")

    results = []
    for layer in target_layers:
        strengths = layer_strength_records[layer]
        topk_vals = layer_topk_records[layer]
        entropies = layer_entropy_records[layer]
        mean_strength = float(np.mean(strengths)) if strengths else float("nan")
        mean_topk = float(np.mean(topk_vals)) if topk_vals else float("nan")
        mean_entropy = float(np.mean(entropies)) if entropies else float("nan")
        results.append((layer, mean_strength, mean_topk, mean_entropy))
        print(
            f"Layer {layer:02d}: strength={mean_strength:.6f}, "
            f"topk_mass={mean_topk:.6f}, entropy={mean_entropy:.6f}"
        )

    xs = [r[0] for r in results]
    ys_strength = [r[1] for r in results]
    ys_topk = [r[2] for r in results]
    ys_entropy = [r[3] for r in results]

    def plot_curve(values, ylabel, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(xs, values, marker="o")
        plt.xlabel("Layer index")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, filename)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[SAVE] {ylabel} curve saved to {out_path}")

    plot_curve(ys_strength, "Mean influence strength", args.output_strength)
    plot_curve(ys_topk, f"Top-{args.topk} mass", args.output_topk)
    plot_curve(ys_entropy, "Entropy", args.output_entropy)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute gradient-based token influence metrics across SD3 layers.")
    p.add_argument("--model", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3")
    p.add_argument("--prompt", type=str, default=None, help="Prompt text when evaluating a single pair")
    p.add_argument("--image", type=str, default=None, help="Image path when evaluating a single pair")
    p.add_argument("--timestep-idx", dest="timestep_idx", type=int, required=True)
    p.add_argument("--timestep-idxs", dest="timestep_idxs", type=int, nargs="+", default=None)
    p.add_argument("--num-timesteps", type=int, default=1, help="Evenly sample this many timesteps if timestep-idxs is not set")
    p.add_argument("--num-seeds", type=int, default=1, help="Number of noise seeds per timestep")
    p.add_argument("--layers", type=int, nargs="+", default=list(range(1, 23)))
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Computation precision for denoiser and embeddings.",
    )
    p.add_argument("--load-ckpt", dest="load_ckpt", type=str, default=None)
    p.add_argument("--lora-ckpt", dest="lora_ckpt", type=str, default=None)
    p.add_argument("--lora-rank", dest="lora_rank", type=int, default=8)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=16)
    p.add_argument("--lora-target", dest="lora_target", type=str, default="all_linear")
    p.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.0)
    p.add_argument("--lora-strict", dest="lora_strict", action="store_true")
    p.add_argument("--output-dir", type=str, default="attn_vis_out")
    p.add_argument("--output-strength", type=str, default="grad_strength_curve.png")
    p.add_argument("--output-topk", type=str, default="grad_topk_mass_curve.png")
    p.add_argument("--output-entropy", type=str, default="grad_entropy_curve.png")
    p.add_argument("--ignore-padding", action="store_true", help="Use attention mask to skip padded tokens")
    p.add_argument(
        "--force-txt-grad",
        action="store_true",
        default=True,
        help="Force text hidden states to require grad (default: enabled).",
    )
    p.add_argument(
        "--no-force-txt-grad",
        action="store_false",
        dest="force_txt_grad",
        help="Disable forcing text hidden states to require grad.",
    )
    p.add_argument("--dataset", type=str, nargs="+", default=None, help="Dataset names to evaluate (e.g., coco)")
    p.add_argument("--datadir", type=str, default=None, help="Root directory containing datasets")
    p.add_argument("--num-samples", type=int, default=-1, help="Number of pairs to evaluate from the dataset (-1 for all)")
    p.add_argument("--dataset-train", action="store_true", help="Use the training split of the dataset")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
