#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute CKNNA & cosine similarities between SD3 text features across layers.

Supports both single image-text pairs and batched evaluation over datasets. The
script aggregates metrics across pairs, averages them, and performs PCA
visualizations on sampled token embeddings collected from all processed pairs.
PCA visualization data can be dumped for external plotting.
"""

import argparse
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from sampler import StableDiffusion3Base
from util import AlignmentMetrics
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


def build_noisy_latent_like_training(scheduler, clean_latent: torch.Tensor, timestep_idx: int):
    device = clean_latent.device
    total = int(scheduler.config.num_train_timesteps)
    bsz = clean_latent.shape[0]
    t_tensor = torch.full((bsz,), timestep_idx, device=device, dtype=torch.long)
    s = (t_tensor.float() / float(total)).view(bsz, 1, 1, 1)
    x1 = torch.randn_like(clean_latent)
    noisy = (1.0 - s) * clean_latent + s * x1
    return noisy, t_tensor, timestep_idx


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
# Similarity functions
# -----------------------------------------------------------------------------
def compute_layer_cknna(layer_feats, layer0_feats, topk, unbiased, token_mask=None):
    feats_a = layer0_feats.detach().to(torch.float32)
    feats_b = layer_feats.detach().to(torch.float32)
    if token_mask is not None:
        feats_a = feats_a[token_mask]
        feats_b = feats_b[token_mask]
    if feats_a.shape[0] <= 1:
        return float("nan")
    topk = max(1, min(topk, feats_a.shape[0] - 1))
    return AlignmentMetrics.cknna(feats_a, feats_b, topk=topk, unbiased=unbiased)


def compute_layer_cosine(layer_feats, layer0_feats, token_mask=None):
    feats_a = layer0_feats.detach().to(torch.float32)
    feats_b = layer_feats.detach().to(torch.float32)
    if token_mask is not None:
        feats_a = feats_a[token_mask]
        feats_b = feats_b[token_mask]
    if feats_a.shape[0] == 0:
        return float("nan"), None
    sim = torch.nn.functional.cosine_similarity(feats_a, feats_b, dim=-1)
    return sim.mean().item(), sim.cpu().numpy()


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
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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

    target_layers = sorted(set(args.layers))
    if not target_layers:
        raise ValueError("No transformer layers specified for evaluation.")
    all_layers = [0] + target_layers

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

    layer_cknna_records = {layer: [] for layer in target_layers}
    layer_cos_records = {layer: [] for layer in target_layers}
    layer_token_collections = {layer: [] for layer in all_layers}
    token_counts = []
    processed_pairs = 0
    t_raw = args.timestep_idx

    for pair_idx, image_data, prompt_value in iterate_pairs():
        prompt = normalize_prompt(prompt_value)
        prompt_preview = prompt if len(prompt) <= 60 else prompt[:57] + "..."
        gt_pil = load_and_resize_pil(image_data, args.height, args.width)
        gt_tensor = pil_to_tensor(gt_pil, device=device)
        z0 = encode_image_to_latent(base, gt_tensor)
        z_t, t_tensor, t_raw = build_noisy_latent_like_training(base.scheduler, z0, args.timestep_idx)

        with torch.no_grad():
            prompt_emb, pooled_emb, token_mask = base.encode_prompt([prompt], batch_size=1)
            token_mask = token_mask[0].to(torch.bool)

            # Screen out the first token (e.g., <s>/<bos>)
            if token_mask.numel() > 0:
                token_mask[0] = False

            if not args.ignore_padding:
                token_mask = None

        with torch.no_grad():
            z_t = z_t.to(dtype=denoiser.dtype)
            prompt_emb = prompt_emb.to(dtype=denoiser.dtype)
            pooled_emb = pooled_emb.to(dtype=denoiser.dtype)
            outputs = denoiser(
                hidden_states=z_t,
                timestep=t_tensor,
                encoder_hidden_states=prompt_emb,
                pooled_projections=pooled_emb,
                return_dict=False,
                output_hidden_states=True,
            )

        layer0_feats = outputs["context_embedder_output"][0]
        txt_hidden_states_list = outputs["txt_hidden_states"]
        if txt_hidden_states_list is None or len(txt_hidden_states_list) == 0:
            raise ValueError("No text hidden states collected; enable output_hidden_states.")
        max_layer = max(target_layers)
        if max_layer >= len(txt_hidden_states_list):
            raise ValueError(
                f"Requested layer {max_layer} but only {len(txt_hidden_states_list)} layers were recorded."
            )
        layer_to_feats = {layer: txt_hidden_states_list[layer][0] for layer in target_layers}

        selector = token_mask if token_mask is not None else slice(None)
        layer0_valid = layer0_feats[selector].detach().to(torch.float32).cpu()
        layer_token_collections[0].append(layer0_valid)
        token_counts.append(layer0_valid.shape[0])

        for layer in target_layers:
            feats_b = layer_to_feats[layer]
            cknna_val = compute_layer_cknna(feats_b, layer0_feats, args.topk, args.unbiased, token_mask)
            cosine_mean, _ = compute_layer_cosine(feats_b, layer0_feats, token_mask)
            layer_cknna_records[layer].append(cknna_val)
            layer_cos_records[layer].append(cosine_mean)
            layer_valid = feats_b[selector].detach().to(torch.float32).cpu()
            layer_token_collections[layer].append(layer_valid)

        processed_pairs += 1
        print(f"[PAIR {processed_pairs}/{total_pairs}] prompt='{prompt_preview}' | tokens={layer0_feats.shape[0]}")

    if processed_pairs == 0:
        print("[WARN] No valid pairs processed.")
        return

    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    print(f"[STATS] Processed {processed_pairs} pairs. Avg valid tokens: {avg_tokens:.2f}")

    # ====== Plot similarity curves ======
    results = []
    for layer in target_layers:
        cknna_vals = [v for v in layer_cknna_records[layer] if not math.isnan(v)]
        cosine_vals = [v for v in layer_cos_records[layer] if not math.isnan(v)]
        mean_cknna = float(np.mean(cknna_vals)) if cknna_vals else float("nan")
        mean_cosine = float(np.mean(cosine_vals)) if cosine_vals else float("nan")
        results.append((layer, mean_cknna, mean_cosine))
        print(f"Layer {layer:02d}: mean CKNNA={mean_cknna:.6f}, mean Cosine={mean_cosine:.6f}")

    xs = [r[0] for r in results]
    ys_cknna = [r[1] for r in results]
    ys_cos = [r[2] for r in results]
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys_cknna, marker="o", label="CKNNA")
    plt.plot(xs, ys_cos, marker="s", linestyle="--", label="Mean Cosine")
    plt.xlabel("Layer index")
    plt.ylabel("Similarity vs layer 0")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_curve = os.path.join(args.output_dir, args.output_name)
    plt.savefig(out_curve, dpi=200)
    plt.close()
    print(f"[SAVE] Curve plot saved to {out_curve}")

    # ===== Dump PCA visualization data =====
    layer_feats_sampled = {}

    for layer in all_layers:
        token_list = layer_token_collections[layer]
        if not token_list:
            layer_feats_sampled[layer] = None
            continue

        feats = torch.cat(token_list, dim=0)  # [N_tokens, D]
        # feats = torch.nn.functional.layer_norm(
        #     feats, normalized_shape=(feats.shape[-1],), eps=1e-6
        # )

        # Avoid subsampling for single-sample runs to keep full token coverage
        if args.num_samples != 1 and args.vis_sample_size > 0 and feats.shape[0] > args.vis_sample_size:
            perm = torch.randperm(feats.shape[0])[: args.vis_sample_size]
            feats = feats[perm]

        layer_feats_sampled[layer] = feats.numpy()

    dump_payload = {"layers": np.array(all_layers, dtype=np.int32)}
    for layer, feats in layer_feats_sampled.items():
        if feats is not None and feats.shape[0] > 0:
            dump_payload[f"layer_{layer}"] = feats

    out_dump = os.path.join(args.output_dir, args.pca_dump_name)
    np.savez_compressed(out_dump, **dump_payload)
    print(f"[SAVE] PCA dump saved to {out_dump}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute CKNNA & cosine similarities across SD3 layers.")
    p.add_argument("--model", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3")
    p.add_argument("--prompt", type=str, default=None, help="Prompt text when evaluating a single pair")
    p.add_argument("--image", type=str, default=None, help="Image path when evaluating a single pair")
    p.add_argument("--timestep-idx", dest="timestep_idx", type=int, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=list(range(1, 23)))
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--unbiased", action="store_true")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--load-ckpt", dest="load_ckpt", type=str, default=None)
    p.add_argument("--lora-ckpt", dest="lora_ckpt", type=str, default=None)
    p.add_argument("--lora-rank", dest="lora_rank", type=int, default=8)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=16)
    p.add_argument("--lora-target", dest="lora_target", type=str, default="all_linear")
    p.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.0)
    p.add_argument("--lora-strict", dest="lora_strict", action="store_true")
    p.add_argument("--output-dir", type=str, default="attn_vis_out")
    p.add_argument("--output-name", type=str, default="cknna_cosine_curve.png")
    p.add_argument("--ignore-padding", action="store_true", help="Use attention mask to skip padded tokens")
    p.add_argument("--dataset", type=str, nargs="+", default=None, help="Dataset names to evaluate (e.g., coco)")
    p.add_argument("--datadir", type=str, default=None, help="Root directory containing datasets")
    p.add_argument("--num-samples", type=int, default=-1, help="Number of pairs to evaluate from the dataset (-1 for all)")
    p.add_argument("--dataset-train", action="store_true", help="Use the training split of the dataset")
    p.add_argument("--vis-sample-size", type=int, default=512, help="Number of tokens to sample per layer for visualization")
    p.add_argument("--pca-dump-name", type=str, default="text_emb_layers.npz", help="Filename for PCA dump data")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
