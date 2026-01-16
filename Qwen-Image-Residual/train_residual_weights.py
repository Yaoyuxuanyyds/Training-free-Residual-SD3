import argparse
import json
import math
import os
import os.path as osp
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import yaml

from datasets import CachedFeatureDataset_Packed, collate_fn_packed, get_target_dataset
from sampler import MyQwenImagePipeline
from util import (
    build_text_token_nonpad_mask,
    get_transform,
    load_residual_procrustes,
    select_residual_rotations,
    set_seed,
)

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

torch.multiprocessing.set_sharing_strategy("file_system")


def _get_batch_value(batch: dict, keys: Sequence[str]) -> Optional[torch.Tensor]:
    for key in keys:
        if key in batch:
            return batch[key]
    return None


def _ensure_prompt_mask(prompt_embeds: torch.Tensor, prompt_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if prompt_mask is not None:
        return prompt_mask
    if prompt_embeds is None:
        return None
    mask = build_text_token_nonpad_mask(prompt_embeds)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return mask.to(device=prompt_embeds.device)


def _pack_latents_if_needed(
    pipe: MyQwenImagePipeline,
    latents: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if hasattr(pipe, "_pack_latents"):
        return pipe._pack_latents(latents, height, width, pipe.vae_scale_factor)
    return latents


def _encode_images_to_latents(
    pipe: MyQwenImagePipeline,
    images: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if images.dim() == 4:
        images = images.unsqueeze(2)
    images = images.to(dtype=pipe.vae.dtype)
    latent_dist = pipe.vae.encode(images).latent_dist
    latents = latent_dist.sample()

    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    )
    latents_mean = latents_mean.to(device=latents.device, dtype=latents.dtype)
    latents_std = latents_std.to(device=latents.device, dtype=latents.dtype)

    latents = (latents - latents_mean) * latents_std
    return _pack_latents_if_needed(pipe, latents, height, width)


# =========================
# 时间采样函数
# =========================

def sample_timesteps(batch_size, num_steps, device, mode="uniform", **kwargs):
    if mode == "uniform":
        t = torch.randint(0, num_steps, (batch_size,), device=device)
    elif mode == "gaussian":
        mu_ratio = kwargs.get("mu_ratio", 0.6)
        sigma_ratio = kwargs.get("sigma_ratio", 0.1)
        mu = int(num_steps * mu_ratio)
        sigma = int(num_steps * sigma_ratio)
        t = torch.normal(
            mean=torch.full((batch_size,), mu, device=device, dtype=torch.float),
            std=sigma,
        )
        t = t.clamp(0, num_steps - 1).long()
    elif mode == "beta":
        alpha = kwargs.get("alpha", 5.0)
        beta = kwargs.get("beta", 2.0)
        u = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        t = (u * (num_steps - 1)).long()
    elif mode == "logitnorm":
        mu = kwargs.get("mu", 0.5)
        sigma = kwargs.get("sigma", 1.0)
        z = torch.normal(
            mean=torch.full((batch_size,), mu, device=device, dtype=torch.float),
            std=sigma,
        )
        u = torch.sigmoid(z)
        t = (u * (num_steps - 1)).long()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return t


# =========================
# 总损失 = diffusion MSE（仅 residual weights）
# =========================

def compute_total_loss(
    denoiser,
    scheduler,
    latents,
    t,
    prompt_embeds,
    prompt_embeds_mask,
    img_shapes,
    txt_seq_lens,
):
    noise = torch.randn_like(latents)
    if hasattr(scheduler, "add_noise"):
        latents_noisy = scheduler.add_noise(latents, noise, t)
    else:
        total = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        s = (t.float() / float(total)).view(-1, *([1] * (latents.dim() - 1)))
        latents_noisy = (1.0 - s) * latents + s * noise

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, t)
    elif prediction_type == "sample":
        target = latents
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    t_scaled = t.float() / float(num_train_timesteps)

    with autocast(enabled=True):
        out = denoiser(
            hidden_states=latents_noisy,
            timestep=t_scaled,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )
        noise_pred = out[0] if isinstance(out, (tuple, list)) else out["sample"]
        denoise_loss = F.mse_loss(noise_pred, target)

    return denoise_loss


# =========================
# 训练主流程
# =========================

def train(args):
    distributed = dist.is_available() and "WORLD_SIZE" in os.environ
    if distributed:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    print(f"[INFO] Use device={device}, GPUs={n_gpus}, distributed={distributed}")
    set_seed(args.seed + rank)

    global_bs = args.batch_size * world_size
    log_dir = osp.join(
        args.logdir,
        (
            f"{args.model}_residual_weights_lr-{args.lr}_bs-{global_bs}_"
            f"init-{args.init_mode}{args.residual_init}_steps-{args.steps}_"
            f"smooth-{args.residual_smoothness_weight}"
        ),
    )
    os.makedirs(log_dir, exist_ok=True)

    tb_dir = osp.join(log_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir) if rank == 0 else None

    if rank == 0:
        with open(osp.join(log_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    use_cache = args.precompute_dir is not None
    if use_cache:
        dataset = CachedFeatureDataset_Packed(
            cache_dirs=args.precompute_dir,
            target_batch_size=args.batch_size,
            cache_meta=True,
            ar_target_layer=args.ar_target_layer,
        )
    else:
        if args.datadir is None:
            raise ValueError("datadir is required when precompute_dir is not provided.")
        transform = get_transform(args.img_size)
        dataset = get_target_dataset(
            args.dataset,
            args.datadir,
            train=True,
            transform=transform,
        )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=1 if use_cache else args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=16,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=collate_fn_packed if use_cache else None,
        drop_last=not use_cache,
    )

    pipe = MyQwenImagePipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    denoiser = pipe.transformer
    denoiser.requires_grad_(False)

    num_layers = len(denoiser.transformer_blocks)

    residual_origin_layer = args.residual_origin_layer
    if residual_origin_layer is None:
        residual_origin_layer = 1

    residual_target_layers = args.residual_target_layers
    if residual_target_layers is None:
        residual_target_layers = list(range(residual_origin_layer + 1, num_layers - 1))

    if len(residual_target_layers) == 0:
        raise ValueError("residual_target_layers cannot be empty.")

    if any(layer <= residual_origin_layer for layer in residual_target_layers):
        raise ValueError(
            "residual_target_layers must be strictly greater than residual_origin_layer."
        )

    def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 20, x, torch.log(torch.expm1(x)))

    residual_init = torch.tensor(args.residual_init, device=device, dtype=torch.float32)

    if args.init_mode == "constant":
        residual_init = torch.clamp(residual_init, min=1e-6)
        residual_weights_raw = torch.nn.Parameter(
            softplus_inverse(residual_init).repeat(len(residual_target_layers))
        )
    elif args.init_mode == "linear":
        num_targets = len(residual_target_layers)
        if num_targets == 1:
            residual_init_values = residual_init.repeat(1)
        else:
            residual_init_values = torch.linspace(
                residual_init,
                0.0,
                steps=num_targets,
                device=device,
                dtype=torch.float32,
            )
        residual_init_values = torch.clamp(residual_init_values, min=1e-6)
        residual_weights_raw = torch.nn.Parameter(softplus_inverse(residual_init_values))
    else:
        raise ValueError(f"Not support init mode:{args.init_mode}")

    if args.residual_weights_ckpt is not None:
        data = torch.load(args.residual_weights_ckpt, map_location="cpu")
        if isinstance(data, dict) and "residual_weights" in data:
            loaded = data["residual_weights"]
        else:
            loaded = data
        loaded = torch.tensor(loaded, dtype=residual_weights_raw.dtype)
        if loaded.numel() != residual_weights_raw.numel():
            raise ValueError(
                "residual_weights_ckpt length mismatch: "
                f"expected {residual_weights_raw.numel()}, got {loaded.numel()}"
            )
        loaded = torch.clamp(loaded.to(device=device), min=1e-6)
        residual_weights_raw.data.copy_(softplus_inverse(loaded))
        print(f"[INFO] Loaded residual weights from {args.residual_weights_ckpt}")

    residual_rotation_matrices = None
    if args.residual_rotation_path is not None:
        rotations, saved_layers, meta = load_residual_procrustes(
            args.residual_rotation_path,
            device=device,
            dtype=torch.float32,
        )
        rotations, residual_target_layers = select_residual_rotations(
            rotations, saved_layers, residual_target_layers
        )
        residual_rotation_matrices = rotations
        if args.residual_origin_layer is None and isinstance(meta, dict):
            residual_origin_layer = meta.get("origin_layer", residual_origin_layer)

    denoiser.set_residual_config(
        residual_origin_layer,
        residual_target_layers,
        None,
        residual_stop_grad=args.residual_stop_grad,
        residual_rotation_matrices=residual_rotation_matrices,
    )

    optimizer = torch.optim.AdamW(
        [{"params": [residual_weights_raw], "lr": args.lr, "weight_decay": args.wd}]
    )
    scaler = GradScaler(enabled=(args.dtype == "float16" and device == "cuda"))

    if args.steps is None:
        args.steps = args.epochs * len(loader)
    num_training_steps = args.steps

    def lr_lambda(current_step: int):
        warmup_steps = int(args.warmup_steps)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    def save_residual_weights(step: int, suffix: str = ""):
        residual_weights = F.softplus(residual_weights_raw)
        payload = {
            "residual_weights": residual_weights.detach().cpu(),
            "origin_layer": residual_origin_layer,
            "target_layers": residual_target_layers,
        }
        filename = f"residual_weights_step{step}{suffix}.pth"
        torch.save(payload, osp.join(log_dir, filename))
        json_filename = f"residual_weights_step{step}{suffix}.json"
        json_payload = {
            "residual_weights": residual_weights.detach().cpu().tolist(),
            "origin_layer": residual_origin_layer,
            "target_layers": residual_target_layers,
        }
        with open(osp.join(log_dir, json_filename), "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)

    step = 0
    epoch = 0
    pbar = tqdm.tqdm(total=args.steps, desc="Training", disable=rank != 0)
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)
        step += 1

        if use_cache:
            prompt_embeds = _get_batch_value(
                batch, ("prompt_embeds", "prompt_emb", "txt_hidden_states")
            )
            prompt_mask = _get_batch_value(
                batch, ("prompt_embeds_mask", "prompt_mask", "txt_mask")
            )
            latents = _get_batch_value(batch, ("latents", "x0", "latent"))

            if prompt_embeds is None or latents is None:
                raise KeyError(
                    "Cached batch must include prompt_embeds/txt_hidden_states and latents/x0."
                )

            prompt_embeds = prompt_embeds.to(device)
            latents = latents.to(device)
            if prompt_mask is not None:
                prompt_mask = prompt_mask.to(device)
            prompt_mask = _ensure_prompt_mask(prompt_embeds, prompt_mask)
        else:
            images, prompts = batch
            images = images.to(device)
            prompts_list = list(prompts)

            with torch.no_grad():
                prompt_embeds, prompt_mask = pipe.encode_prompt(
                    prompt=prompts_list,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                )
                latents = _encode_images_to_latents(
                    pipe,
                    images,
                    height=args.img_size,
                    width=args.img_size,
                )

            prompt_mask = _ensure_prompt_mask(prompt_embeds, prompt_mask)

        if args.dtype == "float32":
            prompt_embeds = prompt_embeds.float()
            latents = latents.float()
        elif args.dtype == "bfloat16":
            prompt_embeds = prompt_embeds.bfloat16()
            latents = latents.bfloat16()

        num_steps = int(getattr(pipe.scheduler.config, "num_train_timesteps", 1000))
        t = sample_timesteps(
            latents.shape[0],
            num_steps,
            device,
            mode=args.time_mode,
            mu=args.time_shift,
            sigma=1.0,
        )

        optimizer.zero_grad(set_to_none=True)

        residual_weights = F.softplus(residual_weights_raw)
        denoiser.set_residual_weights(residual_weights)

        img_shape_value = args.img_size // pipe.vae_scale_factor // 2
        img_shapes = [[(1, img_shape_value, img_shape_value)]] * latents.shape[0]
        txt_seq_lens = (
            prompt_mask.sum(dim=1).tolist() if prompt_mask is not None else None
        )

        denoise_loss = compute_total_loss(
            denoiser=denoiser,
            scheduler=pipe.scheduler,
            latents=latents,
            t=t,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
        )

        if args.residual_smoothness_weight > 0 and residual_weights.numel() > 1:
            smoothness_loss = (residual_weights[1:] - residual_weights[:-1]).pow(2).mean()
        else:
            smoothness_loss = torch.zeros((), device=residual_weights.device)

        total_loss = denoise_loss + args.residual_smoothness_weight * smoothness_loss

        if scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
        else:
            total_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([residual_weights_raw], args.grad_clip)

        if distributed and residual_weights_raw.grad is not None:
            dist.all_reduce(residual_weights_raw.grad, op=dist.ReduceOp.SUM)
            residual_weights_raw.grad.div_(world_size)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        w_min = residual_weights.min().item()
        w_max = residual_weights.max().item()
        w_mean = residual_weights.mean().item()

        denoise_loss_value = denoise_loss.detach()
        smoothness_loss_value = smoothness_loss.detach()
        total_loss_value = total_loss.detach()
        if distributed:
            dist.all_reduce(denoise_loss_value, op=dist.ReduceOp.SUM)
            denoise_loss_value = denoise_loss_value / world_size
            dist.all_reduce(smoothness_loss_value, op=dist.ReduceOp.SUM)
            smoothness_loss_value = smoothness_loss_value / world_size
            dist.all_reduce(total_loss_value, op=dist.ReduceOp.SUM)
            total_loss_value = total_loss_value / world_size

        if rank == 0:
            pbar.set_description(
                f"Step {step} | "
                f"Total {total_loss_value.item():.4f} | "
                f"Denoise {denoise_loss_value.item():.4f} | "
                f"Smooth {smoothness_loss_value.item():.4f} | "
                f"w_mean {w_mean:.4f} | w_min {w_min:.4f} | w_max {w_max:.4f}"
            )
            pbar.update(1)

            writer.add_scalar("train/total_loss", total_loss_value.item(), step)
            writer.add_scalar("train/denoise_loss", denoise_loss_value.item(), step)
            writer.add_scalar("train/smoothness_loss", smoothness_loss_value.item(), step)
            writer.add_scalar("train/weights_mean", w_mean, step)
            writer.add_scalar("train/weights_min", w_min, step)
            writer.add_scalar("train/weights_max", w_max, step)
            writer.add_histogram(
                "train/residual_weights",
                residual_weights.detach().cpu(),
                step,
            )
        if args.save_interval > 0 and step % args.save_interval == 0:
            if rank == 0:
                print(
                    f"Residual weights at step-{step}: {residual_weights.detach().cpu()}"
                )
                save_residual_weights(step)

    pbar.close()

    if rank == 0:
        save_residual_weights(step, suffix="_final")
        writer.close()
        print("[INFO] Training complete.")
    if distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/path/to/Qwen-Image",
    )
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--datadir", type=str, default=None)
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--precompute_dir",
        type=str,
        nargs="+",
        default=None,
        help="缓存特征保存/读取目录（可传多个，用空格分隔）",
    )
    parser.add_argument("--ar_target_layer", type=int, default=10)

    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--time_mode", type=str, default="logitnorm")
    parser.add_argument("--time_shift", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="训练总步数（不指定则使用 epochs * len(dataloader)）",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--logdir", type=str, default="./logs")

    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_init", type=float, default=0.0)
    parser.add_argument("--init_mode", type=str, default="constant")
    parser.add_argument("--residual_weights_ckpt", type=str, default=None)
    parser.add_argument("--residual_stop_grad", type=int, default=1)
    parser.add_argument("--residual_rotation_path", type=str, default=None)
    parser.add_argument(
        "--residual_smoothness_weight",
        type=float,
        default=0.0,
        help="相邻 residual weights 的平滑正则化强度（L2）。",
    )

    args = parser.parse_args()
    args.residual_stop_grad = bool(args.residual_stop_grad)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train(args)


if __name__ == "__main__":
    main()
