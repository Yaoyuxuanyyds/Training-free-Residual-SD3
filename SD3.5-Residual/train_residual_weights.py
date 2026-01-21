import argparse
import json
import math
import os
import os.path as osp
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import yaml

from dataset.datasets import CachedFeatureDataset_Packed, collate_fn_packed, get_target_dataset
from generate_image_res import SD35PipelineWithRES
from sd35_transformer_res import SD35Transformer2DModel_RES
from util import (
    get_transform,
    load_residual_procrustes,
    select_residual_rotations,
    set_seed,
    resolve_rotation_bucket,
)

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

torch.multiprocessing.set_sharing_strategy("file_system")

DEFAULT_SD35_MODEL = "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5-large"


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


@torch.no_grad()
def eval_model(args, model, target_dataset, eval_run_folder, sample_cfg):
    pbar = tqdm.tqdm(range(args.num_eval))
    results = []
    set_seed(args.seed)

    for i in pbar:
        _, label = target_dataset[i]
        img = model(
            prompt=label,
            height=sample_cfg["img_size"],
            width=sample_cfg["img_size"],
            num_inference_steps=sample_cfg["NFE"],
            guidance_scale=sample_cfg["cfg_scale"],
            output_type="pt",
            residual_target_layers=sample_cfg["residual_target_layers"],
            residual_origin_layer=sample_cfg["residual_origin_layer"],
            residual_weights=sample_cfg["residual_weights"],
            residual_rotation_matrices=sample_cfg["residual_rotation_matrices"],
            residual_rotation_meta=sample_cfg["residual_rotation_meta"],
        ).images[0]
        save_path = osp.join(eval_run_folder, f"{i:04d}.png")
        from torchvision.utils import save_image

        save_image(img, save_path, normalize=True)
        results.append({"prompt": label, "img_path": save_path})

    benchmarks = {}
    return benchmarks


def compute_total_loss(
    denoiser,
    scheduler,
    x0,
    t,
    prompt_emb,
    pooled_emb,
    residual_target_layers: Optional[List[int]] = None,
    residual_origin_layer: Optional[int] = None,
    residual_weights: Optional[torch.Tensor] = None,
    residual_use_layernorm: bool = True,
    residual_rotation_matrices: Optional[torch.Tensor] = None,
    residual_rotation_meta: Optional[dict] = None,
    use_autocast: bool = True,
    use_checkpoint: bool = False,
):
    device = x0.device
    B = x0.shape[0]
    T = int(scheduler.config.num_train_timesteps)

    s = (t.float() / float(T)).view(B, 1, 1, 1)
    x1 = torch.randn_like(x0)
    x_s = (1 - s) * x0 + s * x1
    v_target = x1 - x0

    selected_rotations = resolve_rotation_bucket(
        residual_rotation_matrices,
        residual_rotation_meta,
        t,
    )

    def denoiser_forward(x_s, t, prompt_emb, pooled_emb, residual_weights):
        return denoiser(
            x_s,
            timestep=t,
            encoder_hidden_states=prompt_emb,
            pooled_projections=pooled_emb,
            return_dict=False,
            residual_target_layers=residual_target_layers,
            residual_origin_layer=residual_origin_layer,
            residual_weights=residual_weights,
            residual_use_layernorm=residual_use_layernorm,
            residual_rotation_matrices=selected_rotations,
        )

    with autocast(enabled=use_autocast):
        if use_checkpoint:
            out = checkpoint(
                denoiser_forward,
                x_s,
                t,
                prompt_emb,
                pooled_emb,
                residual_weights,
                use_reentrant=False,
            )
        else:
            out = denoiser_forward(x_s, t, prompt_emb, pooled_emb, residual_weights)
        v_pred = out["sample"] if isinstance(out, dict) else out[0]
        denoise_loss = F.mse_loss(v_pred, v_target)

    return denoise_loss


def _offload_text_encoders(pipe: SD35PipelineWithRES) -> None:
    for name in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
        module = getattr(pipe, name, None)
        if module is not None:
            module.to("cpu")


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
    log_dir = osp.join(args.logdir, f"{args.model}_residual_weights_lr-{args.lr}_bs-{global_bs}_init-{args.init_mode}{args.residual_init}_steps-{args.steps}_smooth-{args.residual_smoothness_weight}")
    os.makedirs(log_dir, exist_ok=True)

    tb_dir = osp.join(log_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir) if rank == 0 else None

    if rank == 0:
        with open(osp.join(log_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    transform = get_transform(args.img_size)
    train_set = get_target_dataset(args.dataset, args.datadir, train=True, transform=transform)
    val_set = get_target_dataset(args.dataset, args.datadir, train=False, transform=transform)
    print(f"[INFO] Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    dataset = CachedFeatureDataset_Packed(
        cache_dirs=args.precompute_dir,
        target_batch_size=args.batch_size,
        cache_meta=True,
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
        batch_size=1,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=16,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=collate_fn_packed,
    )

    pipe = SD35PipelineWithRES.from_pretrained(
        args.model_key or DEFAULT_SD35_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    pipe.transformer = SD35Transformer2DModel_RES(pipe.transformer)
    if args.offload_text_encoders:
        _offload_text_encoders(pipe)

    denoiser = pipe.transformer
    denoiser.requires_grad_(False)
    denoiser = denoiser.to(device=device, dtype=torch.float32)

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
    residual_rotation_meta = None
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
        residual_rotation_meta = meta

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

        target_dtype = torch.float16 if args.dtype == "float16" else torch.float32
        prompt_emb = batch["prompt_emb"].to(
            device=device, dtype=target_dtype, non_blocking=True
        )
        pooled_emb = batch["pooled_emb"].to(
            device=device, dtype=target_dtype, non_blocking=True
        )
        x0 = batch["x0"].to(device=device, dtype=target_dtype, non_blocking=True)

        num_steps = pipe.scheduler.config.num_train_timesteps
        t = sample_timesteps(
            x0.shape[0],
            num_steps,
            device,
            mode=args.time_mode,
            mu=args.time_shift,
            sigma=1.0,
        )

        optimizer.zero_grad(set_to_none=True)

        residual_weights = F.softplus(residual_weights_raw)

        denoise_loss = compute_total_loss(
            denoiser=denoiser,
            scheduler=pipe.scheduler,
            x0=x0,
            t=t,
            prompt_emb=prompt_emb,
            pooled_emb=pooled_emb,
            residual_target_layers=residual_target_layers,
            residual_origin_layer=residual_origin_layer,
            residual_weights=residual_weights,
            residual_use_layernorm=args.residual_use_layernorm,
            residual_rotation_matrices=residual_rotation_matrices,
            residual_rotation_meta=residual_rotation_meta,
            use_autocast=scaler.is_enabled(),
            use_checkpoint=args.activation_checkpoint,
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

        if rank == 0:
            pbar.set_postfix(
                {
                    "loss": total_loss.item(),
                    "w_min": w_min,
                    "w_max": w_max,
                    "w_mean": w_mean,
                }
            )
            pbar.update(1)

            if writer is not None:
                writer.add_scalar("train/denoise_loss", denoise_loss.item(), step)
                writer.add_scalar("train/smoothness_loss", smoothness_loss.item(), step)
                writer.add_scalar("train/total_loss", total_loss.item(), step)
                writer.add_scalar("train/w_min", w_min, step)
                writer.add_scalar("train/w_max", w_max, step)
                writer.add_scalar("train/w_mean", w_mean, step)
                writer.add_histogram(
                    "train/residual_weights",
                    residual_weights.detach().cpu(),
                    step,
                )

            if step % args.save_interval == 0:
                save_residual_weights(step)

    if rank == 0:
        save_residual_weights(step, suffix="_final")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=512)

    parser.add_argument("--model", type=str, default="sd3.5")
    parser.add_argument("--model_key", type=str, default=DEFAULT_SD35_MODEL)
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--precompute_dir",
        type=str,
        nargs="+",
        required=True,
        help="Directory for cached prompt_emb / pooled_emb / x0",
    )
    parser.add_argument("--time_mode", type=str, default="logitnorm")
    parser.add_argument("--time_shift", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--activation_checkpoint", action="store_true")

    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--residual_smoothness_weight", type=float, default=0.0)

    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--num_eval", type=int, default=20)

    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_init", type=float, default=0.0)
    parser.add_argument("--init_mode", type=str, default="constant")
    parser.add_argument("--residual_weights_ckpt", type=str, default=None)
    parser.add_argument("--residual_use_layernorm", type=int, default=1)
    parser.add_argument("--residual_rotation_path", type=str, default=None)

    parser.add_argument(
        "--offload_text_encoders",
        action="store_true",
        help="Move text encoders to CPU to save GPU memory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
