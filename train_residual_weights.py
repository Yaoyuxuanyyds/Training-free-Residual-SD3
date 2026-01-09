import argparse
import math
import os
import os.path as osp
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import yaml

from dataset.datasets import CachedFeatureDataset_Packed, collate_fn_packed, get_target_dataset
from sampler import SD3Euler
from util import get_transform, load_residual_procrustes, select_residual_rotations, set_seed

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

torch.multiprocessing.set_sharing_strategy("file_system")


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
# 评估
# =========================

@torch.no_grad()
def eval_model(args, model, target_dataset, eval_run_folder, sample_cfg):
    pbar = tqdm.tqdm(range(args.num_eval))
    results = []
    set_seed(args.seed)

    for i in pbar:
        _, label = target_dataset[i]
        img = model.sampler.sample_residual([label], **sample_cfg)
        save_path = osp.join(eval_run_folder, f"{i:04d}.png")
        from torchvision.utils import save_image

        save_image(img, save_path, normalize=True)
        results.append({"prompt": label, "img_path": save_path})

    benchmarks = {}
    return benchmarks


# =========================
# 总损失 = diffusion MSE（仅 residual weights）
# =========================

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
):
    device = x0.device
    B = x0.shape[0]
    T = int(scheduler.config.num_train_timesteps)

    s = (t.float() / float(T)).view(B, 1, 1, 1)
    x1 = torch.randn_like(x0)
    x_s = (1 - s) * x0 + s * x1
    v_target = x1 - x0

    with autocast(enabled=True):
        out = denoiser(
            x_s,
            timestep=t,
            encoder_hidden_states=prompt_emb,
            pooled_projections=pooled_emb,
            return_dict=False,
            residual_target_layers=residual_target_layers,
            residual_origin_layer=residual_origin_layer,
            residual_weights=residual_weights,
            residual_use_layernorm=residual_use_layernorm,
            residual_rotation_matrices=residual_rotation_matrices,
        )
        v_pred = out["sample"]
        denoise_loss = F.mse_loss(v_pred, v_target)

    return denoise_loss


# =========================
# 训练主流程
# =========================

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    print(f"[INFO] Use device={device}, GPUs={n_gpus}")
    set_seed(args.seed)

    log_dir = osp.join(args.logdir, f"{args.model}_residual_weights")
    os.makedirs(log_dir, exist_ok=True)

    tb_dir = osp.join(log_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

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

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=32,
        pin_memory=False,
        collate_fn=collate_fn_packed,
    )

    sampler_model = SD3Euler(
        model_key=args.model_key,
        device=device,
        use_8bit=args.use_8bit,
        load_ckpt_path=None,
    )
    denoiser = sampler_model.denoiser
    denoiser.requires_grad_(False)
    denoiser = denoiser.to(device=device, dtype=torch.float32)

    base_model = denoiser.base_model
    num_layers = len(base_model.transformer_blocks)

    residual_origin_layer = args.residual_origin_layer
    if residual_origin_layer is None:
        residual_origin_layer = 1

    residual_target_layers = args.residual_target_layers
    if residual_target_layers is None:
        residual_target_layers = list(range(residual_origin_layer + 1, num_layers-1))

    if len(residual_target_layers) == 0:
        raise ValueError("residual_target_layers cannot be empty.")

    if any(layer <= residual_origin_layer for layer in residual_target_layers):
        raise ValueError(
            "residual_target_layers must be strictly greater than residual_origin_layer."
        )

    residual_weights = torch.nn.Parameter(
        torch.full(
            (len(residual_target_layers),),
            args.residual_init,
            device=device,
            dtype=torch.float32,
        )
    )

    if args.residual_weights_ckpt is not None:
        data = torch.load(args.residual_weights_ckpt, map_location="cpu")
        if isinstance(data, dict) and "residual_weights" in data:
            loaded = data["residual_weights"]
        else:
            loaded = data
        loaded = torch.tensor(loaded, dtype=residual_weights.dtype)
        if loaded.numel() != residual_weights.numel():
            raise ValueError(
                "residual_weights_ckpt length mismatch: "
                f"expected {residual_weights.numel()}, got {loaded.numel()}"
            )
        residual_weights.data.copy_(loaded.to(device=device))
        print(f"[INFO] Loaded residual weights from {args.residual_weights_ckpt}")

    residual_rotation_matrices = None
    if args.residual_rotation_path is not None:
        rotations, saved_layers, _ = load_residual_procrustes(
            args.residual_rotation_path,
            device=device,
            dtype=torch.float32,
        )
        rotations, residual_target_layers = select_residual_rotations(
            rotations, saved_layers, residual_target_layers
        )
        residual_rotation_matrices = rotations

    optimizer = torch.optim.AdamW(
        [{"params": [residual_weights], "lr": args.lr, "weight_decay": args.wd}]
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
        payload = {
            "residual_weights": residual_weights.detach().cpu(),
            "origin_layer": residual_origin_layer,
            "target_layers": residual_target_layers,
        }
        filename = f"residual_weights_step{step}{suffix}.pth"
        torch.save(payload, osp.join(log_dir, filename))

    step = 0
    pbar = tqdm.tqdm(total=args.steps, desc="Training")
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        step += 1

        prompt_emb = batch["prompt_emb"].to(device)
        pooled_emb = batch["pooled_emb"].to(device)
        x0 = batch["x0"].to(device)

        if args.dtype == "float32":
            prompt_emb = prompt_emb.float()
            pooled_emb = pooled_emb.float()
            x0 = x0.float()

        num_steps = sampler_model.scheduler.config.num_train_timesteps
        t = sample_timesteps(
            x0.shape[0],
            num_steps,
            device,
            mode=args.time_mode,
            mu=args.time_shift,
            sigma=1.0,
        )

        optimizer.zero_grad(set_to_none=True)

        denoise_loss = compute_total_loss(
            denoiser=denoiser,
            scheduler=sampler_model.scheduler,
            x0=x0,
            t=t,
            prompt_emb=prompt_emb,
            pooled_emb=pooled_emb,
            residual_target_layers=residual_target_layers,
            residual_origin_layer=residual_origin_layer,
            residual_weights=residual_weights,
            residual_use_layernorm=args.residual_use_layernorm,
            residual_rotation_matrices=residual_rotation_matrices,
        )

        if scaler.is_enabled():
            scaler.scale(denoise_loss).backward()
            scaler.unscale_(optimizer)
        else:
            denoise_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([residual_weights], args.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        w_min = residual_weights.min().item()
        w_max = residual_weights.max().item()
        w_mean = residual_weights.mean().item()

        pbar.set_description(
            f"Step {step} | "
            f"Denoise {denoise_loss.item():.4f} | "
            f"w_mean {w_mean:.4f} | w_min {w_min:.4f} | w_max {w_max:.4f}"
        )
        pbar.update(1)

        writer.add_scalar("train/denoise_loss", denoise_loss.item(), step)
        writer.add_scalar("train/weights_mean", w_mean, step)
        writer.add_scalar("train/weights_min", w_min, step)
        writer.add_scalar("train/weights_max", w_max, step)
        writer.add_histogram(
            "train/residual_weights",
            residual_weights.detach().cpu(),
            step
        )
        if args.save_interval > 0 and step % args.save_interval == 0:
            print(f"Residual weights at step-{step}: {residual_weights.detach().cpu()}")
            save_residual_weights(step)

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            eval_dir = osp.join(log_dir, f"val_{step}")
            os.makedirs(eval_dir, exist_ok=True)

            _wrap = type("Wrap", (), {"sampler": sampler_model})()
            sample_cfg = {
                "NFE": 28,
                "img_shape": (1024, 1024),
                "cfg_scale": 7,
                "residual_target_layers": residual_target_layers,
                "residual_origin_layer": residual_origin_layer,
                "residual_weights": residual_weights.detach(),
                "residual_use_layernorm": args.residual_use_layernorm,
                "residual_rotation_matrices": residual_rotation_matrices,
            }
            eval_model(
                args,
                model=_wrap,
                target_dataset=val_set,
                eval_run_folder=eval_dir,
                sample_cfg=sample_cfg,
            )
    pbar.close()

    save_residual_weights(step, suffix="_final")
    writer.close()
    print("[INFO] Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=512)

    parser.add_argument("--model", type=str, default="sd3")
    parser.add_argument("--model_key", type=str, default="/path/to/SD3")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--precompute_dir",
        type=str,
        nargs="+",
        default=None,
        help="缓存特征保存/读取目录（可传多个，用空格分隔）",
    )

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
        "--dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--num_eval", type=int, default=50)

    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_init", type=float, default=0.0)
    parser.add_argument("--residual_weights_ckpt", type=str, default=None)
    parser.add_argument("--residual_use_layernorm", type=int, default=1)
    parser.add_argument("--residual_rotation_path", type=str, default=None)

    args = parser.parse_args()
    args.residual_use_layernorm = bool(args.residual_use_layernorm)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train(args)


if __name__ == "__main__":
    main()
