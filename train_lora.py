# train_with_alignment_model_seq_text.py
import argparse
import os
import os.path as osp
import math
import yaml
import tqdm
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets import get_target_dataset, CachedFeatureDataset_Packed, collate_fn_packed
from util import set_seed, get_transform
from sampler import SD3Euler

from lora_utils import inject_lora, extract_lora_state_dict, LoRALinear, preview_targets, load_lora_state_dict

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
        save_image(img, save_path, normalize=True)
        results.append({"prompt": label, "img_path": save_path})

    # 如需接 ImageReward / CLIP 等 benchmark，可在此处补充
    benchmarks = {}
    return benchmarks


# =========================
# 总损失 = 纯 diffusion MSE（LoRA 微调）
# =========================
def compute_total_loss(
    denoiser,
    scheduler,
    x0,
    t,
    prompt_emb,
    pooled_emb,
    froze_model=False,
    residual_target_layers: Optional[List[int]] = None,
    residual_origin_layer: Optional[int] = None,
    residual_weights: Optional[List[float]] = None,   
    residual_use_layernorm: Optional[bool] = True,
):
    """
    仅计算 diffusion denoise MSE loss，用于 LoRA 微调。
    返回: total_loss, denoise_loss, txt_align_loss(恒为0，仅占位)
    """
    device = x0.device
    B = x0.shape[0]
    T = int(scheduler.config.num_train_timesteps)

    # 线性插值构造噪声状态
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
            residual_use_layernorm=residual_use_layernorm
        )
        v_pred = out["sample"]
        denoise_loss = F.mse_loss(v_pred, v_target)

    # 不再包含任何文本对齐，自监督 forcing 等
    total_loss = denoise_loss if not froze_model else 0.0
    return total_loss, denoise_loss

# =========================
# 训练主流程
# =========================
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    print(f"[INFO] Use device={device}, GPUs={n_gpus}")
    set_seed(42)

    # logdir
    tag = "precompute" if args.do_precompute else "train_cached"
    log_dir = osp.join(args.logdir, f"{args.model}_lora_dp_{tag}")
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard
    tb_dir = osp.join(log_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    with open(osp.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # 数据集（原始图像，用于 eval/日志）
    transform = get_transform(args.img_size)
    train_set = get_target_dataset(args.dataset, args.datadir, train=True, transform=transform)
    val_set = get_target_dataset(args.dataset, args.datadir, train=False, transform=transform)
    print(f"[INFO] Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    # ===== 训练阶段（从缓存加载特征） =====
    dataset = CachedFeatureDataset_Packed(
        cache_dirs=args.precompute_dir,
        target_batch_size=args.batch_size,
        cache_meta=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,  # 每次从若干 cache 文件中凑够 batch_size
        shuffle=True,
        num_workers=32,
        pin_memory=False,
        collate_fn=collate_fn_packed,
    )

    # scheduler + denoiser (SD3)
    sampler_model = SD3Euler(
        model_key=args.model_key,
        device=device,
        use_8bit=args.use_8bit,
        load_ckpt_path=None,
    )
    denoiser = sampler_model.denoiser
    denoiser.requires_grad_(False)

    # 注入 LoRA
    target = "all_linear" if args.lora_target == "all_linear" else tuple(args.lora_target.split(","))
    preview_targets(denoiser, ("to_q", "to_k", "to_v", "to_out.0"))
    inject_lora(
        denoiser,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target=target,
        dropout=args.lora_dropout,
    )
    denoiser = denoiser.to(device=device, dtype=torch.float32)

    # 如有已有 LoRA ckpt，加载
    if args.lora_ckpt is not None:
        print(f"[LoRA] injecting & loading LoRA from: {args.lora_ckpt}")
        lora_sd = torch.load(args.lora_ckpt, map_location="cpu")
        load_lora_state_dict(denoiser, lora_sd, strict=False)
        denoiser = denoiser.to(device=device, dtype=torch.float32)
        print("[LoRA] loaded and ready.")

    if n_gpus > 1 and device == "cuda":
        denoiser = nn.DataParallel(denoiser)

    # ===== 优化器参数组（仅 LoRA 参数） =====
    lora_params = []
    base_denoiser = denoiser.module if isinstance(denoiser, nn.DataParallel) else denoiser
    for _, m in base_denoiser.named_modules():
        if isinstance(m, LoRALinear):
            lora_params += list(m.parameters())

    optim_groups = []
    if not args.froze_model and len(lora_params) > 0:
        optim_groups.append(
            {
                "params": lora_params,
                "lr": args.lr,
                "weight_decay": args.wd,
            }
        )

    optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.999), eps=1e-8)
    scaler = GradScaler(enabled=(args.dtype == "float16" and device == "cuda"))

    # ===== 学习率调度（warmup + cosine） =====
    num_training_steps = args.epochs * len(loader)

    def lr_lambda(current_step: int):
        warmup_steps = int(args.warmup_steps)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ===== 辅助函数: grad norm =====
    def compute_and_clip_grad_norm(params, max_norm=None):
        if len(params) == 0:
            return 0.0
        if max_norm is not None and max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        else:
            total_norm = 0.0
            for p in params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
        return grad_norm

    # ===== 训练循环 =====
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            step += 1

            prompt_emb = batch["prompt_emb"].to(device)
            pooled_emb = batch["pooled_emb"].to(device)
            x0 = batch["x0"].to(device)

            if args.dtype == "float32":
                prompt_emb = prompt_emb.float()
                pooled_emb = pooled_emb.float()
                x0 = x0.float()

            # 采样噪声步
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

            total, denoise_loss = compute_total_loss(
                denoiser=denoiser,
                scheduler=sampler_model.scheduler,
                x0=x0,
                t=t,
                prompt_emb=prompt_emb,
                pooled_emb=pooled_emb,
                froze_model=args.froze_model,
                residual_target_layers=args.residual_target_layers,
                residual_origin_layer=args.residual_origin_layer,
                residual_weights=args.residual_weights,
                residual_use_layernorm=args.residual_use_layernorm
            )

            if scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)  # 先 unscale 再 clip
            else:
                total.backward()

            # grad norm + clip (仅 LoRA)
            grad_norm_lora = compute_and_clip_grad_norm(lora_params, args.grad_clip_lora)

            # optimizer step
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            # 日志
            pbar.set_description(
                f"Epoch {epoch} | Step {step} | "
                f"Total {total.item():.4f} | "
                f"Denoise {denoise_loss.item():.4f} | "
                f"GradNorm_LoRA {grad_norm_lora:.2f} | "
            )

            writer.add_scalar("train/Total_loss", total.item(), step)
            writer.add_scalar("train/Denoise_loss", denoise_loss.item(), step)
            writer.add_scalar("train/GradNorm_LoRA", grad_norm_lora, step)

            # eval + save LoRA
            if args.eval_interval > 0 and step % args.eval_interval == 0:
                eval_dir = osp.join(log_dir, f"val_{step}")
                os.makedirs(eval_dir, exist_ok=True)

                _wrap = type("Wrap", (), {"sampler": sampler_model})()
                sample_cfg = {
                    "NFE": 28,
                    "img_shape": (1024, 1024),
                    "cfg_scale": 4,
                }
                # ====== 自动注入 residual 参数 ======
                if args.residual_origin_layer is not None:
                    sample_cfg["residual_target_layers"] = args.residual_target_layers
                    sample_cfg["residual_origin_layer"] = args.residual_origin_layer
                    sample_cfg["residual_weights"] = args.residual_weights
                    sample_cfg["residual_use_layernorm"] = args.residual_use_layernorm
                benchmarks = eval_model(
                    args,
                    model=_wrap,
                    target_dataset=val_set,
                    eval_run_folder=eval_dir,
                    sample_cfg=sample_cfg,
                )

                # 如需写入 benchmark scalar 到 TB，可在此处使用 benchmarks

                # 保存当前 LoRA 权重
                base = denoiser.module if isinstance(denoiser, nn.DataParallel) else denoiser
                lora_sd = extract_lora_state_dict(base)
                lora_path = osp.join(log_dir, f"lora_step{step}.pth")
                torch.save(lora_sd, lora_path)

    writer.close()
    print("[INFO] Training complete.")


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=512)

    # model
    parser.add_argument("--model", type=str, default="sd3")
    parser.add_argument("--model_key", type=str, default="/path/to/SD3")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # precompute/cache
    parser.add_argument(
        "--do_precompute",
        action="store_true",
        help="只做预计算并退出（本脚本里仅用于命名 tag，可与预计算脚本配合使用）",
    )
    parser.add_argument(
        "--precompute_dir",
        type=str,
        nargs="+",  # 可传多个 cache 目录
        default=None,
        help="缓存特征保存/读取目录（可传多个，用空格分隔）",
    )

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target",
        type=str,
        default="to_q,to_k,to_v,to_out.0",
        help="all_linear 或模块名片段，如: to_q,to_k,to_v,to_out.0",
    )
    parser.add_argument("--lora_ckpt", type=str, default=None)

    # time
    parser.add_argument("--time_mode", type=str, default="logitnorm")
    parser.add_argument("--time_shift", type=float, default=0.0)

    # train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)  # LoRA 学习率
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip_lora", type=float, default=0.1)

    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ImageReward-v1.0,CLIP",
        help="当前未被使用，仅保留接口占位",
    )
    parser.add_argument("--num_eval", type=int, default=50)

    # misc
    parser.add_argument("--froze_model", action="store_true")


    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_use_layernorm", type=int, default=1)
    
    
    args = parser.parse_args()
    args.residual_use_layernorm = bool(args.residual_use_layernorm)


    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train(args)


if __name__ == "__main__":
    main()
