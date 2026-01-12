import argparse
import os
import os.path as osp
import math
import yaml
import pandas as pd
import numpy as np
import tqdm
import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets import get_target_dataset
from util import set_seed, get_transform, build_prompts_from_captions, denormalize, sample_timesteps
from sampler import SD3Euler
# from eval_utils import PickScore, HPSv2
# import ImageReward as RM

# Qwen2-VL target wrapper
# from models import QwenVLTargetModel
# from transformer import Connector




# # =========================
# # 预计算并缓存特征
# # =========================
# @torch.no_grad()
# def precompute_and_save_features(args, train_set, precompute_dir, sampler_model, qwen_model, device):
#     os.makedirs(precompute_dir, exist_ok=True)
#     loader = DataLoader(train_set, batch_size=args.pc_batch_size, num_workers=4, pin_memory=True, shuffle=False)

#     pbar = tqdm.tqdm(loader, desc="[Precompute]")
#     idx_global = 0
#     for imgs, captions in pbar:
#         B = imgs.size(0)
#         start = time.time()
#         imgs = imgs.to(device)
#         if args.dtype == 'float16':
#             imgs = imgs.half()

#         # VAE -> x0
#         x0 = sampler_model.encode(imgs)

#         # prompt encoders（为 teacher 流准备）
#         prompt_emb, pooled_emb = sampler_model.encode_prompt(list(captions), batch_size=B)

#         # Qwen-VL -> 文本序列特征
#         # prompts_vl = build_prompts_from_captions(captions)
#         # tgt = qwen_model(denormalize(imgs), prompts_vl)
#         # txt_hidden_states = F.normalize(tgt["text_feats_unpooled"], dim=-1)

#         for j in range(B):
#             if(args.subset_number is not None):
#                 save_path = osp.join(precompute_dir, f"{args.subset_number:02d}{idx_global:08d}.pt")
#             else:
#                 save_path = osp.join(precompute_dir, f"{idx_global:08d}.pt")
#             torch.save({
#                 "x0": x0[j].half().cpu(),
#                 # "txt_hidden_states": txt_hidden_states[j].half().cpu(),   # (L, N, D_qwen)
#                 "caption": captions[j],
#                 "prompt_emb": prompt_emb[j].half().cpu(),
#                 "pooled_emb": pooled_emb[j].half().cpu(),
#                 "index": idx_global
#             }, save_path)
#             idx_global += 1



@torch.no_grad()
def precompute_and_save_features(args, train_set, precompute_dir, sampler_model, qwen_model, device):
    """
    预计算 SD3 + 文本编码器特征并缓存至磁盘。
    现在 encode_prompt 返回 (prompt_emb, pooled_emb, text_mask)，
    因此我们一并保存 token_mask，供训练阶段 compute_total_loss 使用。
    """
    import gc
    from torch.utils.data import DataLoader
    import tqdm
    import torch
    import os
    import os.path as osp

    os.makedirs(precompute_dir, exist_ok=True)
    loader = DataLoader(
        train_set,
        batch_size=args.pc_batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )

    N = getattr(args, "cache_group_size", 256)  # 每个文件的缓存样本数
    buffer = []
    file_idx = 0
    idx_global = 0

    pbar = tqdm.tqdm(loader, desc="[Precompute]")
    for imgs, captions in pbar:
        B = imgs.size(0)
        imgs = imgs.to(device)
        if args.dtype == 'float16':
            imgs = imgs.half()

        # =========================
        # Encode image -> latent
        # =========================
        x0 = sampler_model.encode(imgs)

        # =========================
        # Encode prompt -> embeddings + mask
        # =========================
        prompt_emb, pooled_emb, token_mask = sampler_model.encode_prompt(
            list(captions), batch_size=B
        )  # token_mask: [B, L] bool tensor

        # move to cpu + half precision for caching
        prompt_emb = prompt_emb.half().cpu()
        pooled_emb = pooled_emb.half().cpu()
        token_mask = token_mask.cpu()  # 不需要half（bool类型）

        # =========================
        # Optional: integrate Qwen-VL features (if needed)
        # =========================
        # prompts_vl = build_prompts_from_captions(captions)
        # tgt = qwen_model(denormalize(imgs), prompts_vl)
        # txt_hidden_states = F.normalize(tgt["text_feats_unpooled"], dim=-1)

        for j in range(B):
            buffer.append({
                "x0": x0[j].half().cpu(),
                "caption": captions[j],
                "prompt_emb": prompt_emb[j],
                "pooled_emb": pooled_emb[j],
                "token_mask": token_mask[j],       # ✅ 新增 mask
                # "txt_hidden_states": txt_hidden_states[j].half().cpu(),
                "index": idx_global
            })
            idx_global += 1

            # 每 N 条保存一次
            if len(buffer) >= N:
                save_path = osp.join(precompute_dir, f"{file_idx:04d}.pt")
                torch.save(buffer, save_path)
                print(f"[SAVE] {len(buffer)} samples -> {save_path}")
                file_idx += 1
                buffer = []
                gc.collect()

    # 保存最后一批
    if len(buffer) > 0:
        save_path = osp.join(precompute_dir, f"{file_idx:04d}.pt")
        torch.save(buffer, save_path)
        print(f"[SAVE] Final {len(buffer)} samples -> {save_path}")

    print(f"[INFO] Precompute finished. Total samples: {idx_global}, "
          f"total files: {file_idx + (1 if len(buffer)>0 else 0)}")

# =========================
# 训练（仅优化 Connector；损失 = diffusion loss + KL loss）
# =========================
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    print(f"[INFO] Use device={device}, GPUs={n_gpus}")
    set_seed(42)

    # logdir
    tag = "precompute" if args.do_precompute else "train"
    log_dir = osp.join(args.logdir, f"{args.model}_connector_{tag}")
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard
    tb_dir = osp.join(log_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    with open(osp.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # dataset
    print("Loading data...")
    transform = get_transform(args.img_size)
    train_set = get_target_dataset(args.dataset, args.datadir, train=True, transform=transform)
    val_set   = get_target_dataset(args.dataset, args.datadir, train=False, transform=transform)
    print(f"[INFO] Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    # SD3 基座（scheduler/vae/denoiser/text encoders）
    sampler_model = SD3Euler(
        model_key=args.model_key,
        device=device,
        use_8bit=args.use_8bit,
        load_ckpt_path=None,
    )
    denoiser = sampler_model.denoiser
    denoiser.eval()
    denoiser.requires_grad_(False)

    # ===== 预计算阶段 =====
    if args.do_precompute:
        assert args.precompute_dir is not None, "--precompute_dir 必须提供"
        qwen_model = None
        precompute_and_save_features(
            args=args,
            train_set=train_set,
            precompute_dir=args.precompute_dir,
            sampler_model=sampler_model,
            qwen_model=qwen_model,
            device=device
        )
        print("[INFO] Precompute done. Exit.")
        return
    else:
        return
    


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=512)

    # model
    parser.add_argument('--model', type=str, default='sd3')
    parser.add_argument('--model_key', type=str, default='/path/to/SD3')
    parser.add_argument('--use_8bit', action='store_true')

    # precompute/cache
    parser.add_argument('--do_precompute', action='store_true', help='只做预计算并退出')
    parser.add_argument('--precompute_dir', type=str, default=None, help='缓存特征保存/读取目录')
    parser.add_argument('--pc_batch_size', type=int, default=4, help='预计算阶段 batch size')

    # time
    parser.add_argument('--time_mode', type=str, default="uniform")
    parser.add_argument('--time_shift', type=float, default=0.0)

    # train
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--benchmark', type=str, default="CLIP")
    parser.add_argument('--num_eval', type=int, default=5)


 
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        print("[INFO] Enabled memory-efficient SDPA kernels (Flash/ME).")
    except Exception as e:
        print("[WARN] SDPA kernel switch not available:", e)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    train(args)


if __name__ == '__main__':
    main()
