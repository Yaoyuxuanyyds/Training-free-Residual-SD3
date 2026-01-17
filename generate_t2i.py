import argparse
import os
import glob
import re
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.transforms import ToTensor  # 其实现在没用到，但留着也无妨
from sampler import SD3Euler, build_timestep_residual_weight_fn
from util import load_residual_procrustes, select_residual_rotations, set_seed, load_residual_weights
from lora_utils import *

torch.set_grad_enabled(False)


# ================================
# 工具类：兼容普通采样和 residual 采样
# ================================
class SD3ImageGenerator:
    def __init__(
        self,
        model_key,
        load_ckpt_path=None,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_rotation_matrices=None,
        residual_timestep_weight_fn=None,
    ):
        """
        封装 sampler，支持普通 sample 和 sample_residual
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化 sampler
        self.sampler = SD3Euler(
            model_key,
            use_8bit=False,
            load_ckpt_path=load_ckpt_path
        )

        # residual 默认参数
        self.residual_target_layers = residual_target_layers
        self.residual_origin_layer = residual_origin_layer
        self.residual_weights = residual_weights
        self.residual_rotation_matrices = residual_rotation_matrices
        self.residual_timestep_weight_fn = residual_timestep_weight_fn

    def generate(
        self,
        prompt,
        seed,
        img_size,
        steps,
        scale,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_rotation_matrices=None,
        residual_timestep_weight_fn=None,
    ):
        """
        如果 residual_origin_layer is None → 普通 sample
        否则 → sample_residual
        """

        # 优先级：函数参数 > 初始化参数
        rt = residual_target_layers if residual_target_layers is not None else self.residual_target_layers
        ro = residual_origin_layer if residual_origin_layer is not None else self.residual_origin_layer
        rw = residual_weights if residual_weights is not None else self.residual_weights
        rr = residual_rotation_matrices if residual_rotation_matrices is not None else self.residual_rotation_matrices
        rtw = (
            residual_timestep_weight_fn
            if residual_timestep_weight_fn is not None
            else self.residual_timestep_weight_fn
        )

        set_seed(seed)
        prompts = [prompt]

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):

            # === 普通 ===
            if ro is None:
                img = self.sampler.sample(
                    prompts,
                    NFE=steps,
                    img_shape=(img_size, img_size),
                    cfg_scale=scale,
                    batch_size=1,
                )
            # === residual ===
            else:
                img = self.sampler.sample_residual(
                    prompts,
                    NFE=steps,
                    img_shape=(img_size, img_size),
                    cfg_scale=scale,
                    batch_size=1,
                    residual_target_layers=rt,
                    residual_origin_layer=ro,
                    residual_weights=rw,
                    residual_rotation_matrices=rr,
                    residual_timestep_weight_fn=rtw,
                )
        return img


def clean_prompt_for_filename(prompt):
    illegal_chars = r'[\/:*?"<>|]'
    cleaned = re.sub(illegal_chars, '_', prompt.strip())
    if len(cleaned) > 3000:
        cleaned = cleaned[:3000] + "_truncated"
    return cleaned


# ==========================================
# 参数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench/examples/dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3",
    )

    parser.add_argument(
        "--outdir_base",
        type=str,
        default="/inspire/.../examples",
    )

    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=7.0)

    # 注意：默认给 10 个 seed，对应 n_samples=10
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627],
        metavar="SEED",
    )

    parser.add_argument("--output_prefix", type=str, default="sd3_multigpu")

    # residual 参数
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument(
        "--timestep_residual_weight_fn",
        type=str,
        default="constant",
        help="Mapping from timestep (0-1000) to residual weight multiplier.",
    )
    parser.add_argument(
        "--timestep_residual_weight_power",
        type=float,
        default=1.0,
        help="Optional power for timestep residual weight mapping.",
    )
    parser.add_argument(
        "--timestep_residual_weight_exp_alpha",
        type=float,
        default=1.5,
        help="Exponent alpha for exponential timestep residual weight mapping.",
    )


    # ---------- LoRA 采样支持 ---------- #
    parser.add_argument('--lora_ckpt', type=str, default=None, help='Path to LoRA-only checkpoint (.pth)')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target', type=str, default='all_linear',
                        help="all_linear 或模块名片段，如: to_q,to_k,to_v,to_out")
    parser.add_argument('--lora_dropout', type=float, default=0.0)



    # 多卡分片参数（仿照 DPG 脚本）
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of workers for sharded txt files (e.g., 8 GPUs).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="This worker's rank in [0, world_size-1].",
    )

    opt = parser.parse_args()

    # 保证 seeds 数量 ≥ n_samples（你也可以要求 ==）
    assert len(opt.seeds) >= opt.n_samples, \
        f"Need at least n_samples ({opt.n_samples}) seeds, but got {len(opt.seeds)}."

    return opt


def tensor_to_pil(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img_tensor = img_tensor.detach().cpu().clamp(0, 1)
    img = (img_tensor * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


# ==========================================
# 主流程（手动多进程 + world_size/rank）
# ==========================================
def main(opt):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Rank {opt.rank}] Using device: {device} | world_size={opt.world_size}")

    # ========= 加载一次生成器（每个进程 / 每张卡各自一份） =========
    residual_rotation_matrices = None
    if opt.residual_procrustes_path is not None:
        residual_rotation_matrices, target_layers, meta = load_residual_procrustes(
            opt.residual_procrustes_path
        )
        residual_rotation_matrices, opt.residual_target_layers = select_residual_rotations(
            residual_rotation_matrices, target_layers, opt.residual_target_layers
        )
        if opt.residual_origin_layer is None and isinstance(meta, dict):
            opt.residual_origin_layer = meta.get("origin_layer")
        if isinstance(meta, dict) and ("mu_src" in meta or "mu_tgt" in meta):
            mu_src = meta.get("mu_src")
            mu_tgt = meta.get("mu_tgt")
            if mu_src is not None and not torch.is_tensor(mu_src):
                mu_src = torch.tensor(mu_src)
            if mu_tgt is not None and not torch.is_tensor(mu_tgt):
                mu_tgt = torch.tensor(mu_tgt)
            if mu_tgt is not None and target_layers is not None:
                saved_layers_list = list(target_layers)
                selected_layers = (
                    opt.residual_target_layers
                    if opt.residual_target_layers is not None
                    else saved_layers_list
                )
                indices = [saved_layers_list.index(layer) for layer in selected_layers]
                mu_tgt = mu_tgt[indices]
            residual_rotation_matrices = {
                "rotation_matrices": residual_rotation_matrices,
                "mu_src": mu_src,
                "mu_tgt": mu_tgt,
            }

    if opt.residual_weights is None and opt.residual_weights_path is not None:
        opt.residual_weights = load_residual_weights(opt.residual_weights_path)

    generator = SD3ImageGenerator(
        model_key=opt.model,
        load_ckpt_path=None,
        residual_target_layers=opt.residual_target_layers,
        residual_origin_layer=opt.residual_origin_layer,
        residual_weights=opt.residual_weights,
        residual_rotation_matrices=residual_rotation_matrices,
        residual_timestep_weight_fn=build_timestep_residual_weight_fn(
            opt.timestep_residual_weight_fn,
            power=opt.timestep_residual_weight_power,
            exp_alpha=opt.timestep_residual_weight_exp_alpha,
        ),
    )


    # ---------- 如果提供了 LoRA ckpt，注入 + 加载 ----------
    if opt.lora_ckpt is not None:
        print(f"[LoRA] injecting & loading LoRA from: {opt.lora_ckpt}")
        target = "all_linear" if opt.lora_target == "all_linear" else tuple(opt.lora_target.split(","))
        # 对 sampler.denoiser（SD3Transformer2DModel_Vanilla）里的 transformer 注入
        inject_lora(generator.sampler.denoiser, rank=opt.lora_rank, alpha=opt.lora_alpha,
                    target=target, dropout=opt.lora_dropout)
        generator.sampler.denoiser.to(device=device, dtype=torch.float32)   # 就地转换
        lora_sd = torch.load(opt.lora_ckpt, map_location="cpu")
        load_lora_state_dict(generator.sampler.denoiser, lora_sd, strict=True)
        
        generator.sampler.denoiser.eval()
        print("[LoRA] loaded and ready.")
        




    # ========= 收集 txt 文件 =========
    txt_files = sorted(glob.glob(os.path.join(opt.dataset_dir, "*val.txt")))
    total_files = len(txt_files)

    # ========= 多 GPU 分片：按下标对 world_size 取模 =========
    if opt.world_size > 1:
        txt_files = [f for i, f in enumerate(txt_files) if i % opt.world_size == opt.rank]

    print(f"[Rank {opt.rank}] Total txt files: {total_files}, this rank will handle: {len(txt_files)}")

    # ========= 处理每个 txt 文件 =========
    for txt_path in txt_files:
        txt_name = os.path.splitext(os.path.basename(txt_path))[0]
        outdir = os.path.join(opt.outdir_base, f"samples_{opt.output_prefix}_{txt_name}")
        os.makedirs(outdir, exist_ok=True)

        with open(txt_path, "r") as f:
            prompts = [x.strip() for x in f if x.strip()]

        # ========= 每个 prompt × n_samples =========
        # 实际用的是前 n_samples 个 seeds
        seeds_to_use = opt.seeds[:opt.n_samples]

        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"[Rank {opt.rank}] {txt_name}")):
            prompt_clean = clean_prompt_for_filename(prompt)

            for si, seed in enumerate(seeds_to_use):
                fname = f"{prompt_clean}_{si:06d}.png"
                fpath = os.path.join(outdir, fname)

                if os.path.exists(fpath):
                    continue

                img = generator.generate(
                    prompt=prompt,
                    seed=seed,
                    img_size=opt.H,
                    steps=opt.steps,
                    scale=opt.scale,
                    residual_timestep_weight_fn=generator.residual_timestep_weight_fn,
                )

                # 如果你更喜欢 PIL：
                # tensor_to_pil(img).save(fpath)
                save_image(img, fpath, normalize=True)

    print(f"[Rank {opt.rank}] All done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
