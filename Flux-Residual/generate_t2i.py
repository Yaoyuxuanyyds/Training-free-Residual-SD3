import argparse
import os
import glob
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
import random

# -------------------------- 直接导入Flux残差相关模块（无需重写）--------------------------
from generate_image_res import FluxPipelineWithRES
from flux_transformer_res import FluxTransformer2DModel_RES
from util import load_residual_procrustes, select_residual_rotations, load_residual_weights

def set_seed(seed):
    """固定随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# ===============================================================
# Flux残差生成器封装（替换原QwenImageGenerator）
# ===============================================================

class FluxResGenerator:
    def __init__(
        self,
        model_dir,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_use_layernorm: bool = True,
        residual_rotation_matrices=None,
        residual_rotation_meta=None,
        cfg=3.5,  # Flux默认引导尺度（原Qwen的4.0调整为Flux推荐值）
        steps=50,
        width=1024,
        height=1024,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.device == "cuda":
            raise RuntimeError("Flux生成需GPU支持（建议16GB以上显存）")

        # 加载Flux残差Pipeline（复用已有逻辑）
        print(f"[INFO] 加载Flux模型: {model_dir}")
        self.pipe = FluxPipelineWithRES.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # 替换为残差Transformer（核心步骤）
        self.pipe.transformer = FluxTransformer2DModel_RES(self.pipe.transformer)
        if not isinstance(self.pipe.transformer, FluxTransformer2DModel_RES):
            raise RuntimeError("残差Transformer替换失败！请检查flux_transformer_res.py")

        # GPU配置与推理优化
        self.pipe.to(self.device)
        core_modules = [self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.transformer, self.pipe.vae]
        for module in core_modules:
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        # 保存生成配置（与Flux参数对齐）
        self.cfg = cfg
        self.steps = steps
        self.width = width
        self.height = height
        self.residual_config = {
            "residual_target_layers": residual_target_layers,
            "residual_origin_layer": residual_origin_layer,
            "residual_weights": residual_weights,
            "residual_use_layernorm": residual_use_layernorm,
            "residual_rotation_matrices": residual_rotation_matrices,
            "residual_rotation_meta": residual_rotation_meta,
        }

    def generate(self, prompt, seed):
        """生成单张图像，返回[-1,1]张量（与原Qwen生成器输出格式一致）"""
        set_seed(seed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                height=self.height,
                width=self.width,
                guidance_scale=self.cfg,
                num_inference_steps=self.steps,
                max_sequence_length=512,  # Flux最大文本长度
                generator=torch.Generator(self.device).manual_seed(seed),
                **self.residual_config  # 注入残差参数
            )

        # 转换为原代码要求的输出格式（[3,H,W]张量，值范围[-1,1]）
        img = result.images[0]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor * 2.0) - 1.0
        return img_tensor

# ===============================================================
# 工具函数（保留原Qwen代码逻辑，确保输入输出兼容）
# ===============================================================
def clean_prompt_for_filename(prompt):
    illegal_chars = r'[\/:*?"<>|]'
    cleaned = re.sub(illegal_chars, "_", prompt.strip())
    if len(cleaned) > 3000:
        cleaned = cleaned[:3000] + "_truncated"
    return cleaned

# ===============================================================
# 参数解析（保留原Qwen代码的输入格式，仅调整残差参数名与Flux对齐）
# ===============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # 原Qwen代码的路径参数（保持不变，确保兼容性）
    parser.add_argument("--dataset_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3-1/T2I-CompBench/examples/dataset")
    parser.add_argument("--model_dir", type=str, default="/inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev")
    parser.add_argument("--outdir_base", type=str, required=True)

    # 生成参数（调整默认值为Flux推荐值）
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.5)  # Flux推荐3.5
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627],
    )
    parser.add_argument("--output_prefix", type=str, default="flux_res_t2i")  # 前缀改为Flux残差标识

    # 残差参数（与SD3一致）
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument("--residual_use_layernorm", type=int, default=1)

    # 多卡参数（保留原逻辑，支持多GPU分片）
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    opt = parser.parse_args()
    opt.residual_use_layernorm = bool(opt.residual_use_layernorm)
    assert len(opt.seeds) >= opt.n_samples, "Seeds 数量不足 n_samples"

    return opt

# ===============================================================
# 主流程（完全保留原Qwen代码逻辑，仅替换生成器实例）
# ===============================================================
def main(opt):
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Rank {opt.rank}] Device={device}, world_size={opt.world_size}")

    residual_rotation_matrices = None
    residual_rotation_meta = None
    if opt.residual_procrustes_path is not None:
        residual_rotation_matrices, target_layers, meta = load_residual_procrustes(
            opt.residual_procrustes_path
        )
        residual_rotation_matrices, opt.residual_target_layers = select_residual_rotations(
            residual_rotation_matrices, target_layers, opt.residual_target_layers
        )
        if opt.residual_origin_layer is None and isinstance(meta, dict):
            opt.residual_origin_layer = meta.get("origin_layer")
        residual_rotation_meta = meta

    if opt.residual_weights is None and opt.residual_weights_path is not None:
        opt.residual_weights = load_residual_weights(opt.residual_weights_path).tolist()
        print(f"Residual weights: {opt.residual_weights}")
        print(f"Num res weights: {len(opt.residual_weights)}")
        print(f"Num res targets: {len(opt.residual_target_layers) if opt.residual_target_layers else 0}")

    # 初始化Flux残差生成器（替换原Qwen生成器）
    generator = FluxResGenerator(
        model_dir=opt.model_dir,
        residual_target_layers=opt.residual_target_layers,
        residual_origin_layer=opt.residual_origin_layer,
        residual_weights=opt.residual_weights,
        residual_use_layernorm=opt.residual_use_layernorm,
        residual_rotation_matrices=residual_rotation_matrices,
        residual_rotation_meta=residual_rotation_meta,
        cfg=opt.cfg,
        steps=opt.steps,
        width=opt.width,
        height=opt.height,
    )

    # 扫描 *val.txt（保留原逻辑，读取T2I-CompBench格式prompt）
    txt_files = sorted(glob.glob(os.path.join(opt.dataset_dir, "*val.txt")))
    total_files = len(txt_files)

    # 先收集所有 prompt，按全局索引分片
    prompt_items = []
    for txt_path in txt_files:
        txt_name = os.path.splitext(os.path.basename(txt_path))[0]
        with open(txt_path, "r") as f:
            prompts = [x.strip() for x in f if x.strip()]
        for prompt in prompts:
            prompt_items.append((txt_path, txt_name, prompt))

    total_prompts = len(prompt_items)
    if opt.world_size > 1:
        prompt_items = [
            item for i, item in enumerate(prompt_items) if i % opt.world_size == opt.rank
        ]

    print(
        f"[Rank {opt.rank}] total txt={total_files}, total prompts={total_prompts}, "
        f"this rank prompts={len(prompt_items)}"
    )
    seeds_to_use = opt.seeds[:opt.n_samples]

    # 遍历每行 prompt（断点续跑+批量生成）
    for txt_path, txt_name, prompt in tqdm(
        prompt_items, desc=f"[Rank {opt.rank}] prompts"
    ):
        # 保存路径保留原格式，仅修改前缀为Flux残差标识
        outdir = os.path.join(opt.outdir_base, f"samples_{opt.output_prefix}_{txt_name}")
        os.makedirs(outdir, exist_ok=True)

        prompt_clean = clean_prompt_for_filename(prompt)

        for si, seed in enumerate(seeds_to_use):
            fname = f"{prompt_clean}_{si:06d}.png"
            fpath = os.path.join(outdir, fname)

            if os.path.exists(fpath):
                continue

            # 调用Flux残差生成器（替换原Qwen生成调用）
            img = generator.generate(prompt=prompt, seed=seed)
            save_image(img, fpath, normalize=True)

    print(f"[Rank {opt.rank}] All Done.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
