import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from typing import List, Optional, Union
from torchvision.utils import make_grid, save_image
import random

# -------------------------- 核心：导入你已有的 SD3.5 残差 Pipeline 和 Transformer --------------------------
# 从你的 SD3.5 生成代码中直接导入（路径已匹配你的项目结构）
from generate_image_res import (
    SD35PipelineWithRES,  # 已定义的带残差 Pipeline
    SD35Transformer2DModel_RES,  # 已定义的残差 Transformer
    XLA_AVAILABLE,  # 复用原有 XLA 定义（不影响，仅保留兼容性）
    # xm  # 复用原有 XLA 模块（不影响）
)
# from diffusers.utils import is_torch_xla_available  # 复用原有依赖


# -------------------------- 工具函数（复用+适配 SD3.5）--------------------------
def set_seed(seed):
    """固定随机种子，保证 Geneval 结果可复现（复用你原有逻辑）"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# -------------------------- Geneval 生成器类（封装批量逻辑，完全复用已有 Pipeline）--------------------------
class SD35GenevalGenerator:
    def __init__(
        self,
        model_path: str,
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weight: float = 0.0,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise Warning("未检测到GPU，SD3.5生成速度会极慢！")

        # 1. 加载你已定义的 SD35PipelineWithRES（无重复代码）
        print(f"[INFO] 加载 SD3.5 模型: {model_path}")
        self.pipe = SD35PipelineWithRES.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # 复用你原有 dtype 配置
            trust_remote_code=True
        ).to(self.device)  # 复用你原有设备配置

        # 2. 替换为你的残差 Transformer（核心步骤，复用已有类）
        print(f"[INFO] 替换前 Transformer 类型: {type(self.pipe.transformer)}")
        self.pipe.transformer = SD35Transformer2DModel_RES(self.pipe.transformer)
        print(f"[INFO] 替换后 Transformer 类型: {type(self.pipe.transformer)}")
        
        # 验证替换是否成功（避免残差逻辑未生效）
        if not isinstance(self.pipe.transformer, SD35Transformer2DModel_RES):
            raise RuntimeError("SD35 残差 Transformer 替换失败！请检查 sd35_transformer_res.py 定义")

        # 3. 推理模式配置（复用你原有逻辑，禁用梯度计算）
        core_modules = [
            self.pipe.text_encoder,    # SD3.5 文本编码器1（CLIP）
            self.pipe.text_encoder_2,  # SD3.5 文本编码器2（CLIP）
            self.pipe.text_encoder_3,  # SD3.5 文本编码器3（T5）
            self.pipe.transformer,     # 残差 Transformer（核心）
            self.pipe.vae              # SD3.5 VAE 解码器
        ]
        for module in core_modules:
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        # 4. 保存残差配置（后续批量生成时透传）
        self.residual_config = {
            "residual_target_layers": residual_target_layers,
            "residual_origin_layer": residual_origin_layer,
            "residual_weight": residual_weight,
        }

    def generate_single_image(
        self,
        prompt: str,
        seed: int,
        img_size: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,  # SD3.5 默认 guidance_scale=7.0（适配原有逻辑）
    ) -> torch.Tensor:
        """生成单张 Geneval 格式图像，完全复用你 Pipeline 的 __call__ 逻辑"""
        set_seed(seed)

        # 调用你已定义的 Pipeline，透传残差参数（无重复代码）
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,          # Geneval 单 prompt 输入
                prompt_2=None,          # SD3.5 可选 prompt_2（Geneval 不用，设为 None）
                prompt_3=None,          # SD3.5 可选 prompt_3（Geneval 不用，设为 None）
                height=img_size,
                width=img_size,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=256,  # SD3.5 默认 T5 序列长度（复用你原有默认值）
                generator=torch.Generator(self.device).manual_seed(seed),  # 固定种子
                negative_prompt=None,    # Geneval 通常不使用负 prompt
                negative_prompt_2=None,
                negative_prompt_3=None,
                **self.residual_config   # 注入残差参数（核心：复用你原有残差逻辑）
            )

        # 转换为 Geneval 标准格式（[-1, 1] 张量，适配评估工具）
        img = result.images[0]  # 从 Pipeline 输出中取第一张图
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # HWC→CHW，归一化到 [0,1]
        img_tensor = (img_tensor * 2.0) - 1.0  # 转换到 Geneval 要求的 [-1, 1]
        return img_tensor


# -------------------------- Geneval 参数解析（适配 SD3.5 特性）--------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SD3.5 残差版 Geneval 批量生成工具")

    # 1. Geneval 标准参数（固定格式，不修改）
    parser.add_argument("--metadata_file", type=str,
                        default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/prompts/evaluation_metadata.jsonl",
                        help="Geneval 标准 prompt 元数据文件（JSONL 格式）")
    parser.add_argument("--outdir", type=str,
                        default="/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3-1/sd3.5/geneval_outputs/out_test",
                        help="SD3.5 Geneval 输出根目录（自动创建）")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="每个 prompt 生成的样本数（Geneval 标准要求≥4）")
    parser.add_argument("--seed", type=int, default=0,
                        help="基础随机种子（样本种子=seed+样本索引，保证可复现）")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="网格图每行显示数量（SD3.5 显存有限，建议≤4）")
    parser.add_argument("--skip_grid", action="store_true",
                        help="是否跳过生成网格图（节省时间）")

    # 2. SD3.5 生成参数（复用你原有默认值）
    parser.add_argument("--model_path", type=str,
                        default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5-large",
                        help="SD3.5 模型本地路径（你的模型路径）")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                        help="SD3.5 去噪推理步数（复用你原有默认值）")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="SD3.5 文本引导强度（复用你原有默认值，勿改）")
    parser.add_argument("--img_size", type=int, default=1024,
                        help="SD3.5 生成图像尺寸（正方形，建议≤1024）")

    # 3. 残差参数（完全匹配你原有定义，无修改）
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=[6,7,8,9,10],
                        help="残差目标层索引列表（如：--residual_target_layers 6 7 8）")
    parser.add_argument("--residual_origin_layer", type=int, default=1,
                        help="残差源层索引（SD3.5 Transformer 层数范围 0~17，复用你原有默认值）")
    parser.add_argument("--residual_weight", type=float, default=0.15,
                        help="残差叠加权重（建议 0.1~0.5，复用你原有调试值）")

    return parser.parse_args()


# -------------------------- Geneval 主逻辑（批量生成+保存，适配 SD3.5）--------------------------
def main(opt):
    # 创建输出目录（Geneval 标准结构）
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"[INFO] SD3.5 Geneval 输出目录: {opt.outdir}")

    # 读取 Geneval 元数据（标准 JSONL 格式）
    if not os.path.exists(opt.metadata_file):
        raise FileNotFoundError(f"Geneval 元数据文件不存在：{opt.metadata_file}")
    with open(opt.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line.strip()) for line in fp if line.strip()]
    print(f"[INFO] 共加载 {len(metadatas)} 个 Geneval prompt")

    # 初始化 SD3.5 残差生成器（复用已有 Pipeline）
    generator = SD35GenevalGenerator(
        model_path=opt.model_path,
        residual_target_layers=opt.residual_target_layers,
        residual_origin_layer=opt.residual_origin_layer,
        residual_weight=opt.residual_weight,
    )

    # 批量生成每个 prompt 的样本
    for prompt_idx, metadata in enumerate(metadatas):
        prompt = metadata["prompt"]  # Geneval 标准 prompt 字段
        prompt_dir = os.path.join(opt.outdir, f"{prompt_idx:05d}")  # 每个 prompt 单独目录
        sample_dir = os.path.join(prompt_dir, "samples")  # 样本保存子目录
        os.makedirs(sample_dir, exist_ok=True)

        # 保存当前 prompt 的元数据（Geneval 标准要求）
        with open(os.path.join(prompt_dir, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)

        print(f"\n[Prompt {prompt_idx:05d}/{len(metadatas)}] {prompt[:50]}...")  # 截断长 prompt 显示
        all_samples = []  # 用于生成网格图
        sample_count = 0  # 已生成样本数

        # 按 batch 生成（适配 SD3.5 显存）
        total_batches = (opt.n_samples + opt.batch_size - 1) // opt.batch_size
        for batch_idx in trange(total_batches, desc="生成进度", leave=False):
            current_batch_size = min(opt.batch_size, opt.n_samples - sample_count)

            for _ in range(current_batch_size):
                sample_seed = opt.seed + sample_count  # 每个样本种子递增（可复现）
                sample_path = os.path.join(sample_dir, f"{sample_count:05d}.png")

                # 跳过已生成的样本（避免重复）
                if os.path.exists(sample_path):
                    print(f"[跳过] 已存在样本: {sample_path}")
                    sample_count += 1
                    continue

                # 调用生成器生成单张图（复用已有逻辑）
                img_tensor = generator.generate_single_image(
                    prompt=prompt,
                    seed=sample_seed,
                    img_size=opt.img_size,
                    num_inference_steps=opt.num_inference_steps,
                    guidance_scale=opt.guidance_scale,
                )

                # 保存样本（Geneval 标准格式：PNG）
                save_image(img_tensor, sample_path, normalize=True)
                print(f"[生成] 样本 {sample_count:05d} → {sample_path}")

                # 收集样本用于生成网格图（如果未跳过）
                if not opt.skip_grid:
                    all_samples.append(img_tensor.unsqueeze(0))

                sample_count += 1

            # 清理显存（避免 SD3.5 显存溢出）
            torch.cuda.empty_cache()
            if XLA_AVAILABLE:
                xm.mark_step()  # 复用你原有 XLA 兼容逻辑

        # 生成网格图（Geneval 可视化需求）
        if not opt.skip_grid and all_samples:
            grid_tensor = torch.cat(all_samples, dim=0)
            grid_img = make_grid(grid_tensor, nrow=opt.batch_size, normalize=True)
            grid_path = os.path.join(prompt_dir, "grid.png")
            save_image(grid_img, grid_path)
            print(f"[生成] 网格图 → {grid_path}")

        # 释放当前 prompt 的内存
        del all_samples
        torch.cuda.empty_cache()

    print("\n[INFO] SD3.5 残差版 Geneval 所有 prompt 生成完成！")


# -------------------------- 入口函数（直接运行）--------------------------
if __name__ == "__main__":
    opt = parse_args()
    main(opt)