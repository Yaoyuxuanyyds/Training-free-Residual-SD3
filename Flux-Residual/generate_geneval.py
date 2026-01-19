import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from typing import List, Optional, Union
from torchvision.utils import make_grid, save_image
import random

# -------------------------- 直接导入已有带残差的Pipeline --------------------------
# 从 generate_image_res.py 中导入你已经定义好的 FluxPipelineWithRES
from generate_image_res import FluxPipelineWithRES
# 导入自定义残差Transformer（用于替换验证）
from flux_transformer_res import FluxTransformer2DModel_RES
from util import load_residual_procrustes, select_residual_rotations, load_residual_weights


# -------------------------- 工具函数（不变）--------------------------
def set_seed(seed):
    """固定随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# -------------------------- 生成器类（复用已有Pipeline，仅封装geneval逻辑）--------------------------
class FluxGenevalGenerator:
    def __init__(
        self,
        model_path: str,
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weights: Optional[List[float]] = None,
        residual_rotation_matrices: Optional[torch.Tensor] = None,
        residual_rotation_meta: Optional[dict] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.device == "cuda":
            raise Warning("未检测到GPU，生成速度会极慢！")

        # 直接使用导入的 FluxPipelineWithRES
        print(f"[INFO] 加载模型: {model_path}")
        self.pipe = FluxPipelineWithRES.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # 替换为自定义残差Transformer（核心步骤，必须保留）
        print(f"[INFO] 替换前Transformer类型: {type(self.pipe.transformer)}")
        self.pipe.transformer = FluxTransformer2DModel_RES(self.pipe.transformer)
        print(f"[INFO] 替换后Transformer类型: {type(self.pipe.transformer)}")
        
        # 验证替换是否成功
        if not isinstance(self.pipe.transformer, FluxTransformer2DModel_RES):
            raise RuntimeError("残差Transformer替换失败！请检查flux_transformer_res.py的导入和定义")

        # 显存优化（启用CPU卸载，删除手动 pipe.to("cuda")，避免冲突）
        # self.pipe.enable_model_cpu_offload()
        self.pipe.to(self.device)

        # 推理模式配置
        core_modules = [self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.transformer, self.pipe.vae]
        for module in core_modules:
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        # 保存残差配置
        self.residual_config = {
            "residual_target_layers": residual_target_layers,
            "residual_origin_layer": residual_origin_layer,
            "residual_weights": residual_weights,
            "residual_rotation_matrices": residual_rotation_matrices,
            "residual_rotation_meta": residual_rotation_meta,
        }

    def generate_single_image(
        self,
        prompt: str,
        seed: int,
        img_size: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """生成单张图像，复用已有Pipeline的__call__逻辑"""
        set_seed(seed)

        # 调用导入的Pipeline，直接传递残差参数
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                height=img_size,
                width=img_size,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                generator=torch.Generator(self.device).manual_seed(seed),
                **self.residual_config  # 注入残差参数
            )

        # 转换为geneval要求的 [-1, 1] 张量格式
        img = result.images[0]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor * 2.0) - 1.0
        return img_tensor


# -------------------------- 参数解析（不变）--------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Flux残差版geneval批量生成工具")

    # geneval核心参数
    parser.add_argument("--metadata_file", type=str,
                        default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/prompts/evaluation_metadata.jsonl",
                        help="JSONL格式的prompt元数据文件")
    parser.add_argument("--outdir", type=str,
                        default="/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3-1/flux/geneval_outputs/out_test",
                        help="输出根目录")
    parser.add_argument("--n_samples", type=int, default=4, help="每个prompt生成的样本数")
    parser.add_argument("--seed", type=int, default=0, help="基础随机种子")
    parser.add_argument("--batch_size", type=int, default=16, help="网格图每行显示数量")
    parser.add_argument("--skip_grid", action="store_true", help="是否跳过生成网格图")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    # Flux生成参数
    parser.add_argument("--model_path", type=str,
                        default="/inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev",
                        help="Flux模型本地路径")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="去噪推理步数")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="文本引导强度")
    parser.add_argument("--img_size", type=int, default=1024, help="生成图像尺寸（正方形）")

    # 残差参数（与SD3一致）
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None,
                        help="残差目标层索引列表（如：--residual_target_layers 6 7 8）")
    parser.add_argument("--residual_origin_layer", type=int, default=None,
                        help="残差源层索引（必须是双流块索引）")
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None,
                        help="残差叠加权重（支持多层权重）")
    parser.add_argument("--residual_weights_path", type=str, default=None,
                        help="从文件加载学习得到的残差权重")
    parser.add_argument("--residual_procrustes_path", type=str, default=None,
                        help="Procrustes旋转矩阵路径")

    return parser.parse_args()


# -------------------------- 主函数（geneval批量生成逻辑，不变）--------------------------
def main(opt):
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"[INFO] 输出目录: {opt.outdir}")

    # 读取元数据
    if not os.path.exists(opt.metadata_file):
        raise FileNotFoundError(f"元数据文件不存在：{opt.metadata_file}")
    with open(opt.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line.strip()) for line in fp if line.strip()]
    print(f"[INFO] 共加载 {len(metadatas)} 个prompt")

    # 初始化生成器（使用导入的Pipeline）
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

    generator = FluxGenevalGenerator(
        model_path=opt.model_path,
        residual_target_layers=opt.residual_target_layers,
        residual_origin_layer=opt.residual_origin_layer,
        residual_weights=opt.residual_weights,
        residual_rotation_matrices=residual_rotation_matrices,
        residual_rotation_meta=residual_rotation_meta,
    )

    # 批量生成
    metadatas_with_idx = list(enumerate(metadatas))
    if opt.world_size > 1:
        metadatas_with_idx = [
            (i, m) for i, m in metadatas_with_idx if i % opt.world_size == opt.rank
        ]
        print(f"[Rank {opt.rank}] total prompts={len(metadatas_with_idx)}")
    for prompt_idx, metadata in metadatas_with_idx:
        prompt = metadata["prompt"]
        prompt_dir = os.path.join(opt.outdir, f"{prompt_idx:05d}")
        sample_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)

        # 保存metadata
        with open(os.path.join(prompt_dir, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)

        print(f"\n[Prompt {prompt_idx:05d}/{len(metadatas)}] {prompt}")
        all_samples = []
        sample_count = 0

        total_batches = (opt.n_samples + opt.batch_size - 1) // opt.batch_size
        for batch_idx in trange(total_batches, desc="生成进度", leave=False):
            current_batch_size = min(opt.batch_size, opt.n_samples - sample_count)

            for _ in range(current_batch_size):
                sample_seed = opt.seed + sample_count
                sample_path = os.path.join(sample_dir, f"{sample_count:05d}.png")

                if os.path.exists(sample_path):
                    print(f"[跳过] 已存在样本: {sample_path}")
                    sample_count += 1
                    continue

                # 调用生成器（复用导入的Pipeline逻辑）
                img_tensor = generator.generate_single_image(
                    prompt=prompt,
                    seed=sample_seed,
                    img_size=opt.img_size,
                    num_inference_steps=opt.num_inference_steps,
                    guidance_scale=opt.guidance_scale,
                )

                # 保存图像
                save_image(img_tensor, sample_path, normalize=True)
                print(f"[生成] 样本 {sample_count:05d} → {sample_path}")

                if not opt.skip_grid:
                    all_samples.append(img_tensor.unsqueeze(0))

                sample_count += 1

            torch.cuda.empty_cache()

        # 生成网格图
        if not opt.skip_grid and all_samples:
            grid_tensor = torch.cat(all_samples, dim=0)
            grid_img = make_grid(grid_tensor, nrow=opt.batch_size, normalize=True)
            grid_path = os.path.join(prompt_dir, "grid.png")
            save_image(grid_img, grid_path)
            print(f"[生成] 网格图 → {grid_path}")

        del all_samples
        torch.cuda.empty_cache()

    print("\n[INFO] 所有prompt生成完成！")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
