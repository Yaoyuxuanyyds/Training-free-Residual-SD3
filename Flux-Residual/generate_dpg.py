import argparse
import numpy as np
import random
import os
import glob
import torch
from PIL import Image
from typing import List, Optional, Union

# -------------------------- 复用geneval的Pipeline（直接导入）--------------------------
from generate_image_res import FluxPipelineWithRES
from flux_transformer_res import FluxTransformer2DModel_RES
from util import load_residual_procrustes, select_residual_rotations, load_residual_weights

# -------------------------- 工具函数（不变）--------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def make_grid_2x2(imgs):
    """将4张PIL图像拼接为2x2网格（DPG核心需求）"""
    assert len(imgs) == 4, f"DPG要求每个prompt生成4张图，实际收到{len(imgs)}张"
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * 2, h * 2))
    grid.paste(imgs[0], (0, 0))
    grid.paste(imgs[1], (w, 0))
    grid.paste(imgs[2], (0, h))
    grid.paste(imgs[3], (w, h))
    return grid

# -------------------------- 主函数（仅保留单卡+固定seed=42~45）--------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基础参数（固定seed为42，n_samples=4）
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子（固定为42）", choices=[42])  # 强制只能用42
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Flux常用步数")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Flux推荐引导尺度")
    parser.add_argument("--n_samples", type=int, default=4, help="每个prompt生成4张图（固定）", choices=[4])  # 强制4张
    parser.add_argument("--img_size", type=int, default=1024, help="生成图像尺寸（正方形）")
    
    # 模型与路径
    parser.add_argument('--model_path', type=str,
                        default="/inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev",
                        help="Flux模型本地路径")
    parser.add_argument("--save_dir", type=str, required=True, help="DPG输出目录（保存2x2网格图）")
    parser.add_argument("--prompt_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/prompts")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    # 残差参数（与SD3一致）
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None,
                        help="残差目标层索引列表（双流块索引）")
    parser.add_argument("--residual_origin_layer", type=int, default=None,
                        help="残差源层索引（必须是双流块索引）")
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None,
                        help="残差叠加权重（支持多层权重）")
    parser.add_argument("--residual_weights_path", type=str, default=None,
                        help="从文件加载学习得到的残差权重")
    parser.add_argument("--residual_procrustes_path", type=str, default=None,
                        help="Procrustes旋转矩阵路径")

    args = parser.parse_args()
    set_seed(args.seed)  # 全局基础seed=42

    # 设备配置：强制单卡GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("未检测到GPU！当前配置仅支持单卡CUDA设备")

    # -------------------------- 模型加载（单卡全量GPU运行）--------------------------
    print(f"[Flux] 加载模型: {args.model_path}（单卡GPU运行）")
    pipe = FluxPipelineWithRES.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 替换为带残差的Transformer
    print(f"[INFO] 替换前Transformer类型: {type(pipe.transformer)}")
    pipe.transformer = FluxTransformer2DModel_RES(pipe.transformer)
    print(f"[INFO] 替换后Transformer类型: {type(pipe.transformer)}")
    if not isinstance(pipe.transformer, FluxTransformer2DModel_RES):
        raise RuntimeError("残差Transformer替换失败！请检查flux_transformer_res.py")

    pipe.to(device)
    print(f"[Flux] 模型已移至 {device}")

    # 推理模式优化
    core_modules = [pipe.text_encoder, pipe.text_encoder_2, pipe.transformer, pipe.vae]
    for module in core_modules:
        if module is not None:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    residual_rotation_matrices = None
    residual_rotation_meta = None
    if args.residual_procrustes_path is not None:
        residual_rotation_matrices, target_layers, meta = load_residual_procrustes(
            args.residual_procrustes_path
        )
        residual_rotation_matrices, args.residual_target_layers = select_residual_rotations(
            residual_rotation_matrices, target_layers, args.residual_target_layers
        )
        if args.residual_origin_layer is None and isinstance(meta, dict):
            args.residual_origin_layer = meta.get("origin_layer")
        residual_rotation_meta = meta

    if args.residual_weights is None and args.residual_weights_path is not None:
        args.residual_weights = load_residual_weights(args.residual_weights_path).tolist()
        print(f"Residual weights: {args.residual_weights}")
        print(f"Num res weights: {len(args.residual_weights)}")
        print(f"Num res targets: {len(args.residual_target_layers) if args.residual_target_layers else 0}")

    # -------------------------- 残差配置--------------------------
    residual_config = {
        "residual_target_layers": args.residual_target_layers,
        "residual_origin_layer": args.residual_origin_layer,
        "residual_weights": args.residual_weights,
        "residual_rotation_matrices": residual_rotation_matrices,
        "residual_rotation_meta": residual_rotation_meta,
    }

    # -------------------------- DPG核心流程（固定seed=42~45）--------------------------
    os.makedirs(args.save_dir, exist_ok=True)

    # 扫描prompt文件（单卡处理所有prompt）
    txt_files = sorted(glob.glob(os.path.join(args.prompt_dir, "*.txt")))
    if args.world_size > 1:
        txt_files = [f for i, f in enumerate(txt_files) if i % args.world_size == args.rank]
    total_prompts = len(txt_files)
    print(f"[DPG] 单卡运行，共处理 {total_prompts} 个prompt")
    print(f"[Seed配置] 每个prompt的4个样本seed固定为：{args.seed}、{args.seed+1}、{args.seed+2}、{args.seed+3}")

    # 遍历每个prompt生成
    for prompt_idx, txt_path in enumerate(txt_files):
        base = os.path.basename(txt_path)
        name = os.path.splitext(base)[0]
        out_path = os.path.join(args.save_dir, f"{name}.png")

        # 跳过已生成文件（断点续跑）
        if os.path.exists(out_path):
            print(f"[Skip] {name}.png 已存在")
            continue

        # 读取prompt
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        if not prompt:
            print(f"[Warn] {txt_path} 为空，跳过")
            continue

        print(f"\n[DPG] 生成进度: {prompt_idx+1}/{total_prompts} | {name}")
        print(f"[Prompt] {prompt[:100]}..." if len(prompt) > 100 else f"[Prompt] {prompt}")
        print(
            f"[Residual] 源层: {residual_config['residual_origin_layer']} | "
            f"目标层: {residual_config['residual_target_layers']} | "
            f"权重: {residual_config['residual_weights']}"
        )

        # 生成4张图像（seed固定为42、43、44、45，不叠加任何索引）
        with torch.inference_mode():
            imgs = []
            for sample_idx in range(args.n_samples):
                # 核心：固定seed=42+sample_idx（每个prompt都复用这4个seed）
                sample_seed = args.seed + sample_idx
                set_seed(sample_seed)  # 每个样本单独固定seed

                result = pipe(
                    prompt=prompt,
                    height=args.img_size,
                    width=args.img_size,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    max_sequence_length=512,
                    generator=torch.Generator(device).manual_seed(sample_seed),
                    **residual_config
                )
                imgs.append(result.images[0])
                print(f"[Seed] 样本{sample_idx} → {sample_seed}")  # 明确打印seed，确认符合要求

        # 拼接2x2网格并保存
        grid = make_grid_2x2(imgs)
        grid.save(out_path)
        print(f"[Saved] 2x2网格图 → {out_path}")

        # 清理显存
        torch.cuda.empty_cache()

    print(f"\n[DPG] 所有任务完成！输出目录：{args.save_dir}")
