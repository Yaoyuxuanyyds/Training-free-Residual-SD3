import argparse
import json
import os
import torch
from tqdm import trange
from torchvision.utils import save_image

from generate_image_res import SD35PipelineWithRES
from sd35_transformer_res import SD35Transformer2DModel_RES
from util import (
    load_residual_procrustes,
    select_residual_rotations,
    set_seed,
    load_residual_weights,
)
from lora_utils import inject_lora, load_lora_state_dict


torch.set_grad_enabled(False)

DEFAULT_SD35_MODEL = "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5-large"


class SD35ImageGenerator:
    def __init__(
        self,
        model_path=None,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_rotation_matrices=None,
        residual_rotation_meta=None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = model_path or DEFAULT_SD35_MODEL
        self.pipe = SD35PipelineWithRES.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.pipe.transformer = SD35Transformer2DModel_RES(self.pipe.transformer)

        self.residual_target_layers = residual_target_layers
        self.residual_origin_layer = residual_origin_layer
        self.residual_weights = residual_weights
        self.residual_rotation_matrices = residual_rotation_matrices
        self.residual_rotation_meta = residual_rotation_meta

    def generate_image(
        self,
        prompt,
        seed=0,
        img_size=1024,
        num_inference_steps=28,
        guidance_scale=7.0,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_rotation_matrices=None,
        residual_rotation_meta=None,
    ):
        rt = residual_target_layers if residual_target_layers is not None else self.residual_target_layers
        ro = residual_origin_layer if residual_origin_layer is not None else self.residual_origin_layer
        rw = residual_weights if residual_weights is not None else self.residual_weights
        rr = residual_rotation_matrices if residual_rotation_matrices is not None else self.residual_rotation_matrices
        rr_meta = residual_rotation_meta if residual_rotation_meta is not None else self.residual_rotation_meta

        set_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                height=img_size,
                width=img_size,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pt",
                residual_target_layers=rt,
                residual_origin_layer=ro,
                residual_weights=rw,
                residual_rotation_matrices=rr,
                residual_rotation_meta=rr_meta,
            )
        image = result.images[0] if hasattr(result, "images") else result[0]
        if image.dim() == 4:
            image = image[0]
        return image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None, help='Path to LoRA-only checkpoint (.pth)')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target', type=str, default='all_linear',
                        help="all_linear 或模块名片段，如: to_q,to_k,to_v,to_out")
    parser.add_argument('--lora_dropout', type=float, default=0.0)

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--load_dir", type=str, default=None)

    return parser.parse_args()


def main(args):
    if not os.path.exists(args.metadata_file):
        raise FileNotFoundError(f"metadata 文件不存在：{args.metadata_file}")
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    total_items = len(metadatas)

    if args.world_size > 1:
        local_indices = [i for i in range(total_items) if i % args.world_size == args.rank]
    else:
        local_indices = list(range(total_items))

    print(f"[Rank {args.rank}] 总任务: {total_items}，本卡处理: {len(local_indices)}")

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
        args.residual_weights = load_residual_weights(args.residual_weights_path)

    generator = SD35ImageGenerator(
        model_path=args.load_dir or DEFAULT_SD35_MODEL,
        residual_target_layers=args.residual_target_layers,
        residual_origin_layer=args.residual_origin_layer,
        residual_weights=args.residual_weights,
        residual_rotation_matrices=residual_rotation_matrices,
        residual_rotation_meta=residual_rotation_meta,
    )

    if args.lora_ckpt is not None:
        print(f"[LoRA] injecting & loading LoRA from: {args.lora_ckpt}")
        target = "all_linear" if args.lora_target == "all_linear" else tuple(args.lora_target.split(","))
        inject_lora(
            generator.pipe.transformer,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target=target,
            dropout=args.lora_dropout,
            is_train=False,
        )
        lora_sd = torch.load(args.lora_ckpt, map_location="cpu")
        load_lora_state_dict(generator.pipe.transformer, lora_sd, strict=True)
        generator.pipe.transformer.eval()
        print("[LoRA] loaded and ready.")

    for index in local_indices:
        metadata = metadatas[index]
        prompt = metadata["prompt"]

        outpath = os.path.join(args.outdir, f"{index:05d}")
        sample_path = os.path.join(outpath, "samples")

        os.makedirs(sample_path, exist_ok=True)

        print(f"[Rank {args.rank}] Prompt {index:05d}/{total_items}: {prompt}")

        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            for _ in trange(
                (args.n_samples + args.batch_size - 1) // args.batch_size,
                desc=f"[Rank {args.rank}] Sampling {index:05d}",
            ):
                current_bs = min(args.batch_size, args.n_samples - sample_count)

                for _ in range(current_bs):
                    img_path = os.path.join(sample_path, f"{sample_count:05d}.png")

                    if os.path.exists(img_path):
                        print(f"[Rank {args.rank}] 跳过已存在: {img_path}")
                        sample_count += 1
                        continue

                    image = generator.generate_image(
                        prompt=prompt,
                        seed=args.seed + sample_count,
                        residual_target_layers=args.residual_target_layers,
                        residual_origin_layer=args.residual_origin_layer,
                        residual_weights=args.residual_weights,
                        residual_rotation_matrices=residual_rotation_matrices,
                    )

                    if image.dim() == 3 and image.shape[0] != 3 and image.shape[-1] == 3:
                        image = image.permute(2, 0, 1)

                    save_image(image, img_path, normalize=True)
                    sample_count += 1

    print(f"[Rank {args.rank}] 完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
