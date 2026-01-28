import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image

from pipeline_taca_sd3 import StableDiffusion3Pipeline


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TACA SD3.5 geneval batch generator")

    parser.add_argument("--metadata_file", type=str,
                        default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/prompts/evaluation_metadata.jsonl",
                        help="JSONL格式的prompt元数据文件")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5-large")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--lora_weights", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/TACA/TACA/sd35-medium-lora-rank-64.safetensors")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    return parser.parse_args()


def load_metadata(metadata_file: str) -> List[dict]:
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"metadata 文件不存在：{metadata_file}")
    with open(metadata_file, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    metadatas = load_metadata(args.metadata_file)
    total_items = len(metadatas)

    if args.world_size > 1:
        local_indices = [i for i in range(total_items) if i % args.world_size == args.rank]
    else:
        local_indices = list(range(total_items))

    print(f"[Rank {args.rank}] 总任务: {total_items}，本卡处理: {len(local_indices)}")
    os.makedirs(args.outdir, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.eval()

    if args.lora_weights is not None:
        print(f"[LoRA] Loading weights from: {args.lora_weights}")
        pipe.load_lora_weights(args.lora_weights)

    for index in local_indices:
        metadata = metadatas[index]
        prompt = metadata["prompt"]

        outpath = os.path.join(args.outdir, f"{index:05d}")
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        print(f"[Rank {args.rank}] Prompt {index:05d}/{total_items}: {prompt}")

        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)

        sample_count = 0
        total_batches = (args.n_samples + args.batch_size - 1) // args.batch_size

        for _ in range(total_batches):
            current_bs = min(args.batch_size, args.n_samples - sample_count)
            for _ in range(current_bs):
                img_path = os.path.join(sample_path, f"{sample_count:05d}.png")
                if os.path.exists(img_path):
                    print(f"[Rank {args.rank}] 跳过已存在: {img_path}")
                    sample_count += 1
                    continue

                generator = torch.Generator(device).manual_seed(args.seed + sample_count)
                with torch.inference_mode():
                    result = pipe(
                        prompt=prompt,
                        height=args.img_size,
                        width=args.img_size,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        num_images_per_prompt=1,
                        generator=generator,
                    )

                image: Image.Image = result.images[0]
                image.save(img_path)
                sample_count += 1

        torch.cuda.empty_cache()

    print(f"[Rank {args.rank}] 完成！")


if __name__ == "__main__":
    main(parse_args())
