import argparse
import glob
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from pipeline_taca_flux import FluxPipeline


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def make_grid_2x2(images: List[Image.Image]) -> Image.Image:
    if len(images) != 4:
        raise ValueError("Expected 4 images for a 2x2 grid.")

    w, h = images[0].size
    grid = Image.new("RGB", (w * 2, h * 2))
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (w, 0))
    grid.paste(images[2], (0, h))
    grid.paste(images[3], (w, h))
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TACA FLUX DPG batch generator")

    parser.add_argument("--prompt_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    return parser.parse_args()


def load_prompt_files(prompt_dir: str) -> List[Tuple[int, str]]:
    txt_files = sorted(glob.glob(os.path.join(prompt_dir, "*.txt")))
    return list(enumerate(txt_files))


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    prompt_files = load_prompt_files(args.prompt_dir)
    total_prompts = len(prompt_files)

    if args.world_size > 1:
        prompt_files = [item for item in prompt_files if item[0] % args.world_size == args.rank]

    print(f"[DPG] World size = {args.world_size}, rank = {args.rank}")
    print(f"[DPG] Total prompts: {total_prompts}, this rank will handle: {len(prompt_files)}")

    os.makedirs(args.save_dir, exist_ok=True)

    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.eval()

    if args.lora_weights is not None:
        print(f"[LoRA] Loading weights from: {args.lora_weights}")
        pipe.load_lora_weights(args.lora_weights)

    for index, txt_path in prompt_files:
        base = os.path.basename(txt_path)
        name = os.path.splitext(base)[0]
        out_path = os.path.join(args.save_dir, f"{name}.png")

        if os.path.exists(out_path):
            print(f"[Skip] {name}.png already exists.")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"\n[DPG] Generating for: {name}")

        generators = [
            torch.Generator(device).manual_seed(args.seed + index * 4 + offset)
            for offset in range(4)
        ]
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                height=args.img_size,
                width=args.img_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=4,
                generator=generators,
                max_sequence_length=512,
            )

        grid = make_grid_2x2(result.images)
        grid.save(out_path)
        print(f"[Saved] {out_path}")

    print("\n[DPG] All done.")


if __name__ == "__main__":
    main(parse_args())
