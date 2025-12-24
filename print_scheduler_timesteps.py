import argparse
import torch
from diffusers import StableDiffusion3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Print scheduler timesteps for SD3.")
    parser.add_argument("--model_key", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps (N)")
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_key,
        torch_dtype=torch.float16,
    )
    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.steps, device="cpu")
    timesteps = scheduler.timesteps
    if isinstance(timesteps, torch.Tensor):
        timesteps = timesteps.cpu().tolist()
    for idx, t in enumerate(timesteps):
        print(f"step {idx:03d}: {int(round(t))}")


if __name__ == "__main__":
    main()
