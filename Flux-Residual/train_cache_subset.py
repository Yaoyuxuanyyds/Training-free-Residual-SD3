import argparse
import gc
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
import tqdm

from dataset.datasets import get_target_dataset
from util import build_text_token_nonpad_mask, get_transform, set_seed
from generate_image_res import FluxPipelineWithRES
from flux_transformer_res import FluxTransformer2DModel_RES


@torch.no_grad()
def encode_image_to_latent(pipe: FluxPipelineWithRES, imgs: torch.Tensor) -> torch.Tensor:
    vae = pipe.vae
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    shift = getattr(vae.config, "shift_factor", 0.0)
    imgs = imgs.to(dtype=vae.dtype)
    posterior = vae.encode(imgs * 2 - 1)
    latent_pre = posterior.latent_dist.sample()
    return (latent_pre - shift) * scaling


@torch.no_grad()
def precompute_and_save_features(
    args: argparse.Namespace,
    train_set,
    precompute_dir: str,
    pipe: FluxPipelineWithRES,
    device: torch.device,
):
    cache_dtype = torch.float32
    if args.precision == "fp16":
        cache_dtype = torch.float16
    elif args.precision == "bf16":
        cache_dtype = torch.bfloat16
    os.makedirs(precompute_dir, exist_ok=True)
    loader = DataLoader(
        train_set,
        batch_size=args.pc_batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    cache_group_size = getattr(args, "cache_group_size", 256)
    buffer = []
    file_idx = 0
    idx_global = 0

    pbar = tqdm.tqdm(loader, desc="[Flux Precompute]")
    for imgs, captions in pbar:
        imgs = imgs.to(device)
        imgs = imgs.to(dtype=cache_dtype)

        x0 = encode_image_to_latent(pipe, imgs)

        prompt_emb, pooled_emb, text_ids = pipe.encode_prompt(
            prompt=list(captions),
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
            lora_scale=None,
        )
        token_mask = torch.stack(
            [build_text_token_nonpad_mask(prompt_emb[i]) for i in range(prompt_emb.shape[0])],
            dim=0,
        )

        prompt_emb = prompt_emb.to(dtype=cache_dtype).cpu()
        pooled_emb = pooled_emb.to(dtype=cache_dtype).cpu()
        text_ids = text_ids.cpu()
        token_mask = token_mask.cpu()
        x0 = x0.to(dtype=cache_dtype).cpu()

        for j in range(x0.shape[0]):
            buffer.append(
                {
                    "x0": x0[j],
                    "caption": captions[j],
                    "prompt_emb": prompt_emb[j],
                    "pooled_emb": pooled_emb[j],
                    "text_ids": text_ids[j],
                    "token_mask": token_mask[j],
                    "index": idx_global,
                }
            )
            idx_global += 1

            if len(buffer) >= cache_group_size:
                save_path = osp.join(precompute_dir, f"{file_idx:04d}.pt")
                torch.save(buffer, save_path)
                print(f"[SAVE] {len(buffer)} samples -> {save_path}")
                file_idx += 1
                buffer = []
                gc.collect()

    if len(buffer) > 0:
        save_path = osp.join(precompute_dir, f"{file_idx:04d}.pt")
        torch.save(buffer, save_path)
        print(f"[SAVE] Final {len(buffer)} samples -> {save_path}")

    print(
        f"[INFO] Precompute finished. Total samples: {idx_global}, "
        f"total files: {file_idx + (1 if len(buffer) > 0 else 0)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--pc-batch-size", type=int, default=4)
    parser.add_argument("--cache-group-size", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "fp16":
        torch_dtype = torch.float16
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    pipe = FluxPipelineWithRES.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    pipe.transformer = FluxTransformer2DModel_RES(pipe.transformer).to(device=device)
    pipe.transformer.eval().requires_grad_(False)

    transform = get_transform(args.img_size)
    train_set = get_target_dataset(args.dataset, args.datadir, train=True, transform=transform)

    precompute_and_save_features(args, train_set, args.output_dir, pipe, device)


if __name__ == "__main__":
    main()
