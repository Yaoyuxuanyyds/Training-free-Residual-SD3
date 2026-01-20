import argparse
import os
import os.path as osp
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm

from dataset.datasets import CachedFeatureDataset_Packed, collate_fn_packed, get_target_dataset
from generate_image_res import FluxPipelineWithRES
from flux_transformer_res import FluxTransformer2DModel_RES
from util import (
    get_transform,
    load_residual_procrustes,
    select_residual_rotations,
    set_seed,
    resolve_rotation_bucket,
)


def sample_timesteps(batch_size, num_steps, device, mode="uniform"):
    if mode == "uniform":
        return torch.randint(0, num_steps, (batch_size,), device=device)
    raise ValueError(f"Unknown mode: {mode}")


def _extract_images_and_prompts(batch):
    if isinstance(batch, dict):
        images = batch.get("image") or batch.get("img") or batch.get("pixel_values")
        prompts = batch.get("prompt") or batch.get("caption") or batch.get("text")
        if images is None or prompts is None:
            raise ValueError("Unable to extract image/prompt from batch dictionary.")
        return images, prompts
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch tuple does not contain image and prompt.")
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


@torch.no_grad()
def encode_images_to_latents(pipe: FluxPipelineWithRES, images: torch.Tensor) -> torch.Tensor:
    vae = pipe.vae
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    shift = getattr(vae.config, "shift_factor", 0.0)
    images = images.to(dtype=vae.dtype)
    posterior = vae.encode(images * 2 - 1)
    latent_pre = posterior.latent_dist.sample()
    return (latent_pre - shift) * scaling


def build_latent_image_ids(
    pipe: FluxPipelineWithRES,
    batch_size: int,
    img_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    num_channels_latents = pipe.transformer.config.in_channels // 4
    _, latent_image_ids = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        img_size,
        img_size,
        dtype,
        device,
        None,
    )
    return latent_image_ids


def compute_total_loss(
    pipe: FluxPipelineWithRES,
    x0: torch.Tensor,
    t: torch.Tensor,
    prompt_emb: torch.Tensor,
    pooled_emb: torch.Tensor,
    text_ids: torch.Tensor,
    img_size: int,
    residual_target_layers: Optional[List[int]] = None,
    residual_origin_layer: Optional[int] = None,
    residual_weights: Optional[torch.Tensor] = None,
    residual_use_layernorm: bool = True,
    residual_rotation_matrices: Optional[torch.Tensor] = None,
    residual_rotation_meta: Optional[dict] = None,
    guidance_scale: float = 3.5,
):
    device = x0.device
    B = x0.shape[0]
    T = int(pipe.scheduler.config.num_train_timesteps)

    s = (t.float() / float(T)).view(B, 1, 1, 1)
    x1 = torch.randn_like(x0)
    x_s = (1 - s) * x0 + s * x1
    v_target = x1 - x0

    selected_rotations = resolve_rotation_bucket(
        residual_rotation_matrices,
        residual_rotation_meta,
        t,
    )

    latent_image_ids = build_latent_image_ids(
        pipe,
        B,
        img_size=img_size,
        dtype=prompt_emb.dtype,
        device=device,
    )

    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([B], guidance_scale, device=device, dtype=torch.float32)

    out = pipe.transformer(
        hidden_states=x_s,
        timestep=t.to(dtype=prompt_emb.dtype) / 1000.0,
        guidance=guidance,
        encoder_hidden_states=prompt_emb,
        pooled_projections=pooled_emb,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
        residual_target_layers=residual_target_layers,
        residual_origin_layer=residual_origin_layer,
        residual_weights=residual_weights,
        residual_use_layernorm=residual_use_layernorm,
        residual_rotation_matrices=selected_rotations,
    )[0]

    denoise_loss = F.mse_loss(out, v_target)
    return denoise_loss


def _normalize_prompt_data(
    prompt_emb: torch.Tensor,
    pooled_emb: torch.Tensor,
    text_ids: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prompt_emb.ndim == 4 and prompt_emb.shape[1] == 1:
        prompt_emb = prompt_emb[:, 0]
    if pooled_emb.ndim == 3 and pooled_emb.shape[1] == 1:
        pooled_emb = pooled_emb[:, 0]

    if text_ids.ndim == 3 and text_ids.shape[0] == 1:
        text_ids = text_ids[0]
    elif text_ids.ndim == 2 and text_ids.shape[-1] == 3 and text_ids.shape[0] == prompt_emb.shape[0]:
        text_ids = torch.zeros(
            prompt_emb.shape[1],
            3,
            device=device,
            dtype=prompt_emb.dtype,
        )
    elif text_ids.ndim == 1 and text_ids.shape[0] == 3:
        text_ids = torch.zeros(
            prompt_emb.shape[1],
            3,
            device=device,
            dtype=prompt_emb.dtype,
        )

    return prompt_emb, pooled_emb, text_ids


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    pipe = FluxPipelineWithRES.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    pipe.transformer = FluxTransformer2DModel_RES(pipe.transformer).to(device=device, dtype=torch.float32)
    pipe.transformer.eval().requires_grad_(False)

    use_cache = args.precompute_dir is not None
    if use_cache:
        dataset = CachedFeatureDataset_Packed(
            cache_dirs=args.precompute_dir,
            target_batch_size=args.batch_size,
            cache_meta=True,
        )
    else:
        if args.datadir is None or args.dataset is None:
            raise ValueError("datadir and dataset must be provided when precompute_dir is not set.")
        transform = get_transform(args.img_size)
        dataset = get_target_dataset(args.dataset, args.datadir, train=True, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=1 if use_cache else args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=collate_fn_packed if use_cache else None,
        drop_last=not use_cache,
    )

    num_layers = len(pipe.transformer.base_model.transformer_blocks)
    residual_origin_layer = args.residual_origin_layer if args.residual_origin_layer is not None else 1
    residual_target_layers = args.residual_target_layers
    if residual_target_layers is None:
        residual_target_layers = list(range(residual_origin_layer + 1, num_layers))
    if len(residual_target_layers) == 0:
        raise ValueError("residual_target_layers cannot be empty.")

    residual_rotation_matrices = None
    residual_rotation_meta = None
    if args.residual_rotation_path is not None:
        residual_rotation_matrices, target_layers, meta = load_residual_procrustes(
            args.residual_rotation_path,
            device=device,
            dtype=torch.float32,
        )
        residual_rotation_matrices, residual_target_layers = select_residual_rotations(
            residual_rotation_matrices, target_layers, residual_target_layers
        )
        residual_rotation_meta = meta
        if args.residual_origin_layer is None and isinstance(meta, dict):
            residual_origin_layer = meta.get("origin_layer", residual_origin_layer)

    def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 20, x, torch.log(torch.expm1(x)))

    residual_init = torch.tensor(args.residual_init, device=device, dtype=torch.float32)
    residual_init = torch.clamp(residual_init, min=1e-6)
    residual_weights_raw = torch.nn.Parameter(
        softplus_inverse(residual_init).repeat(len(residual_target_layers))
    )

    optimizer = torch.optim.AdamW([residual_weights_raw], lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    pbar = tqdm.tqdm(total=args.steps, desc="[Flux Residual Weights]")

    while global_step < args.steps:
        for batch in loader:
            if global_step >= args.steps:
                break
            if use_cache:
                x0 = batch["x0"].to(device=device, dtype=torch.float32)
                prompt_emb = batch["prompt_emb"].to(device=device, dtype=torch.float32)
                pooled_emb = batch["pooled_emb"].to(device=device, dtype=torch.float32)
                text_ids = batch["text_ids"].to(device=device)
            else:
                images, prompts = _extract_images_and_prompts(batch)
                images = images.to(device=device, dtype=torch.float32)
                if isinstance(prompts, (list, tuple)):
                    prompts = list(prompts)
                else:
                    prompts = [str(prompts)]
                with torch.no_grad():
                    prompt_emb, pooled_emb, text_ids, _ = pipe.encode_prompt(
                        prompt=prompts,
                        device=device,
                        num_images_per_prompt=1,
                        max_sequence_length=args.max_sequence_length,
                    )
                    x0 = encode_images_to_latents(pipe, images).to(dtype=torch.float32)

            prompt_emb, pooled_emb, text_ids = _normalize_prompt_data(
                prompt_emb,
                pooled_emb,
                text_ids,
                device=device,
            )

            t = sample_timesteps(
                batch_size=x0.shape[0],
                num_steps=int(pipe.scheduler.config.num_train_timesteps),
                device=device,
            )

            residual_weights = F.softplus(residual_weights_raw)

            loss = compute_total_loss(
                pipe,
                x0,
                t,
                prompt_emb,
                pooled_emb,
                text_ids,
                img_size=args.img_size,
                residual_target_layers=residual_target_layers,
                residual_origin_layer=residual_origin_layer,
                residual_weights=residual_weights,
                residual_rotation_matrices=residual_rotation_matrices,
                residual_rotation_meta=residual_rotation_meta,
                guidance_scale=args.guidance_scale,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % args.log_every == 0:
                print(f"[Step {global_step}] loss={loss.item():.6f} weights={residual_weights.detach().cpu()}")

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_path = osp.join(args.output_dir, f"residual_weights_step{global_step}.pth")
                torch.save(
                    {
                        "residual_weights": residual_weights.detach().cpu(),
                        "residual_target_layers": residual_target_layers,
                        "residual_origin_layer": residual_origin_layer,
                    },
                    save_path,
                )
                print(f"[SAVE] {save_path}")

            global_step += 1
            pbar.update(1)
            if global_step >= args.steps:
                break

    final_path = osp.join(args.output_dir, "residual_weights_final.pth")
    torch.save(
        {
            "residual_weights": F.softplus(residual_weights_raw).detach().cpu(),
            "residual_target_layers": residual_target_layers,
            "residual_origin_layer": residual_origin_layer,
        },
        final_path,
    )
    print(f"[DONE] Saved final weights to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--precompute_dir", type=str, nargs="+", default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--datadir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_rotation_path", type=str, default=None)
    parser.add_argument("--residual_init", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
