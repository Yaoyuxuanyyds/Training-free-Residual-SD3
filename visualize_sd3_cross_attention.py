#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from diffusers.models.attention_processor import JointAttnProcessor2_0

from sampler import StableDiffusion3Base  # must match your local sampler.py


# ========= Cross-attention capture =========

@dataclass
class AttentionRecord:
    # txt2img: [B, contextLen, imageLen], already averaged across heads
    cross_txt2img: torch.Tensor
    # img2txt: [B, imageLen, contextLen], already averaged across heads
    cross_img2txt: torch.Tensor


class CrossAttentionStore:
    """
    Maps layer_idx -> AttentionRecord
    Stores image_token_count = H_lat * W_lat
    """
    def __init__(self):
        self._records: Dict[int, AttentionRecord] = {}
        self.image_token_count: Optional[int] = None

    def add(
        self,
        layer_idx: int,
        cross_txt2img: torch.Tensor,
        cross_img2txt: torch.Tensor,
        image_token_count: int,
    ):
        if self.image_token_count is None:
            self.image_token_count = image_token_count
        self._records[layer_idx] = AttentionRecord(
            cross_txt2img=cross_txt2img.detach().cpu(),
            cross_img2txt=cross_img2txt.detach().cpu(),
        )

    def layers(self) -> Sequence[int]:
        return sorted(self._records.keys())

    def get(self, layer_idx: int) -> Optional[AttentionRecord]:
        return self._records.get(layer_idx, None)


class JointAttentionRecorder(JointAttnProcessor2_0):
    """
    Wrap JointAttnProcessor2_0 to:
    - Perform the same attention math
    - Record
        * text->image cross attention (txt2img)
        * image->text cross attention (img2txt)
    - Average across heads (multi-head mean)
    """

    def __init__(self, store: CrossAttentionStore, layer_idx: int):
        super().__init__()
        self.store = store
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ):
        # 兼容新版 diffusers，忽略 joint_attention_kwargs
        kwargs.pop("joint_attention_kwargs", None)

        bsz, seq_len, _ = hidden_states.shape  # seq_len = #image_tokens

        # project q/k/v for image tokens
        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        # reshape for heads
        query = query.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key   = attn.norm_k(key)

        context_length = 0
        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            # concat image tokens + context tokens along sequence dim
            query = torch.cat([query, enc_q], dim=2)
            key   = torch.cat([key,   enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

            context_length = encoder_hidden_states.shape[1]  # = contextLen

        scale = attn.scale
        if query.dtype != torch.float32:
            attn_scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        else:
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        # attn_scores: [B,H,jointLen,jointLen]

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B,H,jointLen,jointLen]
        attn_probs = attn_probs.to(value.dtype)

        # standard output path
        hidden_states_out = torch.matmul(attn_probs, value)  # [B,H,jointLen,headDim]
        hidden_states_out = hidden_states_out.transpose(1, 2).reshape(
            bsz, -1, attn.heads * head_dim
        )
        hidden_states_out = hidden_states_out.to(query.dtype)

        encoder_output = None
        if encoder_hidden_states is not None:
            hidden_states_out, encoder_output = (
                hidden_states_out[:, :seq_len],
                hidden_states_out[:, seq_len:],
            )
            if not attn.context_pre_only:
                encoder_output = attn.to_add_out(encoder_output)

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        # ------------- record cross-attention -------------
        if encoder_hidden_states is not None and context_length > 0 and self.store is not None:
            image_tokens = seq_len
            # txt2img: text queries (context) -> image keys
            # shape: [B,H,contextLen,imgLen]
            txt2img_slice = attn_probs[:, :, image_tokens: image_tokens + context_length, :image_tokens]
            txt2img_mean = txt2img_slice.mean(dim=1)  # [B,contextLen,imgLen]

            # img2txt: image queries -> text/context keys
            # shape: [B,H,imgLen,contextLen]
            img2txt_slice = attn_probs[:, :, :image_tokens, image_tokens: image_tokens + context_length]
            img2txt_mean = img2txt_slice.mean(dim=1)  # [B,imgLen,contextLen]

            self.store.add(
                self.layer_idx,
                cross_txt2img=txt2img_mean,
                cross_img2txt=img2txt_mean,
                image_token_count=image_tokens,
            )

        if encoder_output is not None:
            return hidden_states_out, encoder_output
        else:
            return hidden_states_out


def register_attention_recorders(denoiser_module, store: CrossAttentionStore, target_layers=None):
    """
    Inject JointAttentionRecorder into specified transformer blocks.
    denoiser_module can be SD3Transformer2DModel_REPA or wrapper with .base_model.
    """
    base_model = getattr(denoiser_module, "base_model", denoiser_module)
    print(f"Total layers : {len(base_model.transformer_blocks)}")
    for idx, block in enumerate(base_model.transformer_blocks):
        if (target_layers is None) or (idx in target_layers):
            block.attn.set_processor(JointAttentionRecorder(store, idx))


# ========= Image / latent utilities =========

def load_and_resize_pil(path: str, height: int, width: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.BICUBIC)
    return img


def pil_to_tensor(pil_img: Image.Image, device: str):
    import torchvision.transforms as T
    t = T.ToTensor()  # [0,1], shape [C,H,W]
    x = t(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]
    return x


def encode_image_to_latent(base: StableDiffusion3Base, img_tensor: torch.Tensor):
    """
    Encode GT image into SD3 latent space.
    We'll mirror StableDiffusion3Base.decode(), but inverted.
    Also ensure dtype matches VAE weights (to handle fp16 VAE).
    """
    vae = base.vae
    scaling = vae.config.scaling_factor
    shift   = vae.config.shift_factor
    with torch.no_grad():
        img_tensor = img_tensor.to(dtype=vae.dtype)              # match fp16/fp32
        posterior = vae.encode(img_tensor * 2 - 1)               # map [0,1]->[-1,1]
        latent_pre = posterior.latent_dist.sample()              # [B,C,h,w]
        z0 = (latent_pre - shift) * scaling                      # training latent space
    return z0  # [B,C,h,w]


def build_noisy_latent_like_training(
    scheduler,
    clean_latent: torch.Tensor,
    timestep_idx: int,
):
    """
    Reproduce the exact procedure in compute_total_loss():
        T = scheduler.config.num_train_timesteps
        s = t / T
        x1 = randn_like(x0)
        x_s = (1-s)*x0 + s*x1
    We'll return x_s (the noisy latent), plus t_tensor and also (for logging) t_idx.
    """
    device = clean_latent.device
    B = clean_latent.shape[0]

    T = int(scheduler.config.num_train_timesteps)
    t_tensor = torch.full((B,), timestep_idx, device=device, dtype=torch.long)

    s = (t_tensor.float() / float(T)).view(B, 1, 1, 1)

    x1 = torch.randn_like(clean_latent)
    x_s = (1.0 - s) * clean_latent + s * x1

    return x_s, t_tensor, timestep_idx


# ========= Attention visualization utilities =========

def normalize_map(x: torch.Tensor):
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x.detach().cpu().numpy()


def upsample_to_imgres(attn_2d: torch.Tensor, out_h: int, out_w: int):
    """
    attn_2d: [h,w] -> resize -> [out_h,out_w] -> numpy[H,W] in [0,1]
    """
    hm = attn_2d[None, None, :, :].float()
    hm_up = F.interpolate(hm, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
    return normalize_map(hm_up)


def overlay_heatmap_on_image(base_img_pil: Image.Image, heatmap_2d: torch.Tensor, alpha=0.5, cmap="jet"):
    """
    base_img_pil: PIL Image
    heatmap_2d: attention map in latent space [h,w] (torch)
    -> returns PIL.Image with overlay
    """
    W, H = base_img_pil.size
    hm_up = upsample_to_imgres(heatmap_2d, H, W)  # shape [H,W], float in [0,1]

    cm = plt.get_cmap(cmap)
    colored = cm(hm_up)[..., :3]    # [H,W,3] in [0,1]
    colored_img = (colored * 255).astype(np.uint8)
    heat_pil = Image.fromarray(colored_img, mode="RGB")

    blended = Image.blend(base_img_pil.convert("RGB"), heat_pil, alpha=alpha)
    return blended


def sanitize_token(token: str) -> str:
    token = token.replace("</s>", "eos")
    token = token.replace("<pad>", "pad")
    token = token.replace("<unk>", "unk")
    token = token.replace(" ", "_")
    token = token.replace("/", "-")
    token = token.replace("\\", "-")
    token = token.replace(":", "-")
    token = token.replace("|", "-")
    return token


def save_grid_for_token(
    token_str: str,
    layer_list: Sequence[int],
    layer_to_map: Dict[int, torch.Tensor],
    base_img_pil: Image.Image,
    out_path: str,
    cmap="jet",
    alpha=0.5,
):
    """
    Make a subplot grid where each subplot shows that token's attention overlay
    at a different layer.
    """
    cols = math.ceil(math.sqrt(len(layer_list)))
    rows = math.ceil(len(layer_list) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, layer_id in enumerate(layer_list):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        attn_map_2d = layer_to_map[layer_id]
        blended = overlay_heatmap_on_image(base_img_pil, attn_map_2d, alpha=alpha, cmap=cmap)

        ax.imshow(blended)
        ax.axis("off")
        ax.set_title(f"Layer {layer_id}")

    # hide unused subplots
    for j in range(len(layer_list), rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis("off")

    fig.suptitle(f"Token: {token_str}", fontsize=16)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ========= Main pipeline =========

def run_single_step_and_visualize(
    model_path: str,
    prompt: str,
    gt_image_path: str,
    device: str,
    height: int,
    width: int,
    timestep_idx: int,
    layer_ids: Sequence[int],
    token_words: Sequence[str],
    max_t5_tokens: int,
    out_dir: str,
    alpha: float,
    cmap: str,
    seed: int,
    residual_target_layers: Optional[List[int]] = None,
    residual_origin_layer: Optional[int] = None,
    residual_weights: Optional[List[float]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    # choose precision
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    # 1. init SD3 base model
    base = StableDiffusion3Base(
        model_key=model_path,
        device=device,
        dtype=dtype,
        use_8bit=False,
        load_ckpt_path=None,
        load_transformer_only=False
    )
    denoiser = base.denoiser
    denoiser.eval()
    denoiser.requires_grad_(False)

    # 2. load & resize GT image
    gt_pil = load_and_resize_pil(gt_image_path, height, width)
    gt_tensor = pil_to_tensor(gt_pil, device=device)  # [1,3,H,W] in [0,1]

    # 3. encode to latent z0
    z0 = encode_image_to_latent(base, gt_tensor)  # [1,C,h_lat,w_lat]

    # 4. build "noisy" latent for given timestep_idx using training's rule
    z_t, t_tensor, t_rawval = build_noisy_latent_like_training(
        scheduler=base.scheduler,
        clean_latent=z0,
        timestep_idx=timestep_idx,
    )

    # 5. encode prompt => prompt_emb (context embeddings), pooled_emb, token_mask
    with torch.no_grad():
        prompt_emb, pooled_emb, _ = base.encode_prompt([prompt], batch_size=1)

    # 6. get T5 tokens for labeling
    t5_inputs = base.tokenizer_3(
        [prompt],
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    t5_tokens = base.tokenizer_3.convert_ids_to_tokens(t5_inputs.input_ids[0])
    t5_mask = t5_inputs.attention_mask[0].tolist()

    valid_token_idxs = [i for i, m in enumerate(t5_mask) if m > 0][:max_t5_tokens]

    # offset: prompt_emb is [CLIP..., T5...]
    t5_len = t5_inputs.input_ids.shape[1]       # usually 256
    total_context = prompt_emb.shape[1]         # clip_len + t5_len
    token_offset = total_context - t5_len       # = clip_len

    # 选出“指定的文本 tokens”，后面 overlay 和 heatmap 都用这个列表
    selected_tokens = []  # list of dicts: {word, tok_idx, tok_str}
    for word in token_words:
        matches = [
            tok_idx for tok_idx in valid_token_idxs
            if word in t5_tokens[tok_idx]
        ]
        if not matches:
            print(f"[WARN] word '{word}' not found in first {len(valid_token_idxs)} valid T5 tokens")
            continue
        tok_idx = matches[0]
        tok_str = t5_tokens[tok_idx]
        selected_tokens.append(
            {"word": word, "tok_idx": tok_idx, "tok_str": tok_str}
        )

    if not selected_tokens:
        print("[WARN] No selected tokens found, nothing to visualize.")
        return

    # 7. register attention recorders for chosen layers
    store = CrossAttentionStore()
    register_attention_recorders(denoiser, store, target_layers=layer_ids)

    # 8. forward once through denoiser
    with torch.no_grad():
        z_t         = z_t.to(dtype=denoiser.dtype)
        prompt_emb  = prompt_emb.to(dtype=denoiser.dtype)
        pooled_emb  = pooled_emb.to(dtype=denoiser.dtype)

        _ = denoiser(
            z_t,
            timestep=t_tensor,
            encoder_hidden_states=prompt_emb,
            pooled_projections=pooled_emb,
            return_dict=False,
            residual_target_layers=residual_target_layers,
            residual_origin_layer=residual_origin_layer,
            residual_weights=residual_weights,
        )

    # after forward: store has head-averaged cross-attn per layer

    # 9. 原有：对每个指定 token，画 txt→img 的 overlay
    for sel in selected_tokens:
        tok_idx = sel["tok_idx"]
        tok_str = sel["tok_str"]

        layer_to_map: Dict[int, torch.Tensor] = {}

        for layer_id in layer_ids:
            rec = store.get(layer_id)
            if rec is None:
                print(f"[WARN] no attention recorded for layer {layer_id}")
                continue

            # rec.cross_txt2img: [B,contextLen,imageLen], B=1
            ctx_index = token_offset + tok_idx
            if ctx_index >= rec.cross_txt2img.shape[1]:
                print(f"[WARN] ctx_index {ctx_index} out of bounds for layer {layer_id}")
                continue

            token_map = rec.cross_txt2img[0, ctx_index]  # [imageLen]
            image_token_count = store.image_token_count
            grid_side = int(math.sqrt(image_token_count))
            if grid_side * grid_side != image_token_count:
                raise RuntimeError(
                    f"image_token_count={image_token_count} not square; cannot reshape"
                )
            token_map_2d = token_map.view(grid_side, grid_side)

            layer_to_map[layer_id] = token_map_2d

        # save grid comparison across layers
        if len(layer_to_map) > 0:
            grid_path = os.path.join(
                out_dir,
                f"grid_token-{sanitize_token(tok_str)}_t{t_rawval}.png"
            )

            save_grid_for_token(
                token_str=tok_str,
                layer_list=[lid for lid in layer_ids if lid in layer_to_map],
                layer_to_map=layer_to_map,
                base_img_pil=gt_pil,
                out_path=grid_path,
                cmap=cmap,
                alpha=alpha,
            )
            print(f"[SAVE] {grid_path}")

    # 10. 新增：img→text 总注意力热力图
    # 每一行：一层；每一列：一个 selected text token
    layers_for_heatmap = [lid for lid in layer_ids if store.get(lid) is not None]
    n_layers = len(layers_for_heatmap)
    n_tokens = len(selected_tokens)

    if n_layers > 0 and n_tokens > 0:
        heat = np.zeros((n_layers, n_tokens), dtype=np.float32)

        for i, layer_id in enumerate(layers_for_heatmap):
            rec = store.get(layer_id)
            # rec.cross_img2txt: [B,imgLen,contextLen]
            img2txt = rec.cross_img2txt[0]  # [imgLen,contextLen]
        
            # 对每个 text token，汇聚从所有 image tokens 来的注意力
            # sum over imgLen dimension
            summed = img2txt.sum(dim=0)  # [contextLen]

            for j, sel in enumerate(selected_tokens):
                ctx_index = token_offset + sel["tok_idx"]
                if ctx_index >= summed.shape[0]:
                    continue
                heat[i, j] = float(summed[ctx_index].item())

        # 为了可视化更稳定，对每一行做归一化（防止层间量级差太大）
        row_max = heat.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        heat_norm = heat / row_max

        fig, ax = plt.subplots(
            figsize=(2 + n_tokens * 0.6, 2 + n_layers * 0.4)
        )
        im = ax.imshow(
            heat_norm,
            aspect="auto",
            cmap=cmap,
            origin="upper",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("Relative img→text attention (per layer)", fontsize=10)

        # x 轴：文本 token（列）
        col_labels = [sanitize_token(sel["tok_str"]) for sel in selected_tokens]
        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=8)

        # y 轴：layer index（行）
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([str(l) for l in layers_for_heatmap], fontsize=8)

        ax.set_xlabel("Text tokens", fontsize=10)
        ax.set_ylabel("Layer index", fontsize=10)
        ax.set_title("Total img→text attention per layer & token", fontsize=12)

        fig.tight_layout()
        heat_path = os.path.join(out_dir, f"img2text_token_heatmap_t{t_rawval}.png")
        fig.savefig(heat_path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {heat_path}")


# ========= CLI =========

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SD3 cross-attention maps for selected tokens across layers, "
                    "conditioning on a real GT image at a chosen diffusion timestep."
    )

    parser.add_argument("--model", type=str, default="base_models/diffusion/sd3",
                        help="Path / repo id for StableDiffusion3Base-compatible SD3 weights.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing the GT image.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to GT image (.png/.jpg).")

    parser.add_argument("--output", type=str, default="attn_vis_out",
                        help="Directory to save visualizations.")
    parser.add_argument("--height", type=int, default=1024,
                        help="Resize GT image height.")
    parser.add_argument("--width", type=int, default=1024,
                        help="Resize GT image width.")

    parser.add_argument("--timestep-idx", type=int, default=500,
                        help="Integer timestep index t in [0, T) to visualize. "
                             "We reproduce x_s = (1-s)*x0 + s*x1 with s=t/T.")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Transformer layer indices to hook and visualize, e.g. --layers 5 8 12 16")
    parser.add_argument("--token-words", type=str, nargs="+", required=True,
                        help="List of substrings to match T5 sub-tokens, e.g. --token-words cat mountain sunrise")
    parser.add_argument("--max-t5-tokens", type=int, default=256,
                        help="Only consider first N valid T5 tokens when matching words.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha for blending heatmap on GT image.")
    parser.add_argument("--cmap", type=str, default="jet",
                        help="Matplotlib colormap for heatmap coloring.")

    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (auto if None).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Noise seed for constructing x_s at timestep t.")

    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_single_step_and_visualize(
        model_path=args.model,
        prompt=args.prompt,
        gt_image_path=args.image,
        device=device,
        height=args.height,
        width=args.width,
        timestep_idx=args.timestep_idx,
        layer_ids=args.layers,
        token_words=args.token_words,
        max_t5_tokens=args.max_t5_tokens,
        out_dir=args.output,
        alpha=args.alpha,
        cmap=args.cmap,
        seed=args.seed,
        residual_target_layers=args.residual_target_layers,
        residual_origin_layer=args.residual_origin_layer,
        residual_weights=args.residual_weights,
    )

    print(f"[DONE] saved visualizations to {args.output}")


if __name__ == "__main__":
    main()
