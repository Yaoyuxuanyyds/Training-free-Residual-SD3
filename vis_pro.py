#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from diffusers.models.attention_processor import JointAttnProcessor2_0
from sampler import StableDiffusion3Base
from util import set_seed

# ============================================================
#  Cross-attention storage
# ============================================================

@dataclass
class AttentionRecord:
    text2img: torch.Tensor   # [B, textLen, imgLen]
    img2text: torch.Tensor   # [B, textLen, imgLen]
    image_token_count: int
    joint_attn: torch.Tensor  # [B, N, N], N = imgLen + textLen


class CrossAttentionStore:
    def __init__(self):
        self._records: Dict[int, AttentionRecord] = {}

    def add(
        self,
        layer_idx: int,
        text2img: torch.Tensor,
        img2text: torch.Tensor,
        image_token_count: int,
        joint_attn: torch.Tensor,
    ):
        self._records[layer_idx] = AttentionRecord(
            text2img.detach().cpu(),
            img2text.detach().cpu(),
            image_token_count,
            joint_attn.detach().cpu(),
        )

    def get(self, layer_idx: int) -> Optional[AttentionRecord]:
        return self._records.get(layer_idx, None)

    def clear(self):
        self._records.clear()


# ============================================================
#  Recorder wrapper
# ============================================================

class JointAttentionRecorder(JointAttnProcessor2_0):
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
        kwargs.pop("joint_attention_kwargs", None)

        bsz, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

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

            query = torch.cat([query, enc_q], dim=2)
            key   = torch.cat([key,   enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

            context_length = encoder_hidden_states.shape[1]

        scale = attn.scale
        if query.dtype != torch.float32:
            attn_scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        else:
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs.to(value.dtype)

        hidden_states_out = torch.matmul(attn_probs, value)
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

        if encoder_hidden_states is not None and context_length > 0 and self.store is not None:
            image_tokens = seq_len  # imgLen
            joint_attn = attn_probs.mean(dim=1)  # [B, N, N]

            # ---------- ✔ TEXT → IMAGE ----------
            # attn_probs[:, :, text, img]
            text2img_slice = attn_probs[:, :, image_tokens:, :image_tokens]
            text2img_mean = text2img_slice.mean(dim=1)  # [B, textLen, imgLen]

            # ---------- ✔ IMAGE → TEXT ----------
            # attn_probs[:, :, img, text]
            img2text_slice = attn_probs[:, :, :image_tokens, image_tokens:]
            img2text_mean = img2text_slice.mean(dim=1)  # [B, imgLen, textLen]

            # 转置到和 text2img 同维度：[B, textLen, imgLen]
            img2text_mean = img2text_mean.transpose(1, 2)

            self.store.add(
                self.layer_idx,
                text2img=text2img_mean,
                img2text=img2text_mean,
                image_token_count=image_tokens,
                joint_attn=joint_attn,
            )

        if encoder_output is not None:
            return hidden_states_out, encoder_output
        else:
            return hidden_states_out


def register_attention_recorders(denoiser_module, store: CrossAttentionStore, target_layers=None):
    base_model = getattr(denoiser_module, "base_model", denoiser_module)
    for idx, block in enumerate(base_model.transformer_blocks):
        if (target_layers is None) or (idx in target_layers):
            block.attn.set_processor(JointAttentionRecorder(store, idx))


# ============================================================
#  Viz utils
# ============================================================

def normalize_map(x: torch.Tensor):
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x.detach().cpu().numpy()


def upsample_to_imgres(attn_2d: torch.Tensor, H: int, W: int):
    hm = attn_2d[None, None].float()
    hm_up = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return normalize_map(hm_up)


def overlay_heatmap_on_image(base_img_pil: Image.Image, heatmap_2d: torch.Tensor, alpha=0.5, cmap="jet"):
    # PIL.Image.size = (W, H)
    W, H = base_img_pil.size
    hm_up = upsample_to_imgres(heatmap_2d, H, W)

    cm = plt.get_cmap(cmap)
    colored = cm(hm_up)[..., :3]    # [H,W,3], [0,1]
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



# ============================================================
#  Viz utils
# ============================================================

def plot_heatmap(matrix: np.ndarray, tokens: List[str], layers: List[int], title: str, path: str):
    fig, ax = plt.subplots(figsize=(2 + 1.2 * len(tokens), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="Reds")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# UPDATED visualize timestep
# ============================================================
def visualize_timestep(
    step_idx: int,
    decoded: torch.Tensor,
    store: CrossAttentionStore,
    t5_tokens: List[str],
    valid_token_idxs: List[int],
    token_offset: int,
    layer_ids: Sequence[int],
    token_words: Sequence[str],
    out_dir: str,
    cmap: str,
    alpha: float,
):
    """
    在单一 timestep 下生成：
      - decoded.png
      - 4 heatmaps:
          TEXT→IMAGE raw
          TEXT→IMAGE softmax
          IMAGE→TEXT raw
          IMAGE→TEXT softmax
      - 每个 token 的 overlay grid (原逻辑)
    """
    step_dir = os.path.join(out_dir, f"t{step_idx:04d}")
    os.makedirs(step_dir, exist_ok=True)

    # ------------------------------
    # 1) 保存当前解码图像
    # ------------------------------
    decoded_path = os.path.join(step_dir, "decoded.png")
    save_image(decoded, decoded_path, normalize=True)
    print(f"[SAVE] {decoded_path}")

    # base PIL for overlay grid
    img_tensor = decoded[0].detach().float().cpu()
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    img_np = (img_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))
    base_img_pil = Image.fromarray(img_np, mode="RGB")

    # ------------------------------
    # 2) 匹配 token_words → token indices
    # ------------------------------
    selected_token_indices = []
    selected_token_strings = []

    for w in token_words:
        matches = [tok_idx for tok_idx in valid_token_idxs if w in t5_tokens[tok_idx]]
        if not matches:
            print(f"[WARN] '{w}' not found in valid tokens")
            continue
        selected_token_indices.append(matches[0])
        selected_token_strings.append(t5_tokens[matches[0]])

    if len(selected_token_indices) == 0:
        print("[WARN] no selected tokens, abort heatmaps.")
        return

    # ------------------------------
    # 3) 收集跨层 TEXT→IMAGE & IMAGE→TEXT
    # ------------------------------
    L_text2img = []   # raw sum per layer
    L_img2text = []
    layer_list = []

    for lid in layer_ids:
        rec = store.get(lid)
        if rec is None:
            continue

        # shape: [1, textLen, imgLen]
        t2i = rec.text2img[0]
        i2t = rec.img2text[0]

        # sum attention over image tokens
        # TEXT→IMAGE: 每 token 把多少注意力给图
        s_text2img = t2i.sum(dim=1)[selected_token_indices]  # [K]

        # IMAGE→TEXT: 每 token 从图收到多少注意力
        s_img2text = i2t.sum(dim=1)[selected_token_indices]  # [K]

        L_text2img.append(s_text2img.unsqueeze(0))
        L_img2text.append(s_img2text.unsqueeze(0))
        layer_list.append(lid)

    if len(L_text2img) == 0:
        print("[WARN] no valid attention records for selected layers")
        return

    # ---------- STACK ----------
    M_text2img_raw = torch.cat(L_text2img, dim=0)     # [L, K]
    M_text2img_soft = torch.softmax(M_text2img_raw, dim=1)

    M_img2text_raw = torch.cat(L_img2text, dim=0)
    M_img2text_soft = torch.softmax(M_img2text_raw, dim=1)

    # ------------------------------
    # 4) 绘制四张 heatmap
    # ------------------------------

    def _plot(M: torch.Tensor, title: str, fname: str, fmt="%.4f"):
        """
        绘制热力图 + 在每个格子标注对应数值
        M: [L, K]
        """
        M_np = M.detach().cpu().numpy()
        L, K = M_np.shape

        fig, ax = plt.subplots(figsize=(2 + 1.2 * K, 6))
        # im = ax.imshow(M_np, aspect="auto", cmap="Reds")
        im = ax.imshow(
            M_np,
            aspect="auto",
            cmap="Reds",
            vmin=0,
            vmax=10
        )
        plt.colorbar(im, ax=ax)

        ax.set_title(title)
        ax.set_xticks(range(K))
        ax.set_xticklabels(selected_token_strings, rotation=90, fontsize=8)
        ax.set_yticks(range(L))
        ax.set_yticklabels([str(l) for l in layer_list])

        # -------- ★ 在每个格子写数值 ★ --------
        for i in range(L):           # row: layer index
            for j in range(K):       # col: token index
                val = M_np[i, j]
                # 自动选择字体颜色（深色背景用白字）
                # text_color = "white" if val > M_np.mean() else "black"
                text_color = "white" if val > 5 else "black"
                ax.text(
                    j, i,
                    fmt % val,
                    ha="center", va="center",
                    color=text_color,
                    fontsize=8
                )

        plt.tight_layout()
        path = os.path.join(step_dir, fname)
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {path}")


    # TEXT→IMAGE
    _plot(M_text2img_raw, "TEXT→IMAGE Raw Sum", "heat_text2img_raw.png")
    _plot(M_text2img_soft, "TEXT→IMAGE Softmax", "heat_text2img_softmax.png")

    # IMAGE→TEXT
    _plot(M_img2text_raw, "IMAGE→TEXT Raw Sum", "heat_img2text_raw.png")
    _plot(M_img2text_soft, "IMAGE→TEXT Softmax", "heat_img2text_softmax.png")

    # ------------------------------
    # 5) 绘制 joint attention 全图 (N x N)
    # ------------------------------
    for lid in layer_ids:
        rec = store.get(lid)
        if rec is None:
            continue

        joint_attn = rec.joint_attn[0].detach().cpu().numpy()
        n_tokens = joint_attn.shape[0]

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(joint_attn, aspect="equal", cmap="Reds")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Joint Attention (Layer {lid}, N={n_tokens})")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        plt.tight_layout()

        path = os.path.join(step_dir, f"joint_attn_layer-{lid}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {path}")

    # =====================================================================
    # 6) 原有 overlay grid: 图上标注 token 被关注的空间分布 (保持原逻辑)
    # =====================================================================
    for word in token_words:
        matches = [tok_idx for tok_idx in valid_token_idxs if word in t5_tokens[tok_idx]]
        if not matches:
            continue

        tok_idx = matches[0]
        tok_str = t5_tokens[tok_idx]
        layer_to_map = {}

        for lid in layer_ids:
            rec = store.get(lid)
            if rec is None:
                continue

            ctx_index = token_offset + tok_idx
            if ctx_index >= rec.text2img.shape[2]:
                continue

            # NOTE: overlay uses TEXT→IMAGE spatial map
            token_map = rec.text2img[0, tok_idx]  # [imgLen]

            img_len = rec.image_token_count
            side = int(math.sqrt(img_len))
            if side * side != img_len:
                raise RuntimeError("image_token_count not square")

            token_map_2d = token_map.view(side, side)
            layer_to_map[lid] = token_map_2d

        if not layer_to_map:
            continue

        grid_path = os.path.join(step_dir, f"grid_token-{sanitize_token(tok_str)}.png")
        save_grid_for_token(
            token_str=tok_str,
            layer_list=[lid for lid in layer_ids if lid in layer_to_map],
            layer_to_map=layer_to_map,
            base_img_pil=base_img_pil,
            out_path=grid_path,
            cmap=cmap,
            alpha=alpha,
        )
        print(f"[SAVE] {grid_path}")


# ============================================================
#  Runtime SD3 visualization
# ============================================================

def run_sd3_runtime_vis(
    model_path: str,
    prompt: str,
    out_dir: str,
    width: int,
    height: int,
    dump_timesteps: List[int],
    cfg_scale: float,
    NFE: int,
    device: str,
    token_words: Sequence[str],
    layer_ids: Sequence[int],
    cmap: str,
    alpha: float,
    residual_origin_layer: Optional[int],
    residual_target_layers: Optional[List[int]],
    residual_weights: Optional[List[float]],
):
    os.makedirs(out_dir, exist_ok=True)

    # 1. 初始化底模
    base = StableDiffusion3Base(
        model_key=model_path,
        device=device,
        dtype=torch.float16 if "cuda" in device else torch.float32,
        use_8bit=False,
        load_ckpt_path=None,
        load_transformer_only=False,
    )

    scheduler = base.scheduler

    # 2. prompt & null prompt embedding
    prompt_emb, pooled_emb, _ = base.encode_prompt([prompt], batch_size=1)
    null_emb, null_pooled_emb, _ = base.encode_prompt([""], batch_size=1)

    # 3. 初始化 latent
    z = base.initialize_latent((height, width), batch_size=1)

    # 4. T5 tokens（用于找文字）
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
    valid_token_idxs = [i for i, m in enumerate(t5_mask) if m > 0]

    t5_len = t5_inputs.input_ids.shape[1]
    total_context = prompt_emb.shape[1]
    token_offset = total_context - t5_len

    # 5. 注册 attention recorder
    store = CrossAttentionStore()
    register_attention_recorders(base.denoiser, store, target_layers=layer_ids)

    # 6. Euler 采样循环
    scheduler.set_timesteps(NFE, device=device)
    timesteps = scheduler.timesteps
    steps = timesteps / scheduler.config.num_train_timesteps

    for i, t in enumerate(timesteps):
        t_tensor = t.expand(z.shape[0]).to(device)

        # 每个时间步之前清空 store
        store.clear()

        # 预测 pred_v（带 residual）
        pred_v = base.predict_vector_residual(
            z, t_tensor, prompt_emb, pooled_emb,
            residual_target_layers=residual_target_layers,
            residual_origin_layer=residual_origin_layer,
            residual_weights=residual_weights,
        )

        # null prompt 分支
        if cfg_scale != 1.0:
            pred_null_v = base.predict_vector_residual(
                z, t_tensor, null_emb, null_pooled_emb,
                residual_target_layers=residual_target_layers,
                residual_origin_layer=residual_origin_layer,
                residual_weights=residual_weights,
            )
        else:
            pred_null_v = 0.0

        step = steps[i]
        step_next = steps[i + 1] if i + 1 < NFE else 0.0
        z = z + (step_next - step) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        # 如果这个 timestep 在 dump 列表中，就可视化
        if i in dump_timesteps:
            decoded = base.decode(z)
            visualize_timestep(
                step_idx=i,
                decoded=decoded,
                store=store,
                t5_tokens=t5_tokens,
                valid_token_idxs=valid_token_idxs,
                token_offset=token_offset,
                layer_ids=layer_ids,
                token_words=token_words,
                out_dir=out_dir,
                cmap=cmap,
                alpha=alpha,
            )

    print(f"[DONE] All visualizations saved under: {out_dir}")


# ============================================================
#  CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Runtime visualization of SD3 cross-attention & img2text heatmaps over timesteps."
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Path / repo id for SD3 base model.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to generate image.")
    parser.add_argument("--output", type=str, required=True,
                        help="Root output directory.")

    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)

    parser.add_argument("--dump-timesteps", type=int, nargs="+", required=True,
                        help="Which timestep indices (0..NFE-1) to dump visualizations for.")
    parser.add_argument("--NFE", type=int, default=28)
    parser.add_argument("--cfg-scale", type=float, default=7.0)

    parser.add_argument("--token-words", type=str, nargs="+", required=True,
                        help="Words (substrings) to match T5 tokens.")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Transformer layer indices to hook.")

    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    run_sd3_runtime_vis(
        model_path=args.model,
        prompt=args.prompt,
        out_dir=args.output,
        width=args.width,
        height=args.height,
        dump_timesteps=args.dump_timesteps,
        cfg_scale=args.cfg_scale,
        NFE=args.NFE,
        device=device,
        token_words=args.token_words,
        layer_ids=args.layers,
        cmap=args.cmap,
        alpha=args.alpha,
        residual_origin_layer=args.residual_origin_layer,
        residual_target_layers=args.residual_target_layers,
        residual_weights=args.residual_weights,
    )


if __name__ == "__main__":
    main()









