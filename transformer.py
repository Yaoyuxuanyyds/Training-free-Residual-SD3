from typing import Any, Dict, List, Optional, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from dataclasses import dataclass
import time
import torch
import torch.nn as nn

    
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.utils import logging
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)


def compute_residual_flops(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_targets: int,
    use_layernorm: bool = True,
    use_rotation: bool = False,
    assume_positive_weight: bool = True,
) -> Dict[str, int]:
    """
    Compute FLOPs for residual-only computation in one forward pass.

    Counting rule: each add/sub/mul/div/sqrt counts as 1 FLOP.
    The counts below are per residual application and then scaled by
    batch_size * seq_len * num_targets.

    Standardize per token (mean + std + normalize) costs:
      mean: (D-1) adds + 1 div
      variance: D subs + D muls + (D-1) adds + 1 div + 1 sqrt + 1 add (eps)
      normalize: D subs + D divs
      total adds/subs: (D-1) + (D-1) + D + D + 1 = 4D - 1
      total muls/divs/sqrt: D (mul) + (1 div) + (1 div) + D (div) + 1 sqrt = D + D + 2 div + 1 sqrt
      total FLOPs: 6D + 2 (div) + 1 (sqrt) - 1
    """
    d = hidden_dim

    def _standardize_flops() -> int:
        adds_subs = 4 * d - 1
        muls = d
        divs = d + 2
        sqrt_ops = 1
        return adds_subs + muls + divs + sqrt_ops

    def _layernorm_flops() -> int:
        # layer_norm is equivalent to standardize (no affine here)
        return _standardize_flops()

    def _rotation_flops() -> int:
        # For each token: D outputs, each D muls + (D-1) adds
        return d * (2 * d - 1)

    def _mix_flops() -> int:
        if assume_positive_weight:
            # t_norm + w * o_norm
            return 2 * d
        # t_norm * (1 - w): 1 sub + D mul
        return d + 1

    flops_per_token = 0
    if use_layernorm:
        flops_per_token += 2 * _standardize_flops()  # target + origin
        if use_rotation:
            flops_per_token += _rotation_flops()
        flops_per_token += _mix_flops()
        flops_per_token += _layernorm_flops()
        # restore original scale: mixed * std + mean
        flops_per_token += 2 * d
    else:
        # target + w * origin
        flops_per_token += 2 * d

    total = batch_size * seq_len * num_targets * flops_per_token
    return {
        "flops_per_token": flops_per_token,
        "total_flops": total,
    }


def compute_sd3_block_flops(
    batch_size: int,
    img_seq_len: int,
    txt_seq_len: int,
    hidden_dim: int,
    mlp_ratio: int = 4,
    include_softmax: bool = True,
    include_layernorm: bool = True,
) -> Dict[str, int]:
    """
    Compute FLOPs for one SD3 transformer block forward pass.

    Assumptions (explicit for strict accounting):
      - Joint attention over concatenated tokens (image + text).
      - Q/K/V projections and output projection are dense (D -> D).
      - MLP is two linear layers with width = mlp_ratio * D (no gating).
      - FLOP rule: each add/sub/mul/div/sqrt = 1 FLOP.
      - Softmax cost approximated as 5 FLOPs per element (exp + sum + div).
      - LayerNorm cost uses the same standardize FLOP count as residual helper.
    """
    d = hidden_dim
    s_img = img_seq_len
    s_txt = txt_seq_len
    s_total = s_img + s_txt

    def _standardize_flops() -> int:
        adds_subs = 4 * d - 1
        muls = d
        divs = d + 2
        sqrt_ops = 1
        return adds_subs + muls + divs + sqrt_ops

    # Q, K, V projections: 3 * (2 * S * D * D)
    qkv_flops = 3 * (2 * s_total * d * d)
    # Attention scores: Q @ K^T: 2 * S * S * D
    attn_score_flops = 2 * s_total * s_total * d
    # Softmax: 5 FLOPs per score
    softmax_flops = 5 * s_total * s_total if include_softmax else 0
    # Attention-weighted values: 2 * S * S * D
    attn_weight_flops = 2 * s_total * s_total * d
    # Output projection: 2 * S * D * D
    out_proj_flops = 2 * s_total * d * d

    # MLP: two linear layers D -> rD -> D
    mlp_hidden = mlp_ratio * d
    mlp_flops = 2 * s_total * d * mlp_hidden + 2 * s_total * mlp_hidden * d

    # LayerNorm: assume two LNs per block over all tokens
    ln_flops = 2 * s_total * _standardize_flops() if include_layernorm else 0

    total_per_batch = (
        qkv_flops
        + attn_score_flops
        + softmax_flops
        + attn_weight_flops
        + out_proj_flops
        + mlp_flops
        + ln_flops
    )

    total = batch_size * total_per_batch
    return {
        "total_flops": total,
        "qkv_flops": batch_size * qkv_flops,
        "attn_score_flops": batch_size * attn_score_flops,
        "softmax_flops": batch_size * softmax_flops,
        "attn_weight_flops": batch_size * attn_weight_flops,
        "out_proj_flops": batch_size * out_proj_flops,
        "mlp_flops": batch_size * mlp_flops,
        "layernorm_flops": batch_size * ln_flops,
    }



class SD3Transformer2DModel_Residual(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.dtype = base_model.dtype

    def to(self, *args, **kwargs):
        self.base_model = self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ===============================================================
    #  token-wise 标准化
    # ===============================================================
    @staticmethod
    def _standardize_tokenwise(x, eps=1e-6):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std, mean, std

    # ===============================================================
    #  改进 residual（支持 stopgrad + 可选 LN）
    # ===============================================================
    def _apply_residual(
        self,
        target: torch.Tensor,
        origin: torch.Tensor,
        w: torch.Tensor,
        use_layernorm: bool = True,
        stop_grad: bool = True,
        rotation_matrix: Optional[torch.Tensor] = None,
    ):
        """
        target/origin: [B, L, D]
        w: scalar tensor
        """

        # ----------- STOP GRADIENT PART -----------
        # residual 的 2 个输入都不参与梯度
        if stop_grad:
            target_nograd = target.detach()
            origin_nograd = origin.detach()
        else:
            target_nograd = target
            origin_nograd = origin

        if use_layernorm:
            # --- standardize ---
            t_norm, t_mean, t_std = self._standardize_tokenwise(target_nograd)
            o_norm, _, _ = self._standardize_tokenwise(origin_nograd)
            if rotation_matrix is not None:
                o_norm = torch.matmul(o_norm, rotation_matrix)

            # --- residual rule ---
            if w >= 0:
                mixed = t_norm + w * o_norm
            else:
                mixed = t_norm * (1 - w)

            # --- restandardize ---
            mixed = torch.nn.functional.layer_norm(
                mixed, normalized_shape=mixed.shape[-1:], eps=1e-6
            )

            # --- restore original scale ---
            return mixed * t_std + t_mean
        else:
            return target_nograd + w * origin_nograd


    # ===============================================================
    #  forward
    # ===============================================================
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        output_hidden_states: bool = False,
        output_text_inputs: bool = False,
        force_txt_grad: bool = False,
        residual_stop_grad: bool = False,

        # --- residual 参数 ---
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weights: Optional[Union[List[float], torch.Tensor]] = None,
        residual_use_layernorm: bool = True,         # ⭐ 新增
        residual_rotation_matrices: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        profile_time: bool = False,
        profile_time_sync: bool = True,
        profile_time_path: Optional[str] = None,
        profile_time_run_id: Optional[str] = None,
        profile_time_append: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        output_hidden_states = True
        if profile_time:
            def _write_timings(path: str, total_time: float):
                mode = "a" if profile_time_append else "w"
                run_id = profile_time_run_id or time.strftime("%Y%m%d-%H%M%S")
                with open(path, mode, encoding="utf-8") as handle:
                    handle.write(f"run_id={run_id}\n")
                    handle.write(f"total: {total_time:.6f}s\n")
                    handle.write("\n")

            def _sync():
                if profile_time_sync and hidden_states.is_cuda:
                    torch.cuda.synchronize()

            def _mark(*_args, **_kwargs):
                return None

            _sync()
            t0 = time.perf_counter()
        height, width = hidden_states.shape[-2:]
        hidden_states = self.base_model.pos_embed(hidden_states)
        if profile_time:
            _sync()
            t1 = time.perf_counter()
            _mark("pos_embed", t0, t1)
            last_timestamp = t1

        temb = self.base_model.time_text_embed(timestep, pooled_projections)
        if profile_time:
            _sync()
            t2 = time.perf_counter()
            _mark("time_text_embed", t1, t2)
            last_timestamp = t2
        encoder_hidden_states = self.base_model.context_embedder(encoder_hidden_states)
        if profile_time:
            _sync()
            t3 = time.perf_counter()
            _mark("context_embedder", t2, t3)
            last_timestamp = t3
        if force_txt_grad and not encoder_hidden_states.requires_grad:
            encoder_hidden_states = encoder_hidden_states.detach().requires_grad_(True)


        img_hidden_states_list = []
        txt_hidden_states_list = []
        txt_input_states_list = []
        
        context_embedder_output = encoder_hidden_states
        txt_hidden_states_list.append(encoder_hidden_states)
        
 

        # ---------------- residual config ----------------
        use_residual = (
            residual_origin_layer is not None
            and residual_target_layers is not None
            and residual_weights is not None
        )

        if use_residual:
            residual_target_to_idx = {layer: idx for idx, layer in enumerate(residual_target_layers)}
            residual_target_set = set(residual_target_layers)
            residual_weights = torch.as_tensor(
                residual_weights,
                dtype=encoder_hidden_states.dtype,
                device=encoder_hidden_states.device,
            )

            residual_rotations = None
            if residual_rotation_matrices is not None:
                if isinstance(residual_rotation_matrices, (list, tuple)):
                    if all(torch.is_tensor(r) for r in residual_rotation_matrices):
                        residual_rotations = torch.stack(residual_rotation_matrices, dim=0)
                    else:
                        residual_rotations = torch.tensor(residual_rotation_matrices)
                elif torch.is_tensor(residual_rotation_matrices):
                    residual_rotations = residual_rotation_matrices
                else:
                    raise TypeError(
                        "residual_rotation_matrices must be a Tensor or a list/tuple of Tensors."
                    )
                if residual_rotations.dim() == 2:
                    residual_rotations = residual_rotations.unsqueeze(0)
                if residual_rotations.dim() != 3:
                    raise ValueError(
                        "residual_rotation_matrices must have shape (N, D, D) or (D, D)."
                    )
                residual_rotations = residual_rotations.to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype,
                )
                if residual_rotations.shape[0] != len(residual_target_layers):
                    raise ValueError(
                        "residual_rotation_matrices length must match residual_target_layers."
                    )
                if residual_rotations.shape[-1] != encoder_hidden_states.shape[-1] or \
                    residual_rotations.shape[-2] != encoder_hidden_states.shape[-1]:
                    raise ValueError(
                        "residual_rotation_matrices feature dimension must match encoder_hidden_states."
                    )

            pre_encoder_states = []

        # ---------------- iterate transformer blocks ----------------
        for index_block, block in enumerate(self.base_model.transformer_blocks):
            is_skip = skip_layers is not None and index_block in skip_layers
            if output_text_inputs and not is_skip:
                txt_input_states_list.append(encoder_hidden_states)

            if use_residual:
                pre_encoder_states.append(encoder_hidden_states)

                if index_block in residual_target_set:
                    tid = residual_target_to_idx[index_block]
                    w = residual_weights[tid]

                    # pick origin state
                    if 0 <= residual_origin_layer < len(pre_encoder_states):
                        origin = pre_encoder_states[residual_origin_layer]
                    else:
                        raise ValueError(f"Invalid residual_origin_layer={residual_origin_layer}")

                    if origin.shape != encoder_hidden_states.shape:
                        raise ValueError(
                            f"[Residual] Shape mismatch: origin={origin.shape} vs target={encoder_hidden_states.shape}"
                        )

                    # --------- 改进版 residual 应用 ---------
                    rotation = residual_rotations[tid] if residual_rotations is not None else None
                    if profile_time:
                        _sync()
                        t_residual_start = time.perf_counter()
                    encoder_hidden_states = self._apply_residual(
                        encoder_hidden_states,
                        origin,
                        w,
                        use_layernorm=residual_use_layernorm,
                        stop_grad=residual_stop_grad,
                        rotation_matrix=rotation,
                    )
                    if profile_time:
                        _sync()
                        t_residual_end = time.perf_counter()
                        _mark("residual_apply", t_residual_start, t_residual_end)

            # ---------------- transformer compute ----------------
            if torch.is_grad_enabled() and self.base_model.gradient_checkpointing and not is_skip:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs = {}
                if profile_time:
                    _sync()
                    t_block_start = time.perf_counter()
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, encoder_hidden_states, temb, joint_attention_kwargs, **ckpt_kwargs
                )
                if profile_time:
                    _sync()
                    t_block_end = time.perf_counter()
                    _mark("blocks", t_block_start, t_block_end)
                    last_timestamp = t_block_end
            elif not is_skip:
                if profile_time:
                    _sync()
                    t_block_start = time.perf_counter()
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                if profile_time:
                    _sync()
                    t_block_end = time.perf_counter()
                    _mark("blocks", t_block_start, t_block_end)
                    last_timestamp = t_block_end

            if output_hidden_states:
                img_hidden_states_list.append(hidden_states)
                # if encoder_hidden_states is not None:
                #         encoder_hidden_states.retain_grad()
                txt_hidden_states_list.append(encoder_hidden_states)

        # -------------- output unchanged --------------
        hidden_states = self.base_model.norm_out(hidden_states, temb)
        if profile_time:
            _sync()
            t_norm = time.perf_counter()
            _mark("norm_out", last_timestamp, t_norm)
            last_timestamp = t_norm
        hidden_states = self.base_model.proj_out(hidden_states)
        if profile_time:
            _sync()
            t_proj = time.perf_counter()
            _mark("proj_out", last_timestamp, t_proj)
            last_timestamp = t_proj

        patch_size = self.base_model.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.base_model.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            (hidden_states.shape[0], self.base_model.out_channels, height * patch_size, width * patch_size)
        )
        if profile_time:
            _sync()
            t_end = time.perf_counter()
            total_time = t_end - t0
            logger.info("[profile] forward_total(s): %.6f", total_time)
            if profile_time_path is not None:
                _write_timings(profile_time_path, total_time)

        if not return_dict:
            return {
                "sample": output,
                "img_hidden_states": img_hidden_states_list,
                "txt_hidden_states": txt_hidden_states_list,
                "txt_input_states": txt_input_states_list,
                "context_embedder_output": context_embedder_output,
            }

        return Transformer2DModelOutput(sample=output)








class SD3Transformer2DModel_Vanilla(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def to(self, *args, **kwargs):
        # 将当前模块和子模块（尤其是 base_model）都转移到 device/dtype
        self.base_model = self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        height, width = hidden_states.shape[-2:]

        hidden_states = self.base_model.pos_embed(hidden_states)
        temb = self.base_model.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.base_model.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.base_model.transformer_blocks):
            is_skip = True if skip_layers is not None and index_block in skip_layers else False

            if torch.is_grad_enabled() and self.base_model.gradient_checkpointing and not is_skip:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )


        hidden_states = self.base_model.norm_out(hidden_states, temb)
        hidden_states = self.base_model.proj_out(hidden_states)


        # unpatchify
        patch_size = self.base_model.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.base_model.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.base_model.out_channels, height * patch_size, width * patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    
    
    
    
class SD3Transformer2DModel_REPA(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.dtype = base_model.dtype
        

    def to(self, *args, **kwargs):
        # 将当前模块和子模块（尤其是 base_model）都转移到 device/dtype
        self.base_model = self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)



    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        target_layers: Optional[List[int]] = None,   # 支持多层
        replace_cond_embed: bool = False,
        output_hidden_states: bool = False,
        # --- 新增：跳连参数 ---
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weight: float = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        height, width = hidden_states.shape[-2:]
        hidden_states = self.base_model.pos_embed(hidden_states)
        temb = self.base_model.time_text_embed(timestep, pooled_projections)

        if not replace_cond_embed: 
            encoder_hidden_states = self.base_model.context_embedder(encoder_hidden_states)

        context_embedder_output = encoder_hidden_states

        img_feats_list = []
        txt_feats_list = []
        img_hidden_states_list = []
        txt_hidden_states_list = []

        if residual_origin_layer is not None:
            # --- 新增：记录每层进入前的 encoder_hidden_states ---
            pre_encoder_states = []

            for index_block, block in enumerate(self.base_model.transformer_blocks):
                # 保存进入该层前的 encoder_hidden_states
                pre_encoder_states.append(encoder_hidden_states)

                # --- 跳连：若该层在 residual_target_layers 中，执行加权残差 ---
                if (
                    residual_target_layers is not None
                    and residual_origin_layer is not None
                    and index_block in residual_target_layers
                    and 0 <= residual_origin_layer < len(pre_encoder_states)
                ):
                    origin_enc = pre_encoder_states[residual_origin_layer]
                    if origin_enc.dtype != encoder_hidden_states.dtype or origin_enc.device != encoder_hidden_states.device:
                        origin_enc = origin_enc.to(dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
                    if origin_enc.shape == encoder_hidden_states.shape:
                        encoder_hidden_states = encoder_hidden_states + residual_weight * origin_enc
                    else:
                        raise ValueError(
                            f"Residual skip shape mismatch: origin {origin_enc.shape} vs current {encoder_hidden_states.shape}"
                        )

                is_skip = skip_layers is not None and index_block in skip_layers

                # gradient checkpointing 支持
                if torch.is_grad_enabled() and self.base_model.gradient_checkpointing and not is_skip:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            return module(*inputs, return_dict=return_dict) if return_dict is not None else module(*inputs)
                        return custom_forward

                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states, encoder_hidden_states, temb, joint_attention_kwargs, **ckpt_kwargs
                    )
                elif not is_skip:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                if output_hidden_states:
                    img_hidden_states_list.append(hidden_states)
                    txt_hidden_states_list.append(encoder_hidden_states)

                if target_layers is not None and index_block in target_layers:
                    img_feats_list.append(hidden_states)
                    txt_feats_list.append(encoder_hidden_states)
        
        else:
            for index_block, block in enumerate(self.base_model.transformer_blocks):
                is_skip = skip_layers is not None and index_block in skip_layers

                if torch.is_grad_enabled() and self.base_model.gradient_checkpointing and not is_skip:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            return module(*inputs, return_dict=return_dict) if return_dict is not None else module(*inputs)
                        return custom_forward

                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states, encoder_hidden_states, temb, joint_attention_kwargs, **ckpt_kwargs
                    )
                elif not is_skip:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                if output_hidden_states:
                    img_hidden_states_list.append(hidden_states)
                    txt_hidden_states_list.append(encoder_hidden_states)

                # 若当前层在 target_layers 中，则记录特征
                if target_layers is not None and index_block in target_layers:
                    img_feats_list.append(hidden_states)
                    txt_feats_list.append(encoder_hidden_states)
    
        hidden_states = self.base_model.norm_out(hidden_states, temb)
        hidden_states = self.base_model.proj_out(hidden_states)

        # unpatchify
        patch_size = self.base_model.config.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.base_model.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.base_model.out_channels, height * patch_size, width * patch_size)
        )

        if not return_dict:
            return {
                "sample": output,
                "img_feats_list": img_feats_list,
                "txt_feats_list": txt_feats_list,
                "img_hidden_states": img_hidden_states_list,
                "txt_hidden_states": txt_hidden_states_list,
                "context_embedder_output": context_embedder_output,
            }

        if len(img_feats_list) or len(txt_feats_list):
            return Transformer2DModelOutput_REPA(
                sample=output,
                img_feats=img_feats_list,
                txt_feats=txt_feats_list,
            )
        else:
            return Transformer2DModelOutput(sample=output)




@dataclass
class Transformer2DModelOutput_REPA(Transformer2DModelOutput):
    """
    Extended output of `Transformer2DModel`, including intermediate representation z.

    Args:
        sample (`torch.Tensor`): The final denoised image output.
        img_feats (`torch.Tensor`, optional): The projected intermediate hidden state (e.g., from target layer).
    """
    img_feats: torch.Tensor = None  # shape: (B, N, D_target)
    txt_feats: torch.Tensor = None  # shape: (B, N, D_target)
    
    




# try:
#     from diffusers.models.transformers.transformer_sd3 import SD3SingleTransformerBlock
# except Exception as e:
#     raise ImportError("Cannot import SD3SingleTransformerBlock from diffusers. "
#                       "Please ensure diffusers >= 0.27 is installed.") from e


# class _AdaFreeLayerNorm(nn.Module):
#     """
#     伪 AdaLN：接口与 AdaLayerNormZero 一致，但只做普通 LN，
#     并返回恒等门控 (gate_msa=1, gate_mlp=1, shift=0, scale=0) 来兼容 SD3SingleTransformerBlock.forward 的解包。
#     """
#     def __init__(self, dim: int, eps: float = 1e-6, affine: bool = False):
#         super().__init__()
#         self.ln = nn.LayerNorm(dim, elementwise_affine=affine, eps=eps)

#     def forward(self, hidden_states: torch.Tensor, emb=None):
#         h = self.ln(hidden_states)  # (B, S, D)
#         B, _, D = h.shape
#         device, dtype = h.device, h.dtype
#         ones  = torch.ones(B, D, device=device, dtype=dtype)
#         zeros = torch.zeros(B, D, device=device, dtype=dtype)
#         # 返回五元组，匹配 AdaLayerNormZero 的签名
#         return h, ones, zeros, zeros, ones


# class SD3StyleTextTransformerLN(nn.Module):
#     """
#     SD3 风格、单流文本 Transformer（去 AdaLN）：
#       - 输入:  x ∈ (B, S, D=trans_hidden)
#       - 输出:  y ∈ (B, S, D=trans_hidden)
#     复用 SD3SingleTransformerBlock（Attention + FFN），将其 norm1 热插拔为普通 LN。
#     """
#     def __init__(
#         self,
#         trans_hidden: int = 3584,   # 你要的隐藏维度
#         depth: int = 2,             # 堆叠层数，替代原 Phi3DecoderLayer×2
#         head_dim: int = 64,         # 与 SD3 风格保持一致
#         num_heads: int | None = None,
#         final_norm: bool = True,
#     ):
#         super().__init__()
#         if num_heads is None:
#             assert trans_hidden % head_dim == 0, "trans_hidden 必须能被 head_dim 整除"
#             num_heads = trans_hidden // head_dim

#         self.blocks = nn.ModuleList([
#             SD3SingleTransformerBlock(
#                 dim=trans_hidden,
#                 num_attention_heads=num_heads,
#                 attention_head_dim=head_dim,
#             ) for _ in range(depth)
#         ])
#         # 把 AdaLayerNormZero 换成普通 LN 兼容模块
#         for blk in self.blocks:
#             blk.norm1 = _AdaFreeLayerNorm(trans_hidden, eps=1e-6, affine=False)

#         self.norm_out = nn.LayerNorm(trans_hidden, elementwise_affine=False, eps=1e-6) if final_norm else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, S, trans_hidden)
#         # 由于 norm1 已改为普通 LN，temb 不再需要，传占位即可
#         B, _, D = x.shape
#         temb = x.new_zeros(B, D)
#         h = x
#         for blk in self.blocks:
#             h = blk(h, temb)  # (B, S, D)
#         h = self.norm_out(h)
#         return h



# class Connector(nn.Module):
#     """
#     MLLM 序列特征 → 
#       - 如果 depth>0: (可选 Linear) → Transformer → Linear(3584→1536)
#       - 如果 depth=0: (可选 Linear) → MLP(3584→1536→1536) → Linear(1536→1536)

#     参数:
#     - in_dim: 输入特征维度，例如 Qwen 3584
#     - out_dim: 输出特征维度 (默认 1536，对应 SD3)
#     - trans_hidden: Transformer/MLP 的中间维度 (默认 3584)
#     - depth: 如果大于0，使用 Transformer；如果等于0，使用 MLP
#     - head_dim, num_heads: Transformer 配置
#     """
#     def __init__(
#         self,
#         in_dim: int = 3584,
#         out_dim: int = 1536,
#         trans_hidden: int = 3584,
#         depth: int = 2,
#         layers: int = 2,
#         head_dim: int = 64,
#         num_heads: int | None = None,
#     ):
#         super().__init__()
#         self.depth = depth

#         # 输入维度对齐
#         self.pre_map = (
#             nn.Linear(in_dim, trans_hidden) 
#             if in_dim != trans_hidden else nn.Identity()
#         )

#         if depth > 0:
#             # Transformer encoder
#             self.encoder = SD3StyleTextTransformerLN(
#                 trans_hidden=trans_hidden,
#                 depth=depth,
#                 head_dim=head_dim,
#                 num_heads=num_heads,
#                 final_norm=True,
#             )
#             self.post_map = nn.Linear(trans_hidden, out_dim)
#         else:
#             if layers == 1:
#                 self.encoder = nn.Identity()
#                 self.post_map = nn.Identity()
#             elif layers == 2:
#                 # 双层 MLP
#                 self.encoder = nn.GELU()
#                 self.post_map = nn.Linear(trans_hidden, out_dim)
#             else:
#                 raise TypeError("Not implement for multi layer projector!")
                



#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, S, in_dim) 
#         """
#         h = self.pre_map(x)   # (B, S, trans_hidden or unchanged)
#         h = self.encoder(h)   # (B, S, trans_hidden or trans_hidden)
#         y = self.post_map(h)  # (B, S, out_dim)
#         return y




# # =========================
# # 加载 connector 权重（支持 DP 前缀去除）
# # =========================
# def load_connector_from_ckpt(
#     ckpt_path: str,
#     device: str = "cuda",
#     in_dim: int | None = None,
#     out_dim: int | None = None,
#     trans_hidden: int = 3584,
#     depth: int = 2,
#     head_dim: int = 64,
#     num_heads: int | None = None,
#     dtype: torch.dtype = torch.float32,
#     strict: bool = True,
# ):
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     sd = ckpt["connector"]

#     in_dim = in_dim if in_dim is not None else ckpt.get("d_qwen", 3584)
#     out_dim = out_dim if out_dim is not None else ckpt.get("sd3_txt_hidden", 1536)

#     connector = Connector(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         trans_hidden=trans_hidden,
#         depth=depth,
#         head_dim=head_dim,
#         num_heads=num_heads,
#     )

#     if any(k.startswith("module.") for k in sd.keys()):
#         sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

#     missing, unexpected = connector.load_state_dict(sd, strict=strict)
#     if missing or unexpected:
#         print(f"[WARN] load_state_dict missing={missing}, unexpected={unexpected}")

#     connector = connector.to(device=device, dtype=dtype)
#     return connector
