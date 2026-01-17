from typing import Any, Dict, List, Optional, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from dataclasses import dataclass
import torch
import torch.nn as nn

    
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.utils import logging
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)



class SD3Transformer2DModel_Residual(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.dtype = base_model.dtype

    def to(self, *args, **kwargs):
        self.base_model = self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ===============================================================
    #  token-wise RMS 归一化
    # ===============================================================
    @staticmethod
    def _rms_norm_tokenwise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x / rms

    # ===============================================================
    #  改进 residual（支持 stopgrad + 可选 LN）
    # ===============================================================
    # def _apply_residual(
    #     self,
    #     target: torch.Tensor,
    #     origin: torch.Tensor,
    #     w: torch.Tensor,
    #     use_layernorm: bool = True,
    #     stop_grad: bool = True,
    #     rotation_matrix: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     target/origin: [B, L, D]
    #     w: scalar tensor
    #     """

    #     # ----------- STOP GRADIENT PART -----------
    #     # residual 的 2 个输入都不参与梯度
    #     if stop_grad:
    #         target_nograd = target.detach()
    #         origin_nograd = origin.detach()
    #     else:
    #         target_nograd = target
    #         origin_nograd = origin

    #     mu_tgt = target_nograd.mean(dim=-1, keepdim=True)
    #     sigma_tgt = target_nograd.std(dim=-1, keepdim=True)

    #     t_norm = self._rms_norm_tokenwise(target_nograd)
    #     o_norm = self._rms_norm_tokenwise(origin_nograd)
    #     if rotation_matrix is not None:
    #         o_norm = torch.matmul(o_norm, rotation_matrix)

    #     w = torch.clamp(w, min=0)
    #     mixed = t_norm + w * o_norm

    #     if use_layernorm:
    #         mixed = self._rms_norm_tokenwise(mixed)

    #     return mixed * sigma_tgt + mu_tgt
    def _apply_residual(
            self,
            target: torch.Tensor,
            origin: torch.Tensor,
            w: torch.Tensor,
            stop_grad: bool = True,
            rotation_matrix: Optional[torch.Tensor] = None,
            mu_src_glob: Optional[torch.Tensor] = None,
            mu_tgt_glob: Optional[torch.Tensor] = None,
        ):
            """
            target/origin: [B, L, D]
            w: scalar tensor
            """

            # ----------- STOP GRADIENT PART -----------
            if stop_grad:
                target_nograd = target.detach()
                origin_nograd = origin.detach()
            else:
                target_nograd = target
                origin_nograd = origin

            # 1. 提取 Target 的锚点统计量
            mu_tgt = target_nograd.mean(dim=-1, keepdim=True)
            sigma_tgt = target_nograd.std(dim=-1, keepdim=True)

            # 2. 几何融合 (保持使用 RMSNorm 以保护语义方向)
            t_norm = self._rms_norm_tokenwise(target_nograd)
            o_norm = self._rms_norm_tokenwise(origin_nograd)
            if mu_tgt_glob is not None:
                t_centered = t_norm - mu_tgt_glob
            else:
                t_centered = t_norm - t_norm.mean(dim=-2, keepdim=True)

            if mu_src_glob is not None:
                o_centered = o_norm - mu_src_glob
            else:
                o_centered = o_norm - o_norm.mean(dim=-2, keepdim=True)

            if rotation_matrix is not None:
                # 注意：如果 rotation_matrix 是 (D, D)，matmul 默认是最后两维运算，符合预期
                o_centered = torch.matmul(o_centered, rotation_matrix)

            w = torch.clamp(w, min=0)
            mixed = t_centered + w * o_centered

            # 3. 分布恢复 (CRITICAL FIX)
            # 必须先将 mixed 强制变为 均值0、方差1 的分布，
            # 才能正确地映射回 Target 的数值空间。
            # 这里不能用 RMSNorm，必须用类似 LayerNorm 的标准化逻辑（但不含 affine 参数）
            if mu_src_glob is not None and mu_tgt_glob is not None:
                mixed_std = mixed.std(dim=-1, keepdim=True)
                mixed_normalized = mixed / (mixed_std + 1e-6)
            else:
                mixed_mean = mixed.mean(dim=-1, keepdim=True)
                mixed_std = mixed.std(dim=-1, keepdim=True)

                # 显式标准化：(x - u) / s
                # 加上 1e-6 防止除以 0
                mixed_normalized = (mixed - mixed_mean) / (mixed_std + 1e-6)

            # 4. 锚定映射
            # 现在 mixed_normalized 是标准的 N(0,1)，乘以 sigma 加 mu 后
            # 就完美变成了 N(mu_tgt, sigma_tgt)
            return mixed_normalized * sigma_tgt + mu_tgt

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
        residual_stop_grad: bool = True,

        # --- residual 参数 ---
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weights: Optional[Union[List[float], torch.Tensor]] = None,
        residual_rotation_matrices: Optional[Union[List[torch.Tensor], torch.Tensor, Dict[str, Any]]] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        height, width = hidden_states.shape[-2:]
        hidden_states = self.base_model.pos_embed(hidden_states)

        temb = self.base_model.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.base_model.context_embedder(encoder_hidden_states)
        if force_txt_grad and not encoder_hidden_states.requires_grad:
            encoder_hidden_states = encoder_hidden_states.detach().requires_grad_(True)

        context_embedder_output = encoder_hidden_states

        img_hidden_states_list = []
        txt_hidden_states_list = []
        txt_input_states_list = []

        # ---------------- residual config ----------------
        use_residual = (
            residual_origin_layer is not None
            and residual_target_layers is not None
            and residual_weights is not None
        )

        if use_residual:
            if isinstance(residual_weights, (list, tuple)):
                residual_weights = torch.tensor(residual_weights, dtype=encoder_hidden_states.dtype)
            residual_weights = residual_weights.to(encoder_hidden_states.device)

            residual_rotations = None
            residual_mu_src_tensor = None
            residual_mu_tgt_tensor = None
            if residual_rotation_matrices is not None:
                if isinstance(residual_rotation_matrices, dict):
                    residual_mu_src_tensor = residual_rotation_matrices.get("mu_src")
                    residual_mu_tgt_tensor = residual_rotation_matrices.get("mu_tgt")
                    residual_rotation_matrices = residual_rotation_matrices.get(
                        "rotation_matrices", residual_rotation_matrices.get("R")
                    )
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

            if residual_mu_src_tensor is not None:
                if isinstance(residual_mu_src_tensor, (list, tuple)):
                    if all(torch.is_tensor(m) for m in residual_mu_src_tensor):
                        residual_mu_src_tensor = torch.stack(residual_mu_src_tensor, dim=0)
                    else:
                        residual_mu_src_tensor = torch.tensor(residual_mu_src_tensor)
                elif not torch.is_tensor(residual_mu_src_tensor):
                    raise TypeError("residual mu_src must be a Tensor or a list/tuple of Tensors.")
                residual_mu_src_tensor = residual_mu_src_tensor.to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype,
                )
                if residual_mu_src_tensor.dim() == 1:
                    residual_mu_src_tensor = residual_mu_src_tensor.unsqueeze(0)
                if residual_mu_src_tensor.dim() != 2:
                    raise ValueError("residual_mu_src must have shape (D,) or (N, D).")
                if residual_mu_src_tensor.shape[-1] != encoder_hidden_states.shape[-1]:
                    raise ValueError(
                        "residual_mu_src feature dimension must match encoder_hidden_states."
                    )
                if residual_mu_src_tensor.shape[0] not in (1, len(residual_target_layers)):
                    raise ValueError(
                        "residual_mu_src length must be 1 or match residual_target_layers."
                    )

            if residual_mu_tgt_tensor is not None:
                if isinstance(residual_mu_tgt_tensor, (list, tuple)):
                    if all(torch.is_tensor(m) for m in residual_mu_tgt_tensor):
                        residual_mu_tgt_tensor = torch.stack(residual_mu_tgt_tensor, dim=0)
                    else:
                        residual_mu_tgt_tensor = torch.tensor(residual_mu_tgt_tensor)
                elif not torch.is_tensor(residual_mu_tgt_tensor):
                    raise TypeError("residual mu_tgt must be a Tensor or a list/tuple of Tensors.")
                residual_mu_tgt_tensor = residual_mu_tgt_tensor.to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype,
                )
                if residual_mu_tgt_tensor.dim() == 1:
                    residual_mu_tgt_tensor = residual_mu_tgt_tensor.unsqueeze(0)
                if residual_mu_tgt_tensor.dim() != 2:
                    raise ValueError("residual_mu_tgt must have shape (D,) or (N, D).")
                if residual_mu_tgt_tensor.shape[-1] != encoder_hidden_states.shape[-1]:
                    raise ValueError(
                        "residual_mu_tgt feature dimension must match encoder_hidden_states."
                    )
                if residual_mu_tgt_tensor.shape[0] not in (1, len(residual_target_layers)):
                    raise ValueError(
                        "residual_mu_tgt length must be 1 or match residual_target_layers."
                    )

            pre_encoder_states = []

        # ---------------- iterate transformer blocks ----------------
        for index_block, block in enumerate(self.base_model.transformer_blocks):
            is_skip = skip_layers is not None and index_block in skip_layers
            if output_text_inputs and not is_skip:
                txt_input_states_list.append(encoder_hidden_states)

            if use_residual:
                pre_encoder_states.append(encoder_hidden_states)

                if index_block in residual_target_layers:
                    tid = residual_target_layers.index(index_block)
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
                    mu_src = None
                    if residual_mu_src_tensor is not None:
                        if residual_mu_src_tensor.shape[0] == 1:
                            mu_src = residual_mu_src_tensor[0]
                        else:
                            mu_src = residual_mu_src_tensor[tid]
                    mu_tgt = None
                    if residual_mu_tgt_tensor is not None:
                        if residual_mu_tgt_tensor.shape[0] == 1:
                            mu_tgt = residual_mu_tgt_tensor[0]
                        else:
                            mu_tgt = residual_mu_tgt_tensor[tid]
                    encoder_hidden_states = self._apply_residual(
                        encoder_hidden_states,
                        origin,
                        w,
                        stop_grad=residual_stop_grad,
                        rotation_matrix=rotation,
                        mu_src_glob=mu_src,
                        mu_tgt_glob=mu_tgt,
                    )

            # ---------------- transformer compute ----------------
            if torch.is_grad_enabled() and self.base_model.gradient_checkpointing and not is_skip:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs = {}
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
                # if encoder_hidden_states is not None:
                #         encoder_hidden_states.retain_grad()
                txt_hidden_states_list.append(encoder_hidden_states)

        # -------------- output unchanged --------------
        hidden_states = self.base_model.norm_out(hidden_states, temb)
        hidden_states = self.base_model.proj_out(hidden_states)

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
