import torch
from typing import List, Optional, Dict, Any, Union
from diffusers import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch import nn


class FluxTransformer2DModel_RES(nn.Module):
    """
    适配FLUX的文本流残差注入模块：
    1. 修复LayerNorm的normalized_shape类型错误（int→tuple）
    2. 修复整数权重导致的AttributeError
    3. 1个源层 + 多目标层 + 单/多权重
    4. 标准化空间叠加残差（z-score → 叠加 → LayerNorm → 恢复分布）
    5. 对齐SD3残差逻辑：LayerNorm Residual + Procrustes + Learnable weights
    5. 注释掉所有详细打印，仅保留极简日志
    """
    def __init__(self, base_model: FluxTransformer2DModel):
        super().__init__()
        self.base_model = base_model  # 原生 Flux Transformer
        self.dtype = base_model.dtype
        self.config = base_model.config
        self.cache_context = base_model.cache_context  # 显式绑定缓存上下文方法

    def to(self, *args, **kwargs):
        """确保基础模型和当前模块设备/ dtype 一致"""
        self.base_model = self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    @staticmethod
    def _standardize_tokenwise(x: torch.Tensor, eps: float = 1e-6, layer_idx: int = -1):
        """
        对最后一维 (hidden_dim) 做 token-wise z-score 标准化
        支持：
        - 3维特征：[batch_size, seq_len, hidden_dim]
        - 4维特征：[batch_size, num_img, seq_len, hidden_dim]（自动展平为3维）
        """
        # 记录原始形状，用于恢复
        original_shape = x.shape
        # 维度适配与校验
        if x.ndim == 4:
            # 展平4维→3维：[batch, num_img, seq_len, hidden] → [batch*num_img, seq_len, hidden]
            batch_size, num_img, seq_len, hidden_dim = x.shape
            x = x.reshape(batch_size * num_img, seq_len, hidden_dim)
        elif x.ndim != 3:
            raise ValueError(
                f"token-wise标准化仅支持3维/4维特征，当前输入维度：{x.ndim}，形状：{original_shape}"
            )
        
        # Token-wise标准化（每个token的hidden_dim维度）
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        
        # 数值裁剪：避免std过小导致x_norm爆炸
        std = torch.clamp(std, min=eps)
        x_norm = (x - mean) / (std + eps)
        
        # 恢复原始形状（若输入是4维）
        if len(original_shape) == 4:
            x_norm = x_norm.reshape(original_shape)
            mean = mean.reshape(original_shape[:-1] + (1,))
            std = std.reshape(original_shape[:-1] + (1,))
        
        return x_norm, mean, std

    def _apply_residual(
        self,
        target: torch.Tensor,
        origin: torch.Tensor,
        w: torch.Tensor,
        *,
        use_layernorm: bool = True,
        stop_grad: bool = False,
        rotation_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if stop_grad:
            target_nograd = target.detach()
            origin_nograd = origin.detach()
        else:
            target_nograd = target
            origin_nograd = origin

        target_norm, target_mean, target_std = self._standardize_tokenwise(target_nograd)
        origin_norm, _, _ = self._standardize_tokenwise(origin_nograd)

        if rotation_matrix is not None:
            origin_norm = torch.matmul(origin_norm, rotation_matrix)

        if w >= 0:
            mixed = target_norm + w * origin_norm
        else:
            mixed = target_norm * (1 - w)

        if use_layernorm:
            mixed = torch.nn.functional.layer_norm(
                mixed, normalized_shape=(mixed.shape[-1],), eps=1e-6
            )

        return mixed * target_std + target_mean

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        output_text_inputs: bool = False,
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weights: Optional[Union[List[float], torch.Tensor]] = None,
        residual_use_layernorm: bool = True,
        residual_rotation_matrices: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        residual_stop_grad: bool = False,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Transformer2DModelOutput]:
        """
        核心逻辑：
        1. 修复LayerNorm参数类型错误
        2. 修复整数权重的类型错误
        3. 仅保存源层输入block前的特征（参考代码逻辑，减少显存占用）
        4. 标准化空间叠加残差：z-score → 叠加 → LayerNorm → 恢复分布
        5. residual_target_layers 支持单流块索引（从双流块数量开始计数）
        """
        # 2. 处理时间步嵌入（严格对齐 FLUX 原生逻辑）
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        
        # 生成时间步嵌入 temb
        temb = (
            self.base_model.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.base_model.time_text_embed(timestep, guidance, pooled_projections)
        )

        # 3. 生成 Rotary 位置嵌入（FLUX 原生逻辑）
        image_rotary_emb = None
        if img_ids is not None and txt_ids is not None:
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]
            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = self.base_model.pos_embed(ids)

        # 4. 文本/图像特征嵌入（原生逻辑）
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.base_model.context_embedder(encoder_hidden_states)
        hidden_states = self.base_model.x_embedder(hidden_states)

        use_residual = (
            residual_origin_layer is not None
            and residual_target_layers is not None
            and residual_weights is not None
        )

        txt_input_states_list = []
        residual_weights_tensor = None
        residual_rotations = None
        if use_residual:
            if isinstance(residual_weights, (list, tuple)):
                residual_weights_tensor = torch.tensor(
                    residual_weights, dtype=encoder_hidden_states.dtype
                )
            elif torch.is_tensor(residual_weights):
                residual_weights_tensor = residual_weights.to(dtype=encoder_hidden_states.dtype)
            else:
                raise TypeError(
                    "residual_weights must be a Tensor or a list/tuple of floats."
                )
            residual_weights_tensor = residual_weights_tensor.to(encoder_hidden_states.device)

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

        # 6. 遍历第一阶段：双流 Transformer 块（transformer_blocks）
        num_transformer_blocks = len(self.base_model.transformer_blocks)
        for index_block, block in enumerate(self.base_model.transformer_blocks):
            if output_text_inputs and encoder_hidden_states is not None:
                txt_input_states_list.append(encoder_hidden_states)
            if use_residual and encoder_hidden_states is not None:
                pre_encoder_states.append(encoder_hidden_states)

                if index_block in residual_target_layers:
                    tid = residual_target_layers.index(index_block)
                    w = residual_weights_tensor[tid]

                    if 0 <= residual_origin_layer < len(pre_encoder_states):
                        origin_enc = pre_encoder_states[residual_origin_layer]
                    else:
                        raise ValueError(f"Invalid residual_origin_layer={residual_origin_layer}")

                    if origin_enc.shape != encoder_hidden_states.shape:
                        raise ValueError(
                            f"[Residual] Shape mismatch: origin={origin_enc.shape} vs target={encoder_hidden_states.shape}"
                        )

                    rotation = residual_rotations[tid] if residual_rotations is not None else None
                    encoder_hidden_states = self._apply_residual(
                        encoder_hidden_states,
                        origin_enc,
                        w,
                        use_layernorm=residual_use_layernorm,
                        stop_grad=residual_stop_grad,
                        rotation_matrix=rotation,
                    )

            # 执行当前双流块（参数顺序完全对齐FLUX原生）
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # 7. 遍历第二阶段：单流 Transformer 块
        for index_block, block in enumerate(self.base_model.single_transformer_blocks):
            if output_text_inputs and encoder_hidden_states is not None:
                txt_input_states_list.append(encoder_hidden_states)

            if use_residual and encoder_hidden_states is not None:
                pre_encoder_states.append(encoder_hidden_states)
                global_index = num_transformer_blocks + index_block

                if global_index in residual_target_layers:
                    tid = residual_target_layers.index(global_index)
                    w = residual_weights_tensor[tid]

                    if 0 <= residual_origin_layer < len(pre_encoder_states):
                        origin_enc = pre_encoder_states[residual_origin_layer]
                    else:
                        raise ValueError(f"Invalid residual_origin_layer={residual_origin_layer}")

                    if origin_enc.shape != encoder_hidden_states.shape:
                        raise ValueError(
                            f"[Residual] Shape mismatch: origin={origin_enc.shape} vs target={encoder_hidden_states.shape}"
                        )

                    rotation = residual_rotations[tid] if residual_rotations is not None else None
                    encoder_hidden_states = self._apply_residual(
                        encoder_hidden_states,
                        origin_enc,
                        w,
                        use_layernorm=residual_use_layernorm,
                        stop_grad=residual_stop_grad,
                        rotation_matrix=rotation,
                    )

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # 8. 输出投影（FLUX原生逻辑）
        hidden_states = self.base_model.norm_out(hidden_states, temb)
        hidden_states = self.base_model.proj_out(hidden_states)

        if not return_dict:
            if output_text_inputs:
                return {"sample": hidden_states, "txt_input_states": txt_input_states_list}
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
