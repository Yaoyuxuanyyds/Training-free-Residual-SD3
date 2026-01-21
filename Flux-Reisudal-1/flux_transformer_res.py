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
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weight: Optional[Union[List[float], torch.Tensor, float, int]] = 1.0,  # 新增int类型支持
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Transformer2DModelOutput]:
        """
        核心逻辑：
        1. 修复LayerNorm参数类型错误
        2. 修复整数权重的类型错误
        3. 仅保存源层输入block前的特征（参考代码逻辑，减少显存占用）
        4. 标准化空间叠加残差：z-score → 叠加 → LayerNorm → 恢复分布
        """
        # 1. 初始化源层特征缓存（仅保存源层，而非所有层）
        _saved_origin_text = None

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

        # 预处理残差权重（修复整数类型错误）
        residual_weights_tensor = None
        if residual_weight is not None:
            # 情况1：列表/元组 → 转为tensor
            if isinstance(residual_weight, (list, tuple)):
                residual_weights_tensor = torch.tensor(residual_weight, dtype=torch.float32)
            # 情况2：Python原生int/float → 转为长度1的tensor
            elif isinstance(residual_weight, (int, float)):
                residual_weights_tensor = torch.tensor([float(residual_weight)], dtype=torch.float32)
            # 情况3：torch.Tensor → 转为float32
            elif isinstance(residual_weight, torch.Tensor):
                residual_weights_tensor = residual_weight.float()
            # 其他情况 → 报错提示
            else:
                raise TypeError(f"不支持的权重类型：{type(residual_weight)}，仅支持list/tuple/int/float/torch.Tensor")

        # 6. 遍历第一阶段：双流 Transformer 块（transformer_blocks）
        for index_block, block in enumerate(self.base_model.transformer_blocks):
            # 源层：保存输入当前block前的文本流特征（仅保存一次，参考代码逻辑）
            if (
                residual_origin_layer is not None
                and index_block == residual_origin_layer
                and encoder_hidden_states is not None
            ):
                _saved_origin_text = encoder_hidden_states.detach().clone()

            # 残差叠加：标准化空间叠加 + LayerNorm + 分布恢复（核心逻辑，和参考代码一致）
            if (
                residual_origin_layer is not None
                and residual_target_layers is not None
                and index_block in residual_target_layers
                and encoder_hidden_states is not None
                and _saved_origin_text is not None
                and residual_weights_tensor is not None
            ):
                # 1. 获取原始层/目标层特征并对齐设备/dtype
                origin_enc = _saved_origin_text.to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype
                )
                target_enc = encoder_hidden_states
                
                # 形状校验（保留严谨性）
                if origin_enc.shape != target_enc.shape:
                    raise ValueError(
                        f"残差形状不匹配：原始层 {origin_enc.shape} vs 目标层 {target_enc.shape}"
                    )
                
                # 2. 获取当前目标层对应的权重（兼容单/多权重）
                tid = residual_target_layers.index(index_block)
                # 权重不足时复用最后一个（容错逻辑）
                if tid >= len(residual_weights_tensor):
                    tid = len(residual_weights_tensor) - 1
                w = residual_weights_tensor[tid].to(
                    device=target_enc.device, dtype=target_enc.dtype
                )

                # 3. Token-wise 标准化（z-score）
                target_norm, target_mean, target_std = self._standardize_tokenwise(target_enc, layer_idx=index_block)
                origin_norm, _, _ = self._standardize_tokenwise(origin_enc, layer_idx=residual_origin_layer)

                # 4. 标准化空间叠加残差
                mixed_norm = target_norm + w * origin_norm

                # 5. LayerNorm稳定分布（修复normalized_shape为tuple）
                mixed_norm_ln = torch.nn.functional.layer_norm(
                    mixed_norm, 
                    normalized_shape=(mixed_norm.shape[-1],),  # 关键修复：int→tuple
                    eps=1e-6  # 和参考代码eps一致
                )

                # 6. 恢复目标层原始分布（保证特征尺度兼容原生逻辑）
                encoder_hidden_states = mixed_norm_ln * target_std + target_mean

            # 执行当前双流块（参数顺序完全对齐FLUX原生）
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # 7. 遍历第二阶段：单流 Transformer 块（无修改）
        for block in self.base_model.single_transformer_blocks:
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

        # 仅保留最终结束提示（如需完全删除，注释/删除以下3行即可）
        # print("\n==================== 【前向传播结束】====================")
        # print(f"最终输出hidden_states形状：{hidden_states.shape}")

        # 9. 返回结果（兼容原生输出格式）
        return Transformer2DModelOutput(sample=hidden_states)