# Copyright 2025 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Dict, Any, Union, Tuple
import torch
import torch.nn as nn

from diffusers import SD3Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph


logger = logging.get_logger(__name__)


@maybe_allow_in_graph
class SD35Transformer2DModel_RES(SD3Transformer2DModel):
    """
    SD3.5 残差增强版 Transformer（融合 FLUX 标准化残差逻辑）
    核心改进：
    1. 最大化复用原生父类逻辑，精简冗余代码，增强参数鲁棒性
    2. 融合 FLUX Token-wise z-score 标准化残差（叠加更稳定，避免分布漂移）
    3. 支持单/多权重，兼容整数/浮点/列表/Tensor 权重输入
    """

    def __init__(self, base_model: SD3Transformer2DModel):
        """
        初始化优化：完全复用base_model的配置和权重，删除冗余属性
        Args:
            base_model: 原生 SD3Transformer2DModel 实例（已加载权重）
        """
        # 1. 直接复用base_model的config，无需手动传递所有参数（父类ConfigMixin自动继承）
        super().__init__(**base_model.config)
        
        # 2. 复用base_model所有权重（一行代码替代手动传参+load_state_dict）
        self.load_state_dict(base_model.state_dict(), strict=True)
        
        # 3. 仅保留必要的属性同步（依赖父类继承，删除冗余的base_model引用）
        self.gradient_checkpointing = base_model.gradient_checkpointing
        
        self.to(base_model.device, dtype=base_model.dtype)

    @staticmethod
    def _standardize_tokenwise(x: torch.Tensor, eps: float = 1e-6, layer_idx: int = -1):
        """
        对齐 FLUX 逻辑：对最后一维 (hidden_dim) 做 token-wise z-score 标准化
        支持：
        - 3维特征：[batch_size, seq_len, hidden_dim]（SD35 主流场景）
        - 4维特征：[batch_size, num_img, seq_len, hidden_dim]（兼容扩展场景）
        返回：标准化特征 + 原始均值 + 原始标准差（用于恢复分布）
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
        # -------------------------- 完全对齐原生参数顺序，无新增位置参数 --------------------------
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        output_text_inputs: bool = False,
        # -------------------------- 残差参数（兼容FLUX多权重逻辑，默认禁用状态） --------------------------
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weights: Optional[Union[List[float], torch.Tensor, float, int]] = 0.0,  # 扩展类型支持
        residual_use_layernorm: bool = True,
        residual_rotation_matrices: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        核心优化点：
        1. 预校验残差参数，无效时直接走原生流程
        2. 缓存仅在残差启用时创建，节省内存
        3. 复用父类所有预处理/后处理逻辑，删除冗余复制
        4. 融合FLUX标准化残差：z-score → 叠加 → LayerNorm → 恢复分布
        5. 支持单/多权重，兼容多种权重输入类型
        """
        # -------------------------- 第一步：残差参数预校验（提前过滤无效场景） --------------------------
        # 标记是否启用残差（所有参数均有效才开启）
        use_residual = (
            residual_origin_layer is not None
            and residual_target_layers is not None
            and len(residual_target_layers) > 0
            and residual_weights != 0.0
            and encoder_hidden_states is not None
        )
        
        # 校验参数合法性（提前抛出明确错误，避免运行时异常）
        if use_residual:
            if not isinstance(residual_target_layers, list) or not all(isinstance(x, int) for x in residual_target_layers):
                raise ValueError("residual_target_layers 必须是整数列表")
            if not isinstance(residual_origin_layer, int) or residual_origin_layer < 0:
                raise ValueError(f"residual_origin_layer 必须是非负整数（当前：{residual_origin_layer}）")
            if residual_origin_layer >= self.config.num_layers:
                raise ValueError(f"residual_origin_layer 超出最大层数（0~{self.config.num_layers-1}，当前：{residual_origin_layer}）")
            if any(x < 0 or x >= self.config.num_layers for x in residual_target_layers):
                raise ValueError(f"residual_target_layers 元素必须在 0~{self.config.num_layers-1} 范围内")
        
        # 残差目标层转为集合（O(1)查询，优化循环效率）
        target_layers_set = set(residual_target_layers) if use_residual else set()

        # -------------------------- 新增：预处理残差权重（对齐FLUX逻辑） --------------------------
        residual_weights_tensor = None
        if use_residual and residual_weights is not None:
            # 情况1：列表/元组 → 转为tensor
            if isinstance(residual_weights, (list, tuple)):
                residual_weights_tensor = torch.tensor(residual_weights, dtype=torch.float32)
            # 情况2：Python原生int/float → 转为长度1的tensor
            elif isinstance(residual_weights, (int, float)):
                residual_weights_tensor = torch.tensor([float(residual_weights)], dtype=torch.float32)
            # 情况3：torch.Tensor → 转为float32
            elif isinstance(residual_weights, torch.Tensor):
                residual_weights_tensor = residual_weights.float()
            # 其他情况 → 报错提示
            else:
                raise TypeError(f"不支持的权重类型：{type(residual_weights)}，仅支持list/tuple/int/float/torch.Tensor")

        residual_rotations = None
        if use_residual and residual_rotation_matrices is not None:
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
            if residual_target_layers is not None and residual_rotations.shape[0] != len(residual_target_layers):
                raise ValueError(
                    "residual_rotation_matrices length must match residual_target_layers."
                )
        
        # -------------------------- 第二步：复用父类原生前处理逻辑 --------------------------
        # 1. 原生前处理（完全复用父类代码，无修改）
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        
        # IP Adapter处理（复用原生逻辑，无修改）
        if joint_attention_kwargs and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)
            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)
        
        # -------------------------- 第三步：核心循环（注入FLUX标准化残差逻辑） --------------------------
        # 仅在残差启用时初始化缓存（节省内存）
        pre_encoder_states = [] if use_residual else None
        txt_input_states_list = [] if output_text_inputs else None
        
        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = skip_layers is not None and index_block in skip_layers
            
            if output_text_inputs and not is_skip:
                txt_input_states_list.append(encoder_hidden_states)

            # 残差缓存：保存当前层输入前的文本流特征
            if use_residual and encoder_hidden_states is not None:
                # 仅缓存到origin_layer（后续层无需缓存，减少内存占用）
                if index_block <= residual_origin_layer:
                    pre_encoder_states.append(encoder_hidden_states.detach().clone())
                # origin_layer之后停止缓存，释放内存
                elif len(pre_encoder_states) > residual_origin_layer + 1:
                    pre_encoder_states = pre_encoder_states[:residual_origin_layer + 1]
            
            # -------------------------- 核心修改：FLUX标准化残差叠加 --------------------------
            if use_residual and index_block in target_layers_set:
                # 确保原始层特征已缓存（index_block >= origin_layer时才可能叠加）
                if residual_origin_layer >= len(pre_encoder_states):
                    logger.warning(f"原始层{residual_origin_layer}未缓存，跳过该目标层{index_block}的残差叠加")
                    pass
                else:
                    origin_enc = pre_encoder_states[residual_origin_layer]
                    # 自动对齐设备和dtype（依赖自身属性，更可靠）
                    origin_enc = origin_enc.to(device=self.device, dtype=self.dtype)
                    target_enc = encoder_hidden_states
                    
                    # 形状校验（保留，避免维度不匹配）
                    if origin_enc.shape != target_enc.shape:
                        raise ValueError(
                            f"残差形状不匹配：原始层{residual_origin_layer}（{origin_enc.shape}）vs "
                            f"目标层{index_block}（{target_enc.shape}）"
                        )
                    
                    # 1. 获取当前目标层对应的权重（兼容单/多权重）
                    tid = residual_target_layers.index(index_block)
                    # 权重不足时复用最后一个（容错逻辑）
                    if tid >= len(residual_weights_tensor):
                        tid = len(residual_weights_tensor) - 1
                    w = residual_weights_tensor[tid].to(device=self.device, dtype=self.dtype)

                    # 2. Token-wise 标准化（z-score）- 对齐FLUX逻辑
                    target_norm, target_mean, target_std = self._standardize_tokenwise(
                        target_enc,
                        layer_idx=index_block,
                    )
                    origin_norm, _, _ = self._standardize_tokenwise(
                        origin_enc,
                        layer_idx=residual_origin_layer,
                    )
                    if residual_rotations is not None:
                        rotation = residual_rotations[tid]
                        origin_norm = torch.matmul(origin_norm, rotation)

                    # 3. 标准化空间叠加残差
                    mixed_norm = target_norm + w * origin_norm

                    # 4. LayerNorm稳定分布（修复normalized_shape为tuple，对齐FLUX）
                    if residual_use_layernorm:
                        mixed_norm = torch.nn.functional.layer_norm(
                            mixed_norm,
                            normalized_shape=(mixed_norm.shape[-1],),  # 关键：int→tuple
                            eps=1e-6,
                        )

                    # 5. 恢复目标层原始分布（保证特征尺度兼容原生逻辑）
                    encoder_hidden_states = mixed_norm * target_std + target_mean
            
            # -------------------------- 复用原生块计算逻辑 --------------------------
            if not is_skip:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states, temb, joint_attention_kwargs
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
            
            # -------------------------- 复用ControlNet残差逻辑 --------------------------
            if block_controlnet_hidden_states is not None and not block.context_pre_only:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]
        
        # -------------------------- 第四步：复用父类原生后处理逻辑 --------------------------
        # 1. 输出归一化+投影（复用原生逻辑）
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        
        # 2. Unpatchify（还原图像尺寸，完全复用原生代码）
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
        
        # -------------------------- 第五步：兼容原生返回格式 --------------------------
        if not return_dict:
            if output_text_inputs:
                return {"sample": output, "txt_input_states": txt_input_states_list}
            return (output,)
        return Transformer2DModelOutput(sample=output)
