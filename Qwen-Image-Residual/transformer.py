import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers.models import QwenImageTransformer2DModel
from diffusers.utils import logging
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)


class MyQwenImageTransformer2DModel(QwenImageTransformer2DModel):
    """
    增加 text stream 的跨层 residual 注入功能
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.residual_origin_layer: Optional[int] = None
        self.residual_target_layers: List[int] = []
        self.residual_weights: Optional[torch.Tensor] = None
        self.residual_use_layernorm: bool = True
        self.residual_stop_grad: bool = True
        self.residual_rotation_matrices: Optional[Union[List[torch.Tensor], torch.Tensor]] = None

        self._saved_origin_text: Optional[torch.Tensor] = None

    @staticmethod
    def _standardize_tokenwise(x: torch.Tensor, eps: float = 1e-6):
        """
        对最后一维 (hidden_dim) 做 z-score：
        x_norm = (x - mean) / (std + eps)
        返回 (x_norm, mean, std)，方便之后把 scale/shift 加回去。
        形状保持和 x 一致。
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + eps)
        return x_norm, mean, std

    def _apply_residual(
        self,
        target: torch.Tensor,
        origin: torch.Tensor,
        w: torch.Tensor,
        use_layernorm: bool = True,
        stop_grad: bool = True,
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
            mixed_norm = target_norm + w * origin_norm
        else:
            mixed_norm = target_norm * (1 - w)

        if use_layernorm:
            mixed_norm = torch.nn.functional.layer_norm(
                mixed_norm,
                normalized_shape=mixed_norm.shape[-1:],
                eps=1e-6,
            )

        return mixed_norm * target_std + target_mean

    def set_residual_config(
        self,
        residual_origin_layer: Optional[int],
        residual_target_layers: Optional[Union[List[int], torch.Tensor]],
        residual_weights: Optional[Union[List[float], torch.Tensor]],
        residual_use_layernorm: bool = True,
        residual_stop_grad: bool = True,
        residual_rotation_matrices: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ):
        if residual_origin_layer is None:
            self.residual_origin_layer = None
            self.residual_target_layers = []
            self.residual_weights = None
            self.residual_rotation_matrices = None
            self._saved_origin_text = None
            return

        # layers
        if isinstance(residual_target_layers, torch.Tensor):
            residual_target_layers = residual_target_layers.tolist()
        self.residual_target_layers = [int(i) for i in residual_target_layers]

        # weights
        if isinstance(residual_weights, (list, tuple)):
            residual_weights = torch.tensor(residual_weights, dtype=torch.float32)
        self.residual_weights = residual_weights

        self.residual_origin_layer = int(residual_origin_layer)
        self.residual_use_layernorm = bool(residual_use_layernorm)
        self.residual_stop_grad = bool(residual_stop_grad)
        self.residual_rotation_matrices = residual_rotation_matrices
        self._saved_origin_text = None

        logger.info(
            f"[Residual] origin={self.residual_origin_layer}, "
            f"targets={self.residual_target_layers}, "
            f"weights={self.residual_weights}"
        )

    def set_residual_weights(self, residual_weights: Optional[Union[List[float], torch.Tensor]]):
        if residual_weights is None:
            self.residual_weights = None
            return
        if isinstance(residual_weights, (list, tuple)):
            residual_weights = torch.tensor(residual_weights, dtype=torch.float32)
        self.residual_weights = residual_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        target_layers: Optional[List[int]] = None,
        output_text_inputs: bool = False,
    ):

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        collect_txt_features = target_layers is not None and len(target_layers) > 0
        target_layers = sorted(set(target_layers)) if collect_txt_features else []
        txt_feats_list: List[torch.Tensor] = []
        txt_input_states_list: List[torch.Tensor] = []
        context_embedder_output: List[torch.Tensor] = []

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        use_residual = (
            self.residual_origin_layer is not None
            and self.residual_weights is not None
            and len(self.residual_target_layers) > 0
        )

        residual_rotations = None
        if use_residual and self.residual_rotation_matrices is not None:
            if isinstance(self.residual_rotation_matrices, (list, tuple)):
                if all(torch.is_tensor(r) for r in self.residual_rotation_matrices):
                    residual_rotations = torch.stack(self.residual_rotation_matrices, dim=0)
                else:
                    residual_rotations = torch.tensor(self.residual_rotation_matrices)
            elif torch.is_tensor(self.residual_rotation_matrices):
                residual_rotations = self.residual_rotation_matrices
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
            if residual_rotations.shape[0] != len(self.residual_target_layers):
                raise ValueError(
                    "residual_rotation_matrices length must match residual_target_layers."
                )
            if residual_rotations.shape[-1] != encoder_hidden_states.shape[-1] or \
                residual_rotations.shape[-2] != encoder_hidden_states.shape[-1]:
                raise ValueError(
                    "residual_rotation_matrices feature dimension must match encoder_hidden_states."
                )

        target_set = set(self.residual_target_layers)
        pre_encoder_states: List[torch.Tensor] = []

        if collect_txt_features:
            context_embedder_output.append(encoder_hidden_states.detach())

        for layer_idx, block in enumerate(self.transformer_blocks):
            if output_text_inputs:
                txt_input_states_list.append(encoder_hidden_states)

            if use_residual:
                pre_encoder_states.append(encoder_hidden_states)

                if layer_idx in target_set:
                    tid = self.residual_target_layers.index(layer_idx)
                    w = self.residual_weights[tid].to(
                        encoder_hidden_states.device, encoder_hidden_states.dtype
                    )

                    if 0 <= self.residual_origin_layer < len(pre_encoder_states):
                        origin = pre_encoder_states[self.residual_origin_layer]
                    else:
                        raise ValueError(
                            f"Invalid residual_origin_layer={self.residual_origin_layer}"
                        )

                    if origin.shape != encoder_hidden_states.shape:
                        raise ValueError(
                            f"[Residual] Shape mismatch: origin={origin.shape} vs target={encoder_hidden_states.shape}"
                        )

                    rotation = residual_rotations[tid] if residual_rotations is not None else None
                    encoder_hidden_states = self._apply_residual(
                        encoder_hidden_states,
                        origin,
                        w,
                        use_layernorm=self.residual_use_layernorm,
                        stop_grad=self.residual_stop_grad,
                        rotation_matrix=rotation,
                    )


            # 3. 正常执行 block
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

            if collect_txt_features and layer_idx in target_layers:
                txt_feats_list.append(encoder_hidden_states.detach())

        # 清空缓存
        self._saved_origin_text = None

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓ 剩余部分保持官方一致 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if collect_txt_features:
            # 当需要文本特征时，总是返回字典，便于可视化脚本读取
            result = {
                "sample": output,
                "txt_feats_list": txt_feats_list,
                "context_embedder_output": context_embedder_output,
            }
            if output_text_inputs:
                result["txt_input_states"] = txt_input_states_list
            return result

        if not return_dict:
            if output_text_inputs:
                return {
                    "sample": output,
                    "txt_input_states": txt_input_states_list,
                }
            return (output,)

        return Transformer2DModelOutput(sample=output)
