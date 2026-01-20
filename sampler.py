from typing import Callable, List, Tuple, Optional
import torch
from transformers import BitsAndBytesConfig, T5EncoderModel
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
from transformer import SD3Transformer2DModel_Vanilla, SD3Transformer2DModel_REPA, SD3Transformer2DModel_Residual
from torch.nn.parallel import DistributedDataParallel
from torch import nn
from torch.amp import autocast
from util import set_seed, resolve_rotation_bucket

from typing import Callable, Optional
import torch


def build_timestep_residual_weight_fn(
    name: Optional[str] = "constant",
    power: float = 1.0,
    exp_alpha: float = 1.5,   
) -> Optional[Callable[[torch.Tensor, int], torch.Tensor]]:
    if name is None:
        return None

    name = name.lower()

    def _apply_power(weight: torch.Tensor) -> torch.Tensor:
        if power != 1.0:
            return weight ** power
        return weight

    def constant(timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
        weight = torch.ones_like(timestep, dtype=torch.float32)
        return _apply_power(weight)

    def linear(timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
        weight = timestep.float() / float(num_train_timesteps)
        weight = weight.clamp(0.0, 1.0)
        return _apply_power(weight)

    def cosine(timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
        weight = 0.5 * (1.0 + torch.cos(torch.pi * (1 - timestep.float() / float(num_train_timesteps))))
        return _apply_power(weight.clamp(0.0, 1.0))

    def exponential(timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
        s = timestep.float() / float(num_train_timesteps)
        weight = torch.exp(-exp_alpha * s)
        return _apply_power(weight)

    if name == "constant":
        return constant
    if name == "linear":
        return linear
    if name == "cosine":
        return cosine
    if name in ("exp", "exponential"):
        return exponential

    raise ValueError(f"Unsupported timestep residual weight fn: {name}")


class StableDiffusion3Base():
    def __init__(self,
                 model_key: str = '/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3',
                 device='cuda',
                 dtype=torch.float16,
                 use_8bit=False,
                 load_ckpt_path: Optional[str] = None,
                 load_transformer_only: bool = False,
                 ):
        
        self.device = device
        self.dtype = dtype
        
        keep_on_cpu = load_transformer_only

        # 加载 8bit 模式或正常模式
        if use_8bit:
            print('[INFO] Load 8-bit encoder for text_encoder_3')
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            text_encoder_3 = T5EncoderModel.from_pretrained(
                model_key,
                subfolder='text_encoder_3',
                quantization_config=quant_config,
                torch_dtype=self.dtype,
                device_map={"": "cuda:0"}
            )
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_key,
                text_encoder_3=text_encoder_3,
                torch_dtype=self.dtype
            )
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_key,
                torch_dtype=self.dtype,
            )

        # 加载本地 checkpoint（只加载 transformer；VAE/text_encoders 用预训练）
        if load_ckpt_path is not None:
            ckpt = torch.load(load_ckpt_path, map_location=device)
            if 'transformer' not in ckpt:
                raise KeyError(f"Checkpoint {load_ckpt_path} does not contain 'transformer' key.")

            # 目标 dtype：以 pipe.transformer 的参数 dtype 为准
            tgt_dtype = next(pipe.transformer.parameters()).dtype
            trans_sd = {k: v.to(dtype=tgt_dtype) for k, v in ckpt['transformer'].items()}

            missing, unexpected = pipe.transformer.load_state_dict(trans_sd, strict=False)
            print(f"[INFO] Loaded transformer from {load_ckpt_path}")
            if missing:
                print(f"[WARN] Missing keys when loading transformer: {missing}")
            if unexpected:
                print(f"[WARN] Unexpected keys when loading transformer: {unexpected}")
        
        


        # 保存模块
        self.scheduler = pipe.scheduler
        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3



        self.vae = pipe.vae.to(device)
        self.text_enc_1 = pipe.text_encoder.to(device)
        self.text_enc_2 = pipe.text_encoder_2.to(device)
        self.text_enc_3 = pipe.text_encoder_3.to(device)
        # self.denoiser = SD3Transformer2DModel_Vanilla(pipe.transformer.to(device))
        # self.denoiser = SD3Transformer2DModel_REPA(pipe.transformer.to(device))
        self.denoiser = SD3Transformer2DModel_Residual(pipe.transformer.to(device))


        self.denoiser.eval()
        self.denoiser.requires_grad_(False)
        self.device_diff = next(self.denoiser.parameters()).device   
        
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None else 8
        )

        del pipe

    def offload_text_encoders(self, device: str = "cpu") -> None:
        for name in ("text_enc_1", "text_enc_2", "text_enc_3"):
            encoder = getattr(self, name, None)
            if encoder is None:
                continue
            try:
                encoder.to(device)
            except (RuntimeError, NotImplementedError) as exc:
                print(f"[WARN] Failed to offload {name} to {device}: {exc}")
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_text_encoders(self, device: Optional[str] = None) -> None:
        target_device = device or self.device
        for name in ("text_enc_1", "text_enc_2", "text_enc_3"):
            encoder = getattr(self, name, None)
            if encoder is None:
                continue
            try:
                encoder.to(target_device)
            except (RuntimeError, NotImplementedError) as exc:
                print(f"[WARN] Failed to move {name} to {target_device}: {exc}")

    @torch.no_grad()
    def encode_prompt(self, prompt: List[str], batch_size: int = 1):
        # --- CLIP branch 1 ---
        text_clip1 = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors='pt'
        )
        text_clip1_ids = text_clip1.input_ids
        text_clip1_mask = text_clip1.attention_mask.to(device=self.device_diff)
        text_clip1_emb = self.text_enc_1(text_clip1_ids.to(self.text_enc_1.device), output_hidden_states=True)
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=self.device_diff)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(dtype=self.dtype, device=self.device_diff)

        # --- CLIP branch 2 ---
        text_clip2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors='pt'
        )
        text_clip2_ids = text_clip2.input_ids
        text_clip2_mask = text_clip2.attention_mask.to(device=self.device_diff)
        text_clip2_emb = self.text_enc_2(text_clip2_ids.to(self.text_enc_2.device), output_hidden_states=True)
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=self.device_diff)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(dtype=self.dtype, device=self.device_diff)

        # --- T5 branch ---
        text_t5 = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        text_t5_ids = text_t5.input_ids
        text_t5_mask = text_t5.attention_mask.to(device=self.device_diff)
        text_t5_emb = self.text_enc_3(text_t5_ids.to(self.text_enc_3.device))[0].to(dtype=self.dtype, device=self.device_diff)

        # --- Combine embeddings ---
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1]))
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        # --- Combine masks ---
        clip_mask = torch.logical_or(text_clip1_mask.bool(), text_clip2_mask.bool())  # [B,77]
        text_mask = torch.cat([clip_mask, text_t5_mask.bool()], dim=1)                # [B, 77+256]

        return prompt_emb, pooled_prompt_emb, text_mask


    def initialize_latent(self, img_size: Tuple[int], batch_size: int = 1):
        H, W = img_size
        lH, lW = H // self.vae_scale_factor, W // self.vae_scale_factor

        # >>> 兼容 DP / 非 DP，兼容有无 base_model / config 的情况 <<<
        den = getattr(self.denoiser, "module", self.denoiser)        # unwrap DataParallel
        base = getattr(den, "base_model", den)                       # 有些实现把真正的 transformer 放在 base_model
        base = getattr(base, "module", base)                         # 再次防御性 unwrap

        lC = None
        # 优先从 config 里取
        if hasattr(base, "config") and hasattr(base.config, "in_channels"):
            lC = int(base.config.in_channels)
        # 有些实现直接把 in_channels 挂在模块上
        elif hasattr(base, "in_channels"):
            lC = int(base.in_channels)
        else:
            # 再尝试一层常见命名
            inner = getattr(base, "model", None) or getattr(base, "transformer", None)
            inner = getattr(inner, "module", inner)
            if inner is not None and hasattr(inner, "config") and hasattr(inner.config, "in_channels"):
                lC = int(inner.config.in_channels)

        if lC is None:
            raise AttributeError(
                "Cannot resolve denoiser's in_channels. Please expose `.config.in_channels` "
                "or `.in_channels` on your (wrapped) transformer."
            )

        latent_shape = (batch_size, lC, lH, lW)
        z = torch.randn(latent_shape, device=self.device_diff, dtype=self.dtype)
        return z


    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z.to(self.device_diff)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    # def predict_vector(self, z, t, prompt_emb, pooled_emb):
    #     return self.denoiser(
    #         hidden_states=z,
    #         timestep=t,
    #         pooled_projections=pooled_emb,
    #         encoder_hidden_states=prompt_emb,
    #         return_dict=False
    #     )[0]

    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        with autocast('cuda', enabled=(self.dtype == torch.float16 and torch.cuda.is_available())):
            v = self.denoiser(
                hidden_states=z,
                timestep=t,
                pooled_projections=pooled_emb,
                encoder_hidden_states=prompt_emb,
                return_dict=False
            )['sample']
        return v
    
    def predict_vector_residual(
        self, z, t, prompt_emb, pooled_emb,
        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weights: Optional[List[float]] = None,
        residual_use_layernorm: bool = True,    # ⭐ 新增
        residual_rotation_matrices: Optional[torch.Tensor] = None,
    ):
        with autocast('cuda', enabled=(self.dtype == torch.float16 and torch.cuda.is_available())):
            v = self.denoiser(
                hidden_states=z,
                timestep=t,
                pooled_projections=pooled_emb,
                encoder_hidden_states=prompt_emb,
                return_dict=False,
                residual_target_layers=residual_target_layers,
                residual_origin_layer=residual_origin_layer,
                residual_weights=residual_weights,
                residual_use_layernorm=residual_use_layernorm,   # ⭐ Forward 参数传递
                residual_rotation_matrices=residual_rotation_matrices,
            )['sample']
        return v


class SD3Euler(StableDiffusion3Base):
    def __init__(self, model_key='/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3', device='cuda', use_8bit=False, load_ckpt_path=None, load_transformer_only: bool = False):
        super().__init__(model_key=model_key, device=device, use_8bit=use_8bit, load_ckpt_path=load_ckpt_path, load_transformer_only=load_transformer_only)

    @staticmethod
    def _resolve_timestep_residual_weight(
        timestep: torch.Tensor,
        num_train_timesteps: int,
        weight_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if weight_fn is None:
            return None
        try:
            weight = weight_fn(timestep, num_train_timesteps)
        except TypeError:
            weight = weight_fn(timestep)

        if not torch.is_tensor(weight):
            weight = torch.tensor(weight, device=timestep.device, dtype=timestep.dtype)
        else:
            weight = weight.to(device=timestep.device, dtype=timestep.dtype)

        if weight.numel() > 1:
            weight = weight.reshape(-1)[0]
        return weight

    @staticmethod
    def _scale_residual_weights(
        residual_weights: Optional[List[float]],
        timestep_weight: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if residual_weights is None:
            return None
        if timestep_weight is None:
            if isinstance(residual_weights, torch.Tensor):
                return residual_weights.to(device=device, dtype=dtype)
            return torch.tensor(residual_weights, device=device, dtype=dtype)

        if isinstance(residual_weights, torch.Tensor):
            weights = residual_weights.to(device=device, dtype=dtype)
        else:
            weights = torch.tensor(residual_weights, device=device, dtype=dtype)
        return weights * timestep_weight

    def inversion(self, src_img, prompts: List[str], NFE: int, cfg_scale: float = 1.0, batch_size: int = 1):
        prompt_emb, pooled_emb, _ = self.encode_prompt(prompts, batch_size)
        null_prompt_emb, null_pooled_emb, _ = self.encode_prompt([""], batch_size)

        src_img = src_img.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img)

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.device)])
        timesteps = reversed(timesteps)
        steps = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(timesteps[:-1], total=NFE, desc='SD3 Euler Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb) if cfg_scale != 1.0 else 0.0
            step = steps[i]
            step_next = steps[i + 1]
            z = z + (step_next - step) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))
        return z

    def sample(self, prompts: List[str], NFE: int, img_shape: Optional[Tuple[int]] = None, cfg_scale: float = 1.0, batch_size: int = 1, latent: Optional[torch.Tensor] = None):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        with torch.no_grad():
            prompt_emb, pooled_emb, _ = self.encode_prompt(prompts, batch_size)
            null_prompt_emb, null_pooled_emb, _ = self.encode_prompt([""]*batch_size, batch_size)
        z = self.initialize_latent((imgH, imgW), batch_size) if latent is None else latent

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        steps = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(timesteps, total=NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)           # exp 
            pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb) if cfg_scale != 1.0 else 0.0
            step = steps[i]
            step_next = steps[i + 1] if i + 1 < NFE else 0.0
            z = z + (step_next - step) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))
        with torch.no_grad():
            img = self.decode(z)
        return img
    
    def sample_residual(
        self, prompts: List[str], NFE: int,
        img_shape: Optional[Tuple[int]] = None,
        cfg_scale: float = 1.0,
        batch_size: int = 1,
        latent: Optional[torch.Tensor] = None,

        residual_target_layers: Optional[List[int]] = None,
        residual_origin_layer: Optional[int] = None,
        residual_weights: Optional[List[float]] = None,
        residual_use_layernorm: bool = True,  # ⭐ 新增
        residual_rotation_matrices: Optional[torch.Tensor] = None,
        residual_rotation_meta: Optional[dict] = None,
        residual_timestep_weight_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    ):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        with torch.no_grad():
            prompt_emb, pooled_emb, _ = self.encode_prompt(prompts, batch_size)
            null_prompt_emb, null_pooled_emb, _ = self.encode_prompt([""]*batch_size, batch_size)
        z = self.initialize_latent((imgH, imgW), batch_size) if latent is None else latent

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        steps = timesteps / self.scheduler.config.num_train_timesteps
        weight_fn = None
        if residual_weights is not None:
            weight_fn = residual_timestep_weight_fn or build_timestep_residual_weight_fn()

        pbar = tqdm(timesteps, total=NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            timestep_weight = self._resolve_timestep_residual_weight(
                timestep,
                self.scheduler.config.num_train_timesteps,
                weight_fn,
            )
            effective_residual_weights = self._scale_residual_weights(
                residual_weights,
                timestep_weight,
                device=self.device_diff,
                dtype=self.dtype,
            )
            selected_rotations = resolve_rotation_bucket(
                residual_rotation_matrices,
                residual_rotation_meta,
                timestep,
            )

            pred_v = self.predict_vector_residual(
                z, timestep, prompt_emb, pooled_emb,
                residual_target_layers=residual_target_layers,
                residual_origin_layer=residual_origin_layer,
                residual_weights=effective_residual_weights,
                residual_use_layernorm=residual_use_layernorm,  # ⭐ 传递
                residual_rotation_matrices=selected_rotations,
            )

            pred_null_v = (
                self.predict_vector_residual(
                    z, timestep, null_prompt_emb, null_pooled_emb,
                    residual_target_layers=residual_target_layers,
                    residual_origin_layer=residual_origin_layer,
                    residual_weights=effective_residual_weights,
                    residual_use_layernorm=residual_use_layernorm,  # ⭐ 传递
                    residual_rotation_matrices=selected_rotations,
                )
                if cfg_scale != 1.0 else 0.0
            )

            step = steps[i]
            step_next = steps[i + 1] if i + 1 < NFE else 0.0
            z = z + (step_next - step) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        with torch.no_grad():
            img = self.decode(z)
        return img
