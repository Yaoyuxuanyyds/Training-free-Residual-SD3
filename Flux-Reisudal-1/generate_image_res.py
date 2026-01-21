import torch
from typing import Optional, List, Dict, Any
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline as OriginalFluxPipeline,
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    XLA_AVAILABLE,
)
if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm

# 导入你的残差Transformer
from flux_transformer_res import FluxTransformer2DModel_RES


class FluxPipelineWithRES(OriginalFluxPipeline):
    def __call__(self, *args, 
                 residual_target_layers: Optional[List[int]] = None,
                 residual_origin_layer: Optional[int] = None,
                 residual_weight: float = 1.0,** kwargs):
        """
        重写__call__方法，核心功能：
        1. 接收残差参数并注入Transformer调用
        2. 兼容Flux原生所有功能（真CFG、IP Adapter、长文本支持等）
        3. 复用父类核心逻辑，避免重复代码
        """
        # -------------------------- 1. 初始化残差参数（核心新增）--------------------------
        self.residual_target_layers = residual_target_layers  # 残差目标层（双流块索引）
        self.residual_origin_layer = residual_origin_layer    # 残差原始层（双流块索引）
        self.residual_weight = residual_weight                # 残差叠加权重

        # -------------------------- 2. 输入校验（过滤无关参数，避免报错）--------------------------
        # 提取check_inputs支持的参数（避免传递guidance_scale等不支持的参数）
        prompt = args[0] if len(args) > 0 else kwargs.get("prompt")
        prompt_2 = args[1] if len(args) > 1 else kwargs.get("prompt_2")
        check_inputs_kwargs = {
            "prompt": prompt,
            "prompt_2": prompt_2,
            "height": kwargs.get("height"),
            "width": kwargs.get("width"),
            "negative_prompt": kwargs.get("negative_prompt"),
            "negative_prompt_2": kwargs.get("negative_prompt_2"),
            "prompt_embeds": kwargs.get("prompt_embeds"),
            "negative_prompt_embeds": kwargs.get("negative_prompt_embeds"),
            "pooled_prompt_embeds": kwargs.get("pooled_prompt_embeds"),
            "negative_pooled_prompt_embeds": kwargs.get("negative_pooled_prompt_embeds"),
            "callback_on_step_end_tensor_inputs": kwargs.get("callback_on_step_end_tensor_inputs"),
            "max_sequence_length": kwargs.get("max_sequence_length"),
        }
        # 调用原生输入校验（确保prompt、尺寸等参数合法）
        self.check_inputs(** check_inputs_kwargs)

        # -------------------------- 3. 基础参数准备（尺寸、上下文）--------------------------
        # 确定生成图像尺寸（默认使用模型默认尺寸×vae缩放因子）
        height = kwargs.get("height") or self.default_sample_size * self.vae_scale_factor
        width = kwargs.get("width") or self.default_sample_size * self.vae_scale_factor

        # 准备全局上下文（batch_size、设备、引导强度等，关键：传*args确保prompt提取正确）
        self._prepare_call_context(*args, **kwargs)

        # -------------------------- 4. 核心预处理（复用父类逻辑）--------------------------
        # 替换 __call__ 里的 encode_prompt 调用，精准传参，仅3行核心代码！
        # 1. 从 args/kwargs 提取源码支持的核心参数（prompt/prompt_2）
        prompt = args[0] if len(args) > 0 else kwargs.get("prompt")
        prompt_2 = args[1] if len(args) > 1 else kwargs.get("prompt_2")  # 源码默认None，可省略（但保留更灵活）

        # 2. 精准调用 encode_prompt，只传源码支持的参数（完全匹配源码定义）
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=self._execution_device,  # 源码支持的device参数，复用上下文设备
            num_images_per_prompt=kwargs.get("num_images_per_prompt", 1),  # 源码默认1，按需从kwargs取
            prompt_embeds=kwargs.get("prompt_embeds"),  # 源码支持的预生成embeds（可选）
            pooled_prompt_embeds=kwargs.get("pooled_prompt_embeds"),  # 源码支持（可选）
            max_sequence_length=kwargs.get("max_sequence_length", 512),  # 源码默认512
            lora_scale=kwargs.get("lora_scale"),  # 源码支持的Lora缩放（可选）
        )
                
        # 计算潜变量通道数（Flux原生逻辑：in_channels//4）
        num_channels_latents = self.transformer.config.in_channels // 4
        
        # 生成初始潜变量（扩散模型起点：随机噪声）和图像位置ID
        latents, latent_image_ids = self.prepare_latents(
            self.batch_size, num_channels_latents, height, width,
            prompt_embeds.dtype, self._execution_device, kwargs.get("generator")
        )
        
        # 计算shift参数（用于timesteps调整，Flux原生逻辑）
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        
        # 准备timesteps（去噪步骤序列）
        sigmas = kwargs.get("sigmas")
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, kwargs.get("num_inference_steps", 28),
            self._execution_device, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # -------------------------- 5. 高级功能准备（guidance、IP Adapter、真CFG）--------------------------
        # 准备guidance参数（文本引导强度）
        guidance = self._prepare_guidance(**kwargs)
        
        # 准备IP Adapter嵌入（图像引导，若启用）
        image_embeds, negative_image_embeds = self._prepare_ip_adapter(** kwargs)
        
        # 准备真CFG参数（负样本引导，若启用）
        do_true_cfg, negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = self._prepare_true_cfg(** kwargs)

        # -------------------------- 6. 核心去噪循环（注入残差参数）--------------------------
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # 注入IP Adapter图像嵌入（若启用）
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                
                # 时间步格式调整（适配Transformer输入：扩展batch维度+对齐dtype）
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # -------------------------- 关键：条件样本（cond）去噪（注入残差）--------------------------
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,  # Flux原生逻辑：timestep缩放（还原到[0,1]范围）
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self._joint_attention_kwargs,  # 修复：原代码少了下划线，这里修正
                        # 注入残差参数（核心新增）
                        residual_target_layers=self.residual_target_layers,
                        residual_origin_layer=self.residual_origin_layer,
                        residual_weight=self.residual_weight,
                        return_dict=False,
                    )[0]

                # -------------------------- 真CFG：负样本（uncond）去噪（注入残差）--------------------------
                if do_true_cfg:
                    # 注入负样本IP Adapter嵌入（若启用）
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self._joint_attention_kwargs,  # 修复：原代码少了下划线
                            # 负样本也注入残差参数（保持逻辑一致）
                            residual_target_layers=self.residual_target_layers,
                            residual_origin_layer=self.residual_origin_layer,
                            residual_weight=self.residual_weight,
                            return_dict=False,
                        )[0]
                    # 真CFG融合：负样本预测 + 缩放因子×（条件预测-负样本预测）
                    noise_pred = neg_noise_pred + kwargs.get("true_cfg_scale", 1.0) * (noise_pred - neg_noise_pred)

                # -------------------------- 更新潜变量（复用scheduler逻辑）--------------------------
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # 兼容MPS设备的dtype一致性
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                # -------------------------- 回调函数处理（复用逻辑）--------------------------
                self._handle_callback(
                    callback_on_step_end=kwargs.get("callback_on_step_end"),
                    callback_on_step_end_tensor_inputs=kwargs.get("callback_on_step_end_tensor_inputs"),
                    step=i, timestep=t, latents=latents, prompt_embeds=prompt_embeds
                )

                # -------------------------- 进度条更新 + XLA兼容--------------------------
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()  # XLA环境触发计算（普通GPU环境无影响）

        # -------------------------- 7. 解码+后处理（从潜变量生成最终图像）--------------------------
        image = self._decode_and_postprocess(
            latents=latents,
            height=height,
            width=width,
            output_type=kwargs.get("output_type", "pil")
        )

        # -------------------------- 8. 清理+返回结果--------------------------
        self.maybe_free_model_hooks()  # 释放模型钩子，节省显存
        if kwargs.get("return_dict", True):
            return FluxPipelineOutput(images=image)
        else:
            return (image,)

    # -------------------------- 辅助方法（复用+适配，无修改）--------------------------
    def _prepare_call_context(self, *args, **kwargs):
        """准备全局调用上下文：batch_size、设备、引导强度等"""
        if self._execution_device is None:
            self._execution_device = self.device  # 统一运行设备（cuda/CPU）
        
        # 计算batch_size（从prompt或prompt_embeds提取，避免依赖未生成的变量）
        prompt = args[0] if len(args) > 0 else kwargs.get("prompt")
        prompt_embeds = kwargs.get("prompt_embeds")
        
        if prompt_embeds is not None:
            self.batch_size = prompt_embeds.shape[0]
        elif prompt is not None:
            self.batch_size = 1 if isinstance(prompt, str) else len(prompt)
        else:
            raise ValueError("必须提供 `prompt` 或 `prompt_embeds` 来计算 batch_size")
        
        # 初始化其他上下文参数
        self._joint_attention_kwargs = kwargs.get("joint_attention_kwargs", {})
        self._guidance_scale = kwargs.get("guidance_scale", 3.5)
        self._interrupt = False
        self._current_timestep = None

    def _prepare_guidance(self, **kwargs):
        """准备guidance参数（文本引导强度）"""
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], self._guidance_scale, device=self._execution_device, dtype=torch.float32)
            return guidance.expand(self.batch_size)
        return None

    def _prepare_ip_adapter(self, **kwargs):
        """准备IP Adapter图像嵌入（图像引导）"""
        ip_adapter_image = kwargs.get("ip_adapter_image")
        ip_adapter_image_embeds = kwargs.get("ip_adapter_image_embeds")
        negative_ip_adapter_image = kwargs.get("negative_ip_adapter_image")
        negative_ip_adapter_image_embeds = kwargs.get("negative_ip_adapter_image_embeds")

        # 生成IP Adapter嵌入（复用父类逻辑）
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image, ip_adapter_image_embeds, self._execution_device, self.batch_size
        ) if (ip_adapter_image or ip_adapter_image_embeds) else None
        negative_image_embeds = self.prepare_ip_adapter_image_embeds(
            negative_ip_adapter_image, negative_ip_adapter_image_embeds, self._execution_device, self.batch_size
        ) if (negative_ip_adapter_image or negative_ip_adapter_image_embeds) else None
        return image_embeds, negative_image_embeds

    def _prepare_true_cfg(self, **kwargs):
        """准备真CFG参数（负样本引导）"""
        true_cfg_scale = kwargs.get("true_cfg_scale", 1.0)
        negative_prompt = kwargs.get("negative_prompt")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        negative_pooled_prompt_embeds = kwargs.get("negative_pooled_prompt_embeds")
        has_neg_prompt = negative_prompt is not None or (negative_prompt_embeds and negative_pooled_prompt_embeds)
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        if do_true_cfg:
            # 编码负样本prompt（复用父类编码逻辑）
            neg_embeds, neg_pooled, neg_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=kwargs.get("negative_prompt_2"),
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=self._execution_device,
                num_images_per_prompt=kwargs.get("num_images_per_prompt", 1),
                max_sequence_length=kwargs.get("max_sequence_length", 512),
            )
            return do_true_cfg, neg_embeds, neg_pooled, neg_text_ids
        return do_true_cfg, None, None, None

    def _decode_and_postprocess(self, latents, height, width, output_type):
        """从潜变量解码为最终图像（复用父类逻辑）"""
        if output_type == "latent":
            return latents
        # 潜变量解包（Flux原生逻辑：适配VAE输入尺寸）
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        # VAE解码前的潜变量缩放（还原到VAE期望范围）
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        # VAE解码+后处理（转PIL/张量等）
        image = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)

    def _handle_callback(self, callback_on_step_end, callback_on_step_end_tensor_inputs, **kwargs):
        """处理回调函数（支持中途修改参数）"""
        if callback_on_step_end is None:
            return
        # 提取回调函数需要的张量参数
        callback_kwargs = {k: kwargs[k] for k in callback_on_step_end_tensor_inputs if k in kwargs}
        # 执行回调函数
        callback_outputs = callback_on_step_end(self, kwargs["step"], kwargs["timestep"], callback_kwargs)
        # 应用回调函数的输出（更新参数）
        if callback_outputs:
            for k, v in callback_outputs.items():
                if k in kwargs:
                    kwargs[k] = v


# -------------------------- 生成代码（直接运行）--------------------------
if __name__ == "__main__":
    # 检查GPU可用性（Flux建议用GPU运行）
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用GPU，请检查CUDA配置或使用CPU（不推荐）")

    # 1. 加载Flux模型并替换为带残差的Transformer
    pipe = FluxPipelineWithRES.from_pretrained(
        pretrained_model_name_or_path="/inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev",
        torch_dtype=torch.bfloat16,  # 用bfloat16节省显存且保证精度
        trust_remote_code=True       # 信任远程代码（Flux模型需要）
    )
    # pipe.enable_model_cpu_offload() 
    
    # -------------------------- 关键新增：替换为自定义残差 Transformer --------------------------
    print(f"替换前 Transformer 类型：{type(pipe.transformer)}")  # 输出原生类型：FluxTransformer2DModel
    pipe.transformer = FluxTransformer2DModel_RES(pipe.transformer)  # 替换核心代码！
    print(f"替换后 Transformer 类型：{type(pipe.transformer)}")  # 应输出：FluxTransformer2DModel_RES

    # 验证替换是否成功（失败直接报错，避免后续白跑）
    if not isinstance(pipe.transformer, FluxTransformer2DModel_RES):
        raise RuntimeError("模型替换失败！请检查 flux_transformer_res.py 的导入路径和类定义")

    core_modules = [
        pipe.text_encoder,    # CLIP文本编码器
        pipe.text_encoder_2,  # T5文本编码器
        pipe.transformer,     # 核心残差Transformer
        pipe.vae,             # VAE解码器
    ]
    for module in core_modules:
        if module is not None:  # 避免某些版本没有该模块导致报错
            module.eval()  # 启用推理模式
            for param in module.parameters():
                param.requires_grad = False 
                
    pipe.transformer.eval().requires_grad_(False)  # 推理模式，禁用梯度计算
    pipe.to("cuda")  # 移动模型到GPU

    # 3. 采样参数配置
    prompt = "A cat holding a sign that says hello world"  # 生成提示词
    residual_params = {  # 残差参数配置
        "residual_target_layers": [11, 12, 13, 14, 15],  # 要叠加残差的双流块索引
        "residual_origin_layer": 2,                   # 提供残差的原始双流块索引
        "residual_weight": 0.25                      # 残差强度（可根据效果调整）
    }
    seeds = [0]  # 随机种子（可扩展为多个种子生成多张图）

    # 4. 循环生成图像
    for i, seed in enumerate(seeds, start=1):
        # 设置随机种子（保证结果可复现）
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # 调用Pipeline生成图像
        image = pipe(
            prompt=prompt,                # 提示词
            height=1024,                  # 生成图像高度
            width=1024,                   # 生成图像宽度
            guidance_scale=3.5,           # 文本引导强度（默认3.5）
            num_inference_steps=50,       # 去噪步骤（默认50，可调整）
            max_sequence_length=256,      # 最大文本长度（T5编码器支持）
            generator=generator,          # 随机种子生成器
            **residual_params             # 传递残差参数
        ).images[0]  # 取第一张图（batch_size=1时直接取）

        # 保存图像
        image_path = f"flux_residual_{i}.png"
        image.save(image_path)
        print(f"已保存第{i}张图像：{image_path}")
        
        torch.cuda.empty_cache() 