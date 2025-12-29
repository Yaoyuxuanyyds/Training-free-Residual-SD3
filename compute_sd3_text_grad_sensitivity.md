# `compute_sd3_text_grad_sensitivity.py` 脚本说明

本文档介绍 `compute_sd3_text_grad_sensitivity.py` 的用途、整体流程、关键函数与参数含义，帮助你理解脚本如何计算 SD3 模型中不同层的文本 token 影响力指标。

## 1. 脚本用途

该脚本用于计算 **文本 token 在 SD3 不同 transformer 层中的影响力**，方法是：

- 对 denoiser 的输出与 **文本隐藏状态** 求梯度；
- 将每个 token 的梯度范数作为影响力分数；
- 对各层的 token 分数计算统计指标：
  - **Mean influence strength**（平均强度）
  - **Top‑k mass**（前 k 个 token 的质量占比）
  - **Entropy**（分布熵）

最后会保存三条曲线图，展示各层指标随层数变化的趋势。

## 2. 核心流程概览

脚本的高层执行流程为：

1. 解析命令行参数并初始化随机种子。
2. 创建 `StableDiffusion3Base` 模型对象并配置精度与 LoRA（可选）。
3. 获取单条图文对或批量数据集样本。
4. 对每个样本：
   - 编码图像到 latent；
   - 编码文本得到 `prompt_emb` 与 `pooled_emb`；
   - 在多个 timestep 和随机种子下采样噪声 latent；
   - 获取指定层的文本隐藏状态，对输出求梯度；
   - 计算 token 影响力分数，并累积统计。
5. 汇总每层指标，输出到终端并保存曲线图。

## 3. 关键函数说明

### 3.1 图像相关函数

- **`load_and_resize_pil(image_source, height, width)`**
  - 支持 PIL / torch.Tensor / 文件路径输入。
  - 返回指定尺寸的 RGB PIL 图像。

- **`pil_to_tensor(pil_img, device)`**
  - 将 PIL 图像转为 `[1, C, H, W]` tensor。

- **`encode_image_to_latent(base, img_tensor)`**
  - 使用 VAE 将图像编码成 latent `z0`。

- **`build_noisy_latent_like_training(scheduler, clean_latent, timestep_idx, generator)`**
  - 根据指定 timestep 线性插值产生噪声 latent `z_t`。
  - 模拟训练时的噪声注入方式。

### 3.2 LoRA 相关函数

- **`_parse_lora_target(target)`**
  - 解析 LoRA 注入的目标模块。

- **`_maybe_apply_lora(...)`**
  - 若提供 LoRA 权重，进行模块注入和权重加载。

### 3.3 指标计算

- **`compute_metrics(scores, topk, warn_prefix="")`**
  - 输入为每个 token 的影响力分数（`scores`）。
  - 输出：
    - mean strength
    - top‑k mass
    - entropy
  - 对 NaN/Inf/空张量做防护处理，避免指标为 NaN。

### 3.4 采样策略

- **`build_timesteps(args, scheduler)`**
  - 生成要评估的 timestep 列表。

- **`build_seed_list(args)`**
  - 生成要评估的随机种子列表。

## 4. `run()` 主函数细节

`run(args)` 是主要执行逻辑，关键步骤如下：

1. **设置随机种子**，确保可复现。
2. **初始化模型**：
   - 使用 `StableDiffusion3Base` 加载模型。
   - 根据 `--precision` 选择 `fp16 / bf16 / fp32 / auto`。
3. **获取样本**：
   - 单条图文对（`--prompt` + `--image`）或
   - 数据集模式（`--dataset`）。
4. **编码文本与图像**：
   - `encode_prompt()` 返回文本 embedding 和 mask。
   - 可通过 `--ignore-padding` 过滤 padding token。
5. **梯度计算**：
   - 获取 denoiser 输出并计算平方和作为目标函数 `y`。
   - 对目标层文本隐藏状态求梯度。
   - 梯度范数作为 token 影响力分数。
6. **统计汇总与输出**：
   - 每层指标取均值。
   - 打印并保存曲线图。

## 5. CLI 参数说明（核心）

| 参数 | 含义 |
|------|------|
| `--prompt` | 单条文本 prompt（单样本模式） |
| `--image` | 单条图像路径（单样本模式） |
| `--dataset` / `--datadir` | 数据集模式输入 |
| `--timestep-idx` / `--num-timesteps` | 评估的 timestep 控制 |
| `--num-seeds` | 每个 timestep 评估多少随机种子 |
| `--layers` | 评估的 transformer 层索引 |
| `--topk` | 统计 top‑k mass 的 k 值 |
| `--precision` | 控制计算精度：`auto|fp16|bf16|fp32` |
| `--ignore-padding` | 使用 mask 忽略 padding token |
| `--force-txt-grad` / `--no-force-txt-grad` | 是否强制文本隐藏状态参与梯度计算（默认开启） |
| `--output-dir` | 输出曲线图保存目录 |

## 6. 输出内容

脚本运行结束后会输出：

- 终端打印每层的三项指标：
  - `strength`
  - `topk_mass`
  - `entropy`
- 保存三张曲线图：
  - `grad_strength_curve.png`
  - `grad_topk_mass_curve.png`
  - `grad_entropy_curve.png`

## 7. 常见注意点

- 使用 `fp16` 时若出现 NaN，可尝试：
  - 设置 `--precision fp32`；
  - 减少 timestep / seeds 以降低数值波动；
  - 确认 `--ignore-padding` 不会使 mask 全部为 False。
- 若单样本模式输出为空，确认 `--prompt` 与 `--image` 同时提供。

---

如需进一步扩展（例如保存每个 token 的原始分数分布、可视化 token 重要性等），可以在 `compute_metrics` 或主循环中加入日志与保存逻辑。
