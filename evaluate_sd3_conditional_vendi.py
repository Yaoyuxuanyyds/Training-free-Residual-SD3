from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import inception_v3


_INCEPTION_CACHE: dict[str, nn.Module] = {}
DEFAULT_INCEPTION_WEIGHTS_PATH = (
    "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/torchvision/"
    "inception_v3/inception_v3_google-0cc3c7bd.pth"
)


@dataclass(frozen=True)
class PromptEntry:
    global_index: int
    image_id: int
    prompt: str


def clean_prompt_for_filename(prompt: str, max_length: int = 80) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", prompt.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "empty_prompt"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("._")
    return cleaned


def chunked(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, but got {chunk_size}.")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    if img_tensor.dim() == 4:
        if img_tensor.shape[0] != 1:
            raise ValueError(f"Expected a single image tensor, but got shape {tuple(img_tensor.shape)}.")
        img_tensor = img_tensor[0]

    img_tensor = img_tensor.detach().cpu().float().clamp(-1.0, 1.0)
    img_tensor = (img_tensor + 1.0) / 2.0
    img_tensor = img_tensor.clamp(0.0, 1.0)
    img_array = (img_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(img_array)


def build_prompt_output_dir(output_root: str | Path, entry: PromptEntry) -> Path:
    prompt_tag = clean_prompt_for_filename(entry.prompt, max_length=80)
    dirname = f"prompt_{entry.global_index:05d}_img{entry.image_id:012d}_{prompt_tag}"
    return Path(output_root) / "per_prompt" / dirname


def prepare_seeds(
    num_images_per_prompt: int,
    seeds: Sequence[int] | None = None,
    seed_offset: int = 42,
) -> list[int]:
    if num_images_per_prompt <= 0:
        raise ValueError(
            f"num_images_per_prompt must be positive, but got {num_images_per_prompt}."
        )
    if seeds is None or len(seeds) == 0:
        return [seed_offset + i for i in range(num_images_per_prompt)]

    resolved = [int(seed) for seed in seeds]
    if len(resolved) < num_images_per_prompt:
        raise ValueError(
            "The provided seeds are fewer than num_images_per_prompt: "
            f"{len(resolved)} < {num_images_per_prompt}."
        )
    return resolved[:num_images_per_prompt]


def load_coco_prompts(datadir: str, num_prompts: int) -> list[PromptEntry]:
    from pycocotools.coco import COCO

    ann_path = Path(datadir) / "coco" / "annotations" / "captions_val2017.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO caption annotation file not found: {ann_path}")

    coco = COCO(str(ann_path))
    image_ids = sorted(coco.imgs.keys())
    total_available = len(image_ids)

    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, but got {num_prompts}.")
    if num_prompts > total_available:
        raise ValueError(
            f"Requested num_prompts={num_prompts}, but COCO val2017 only has {total_available} images."
        )

    entries: list[PromptEntry] = []
    for global_index, image_id in enumerate(image_ids[:num_prompts]):
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        if not anns:
            raise RuntimeError(f"No caption annotations found for image_id={image_id}.")

        prompt = str(anns[0]["caption"]).strip()
        if not prompt:
            raise RuntimeError(f"Empty prompt found for image_id={image_id}.")

        entries.append(
            PromptEntry(
                global_index=global_index,
                image_id=int(image_id),
                prompt=prompt,
            )
        )
    return entries


def compute_cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, but got shape {features.shape}.")

    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = features / norms

    similarity = normalized @ normalized.T
    similarity = 0.5 * (similarity + similarity.T)
    similarity = np.clip(similarity, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def compute_vendi_score_from_similarity_matrix(K: np.ndarray) -> tuple[float, np.ndarray]:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix, but got shape {K.shape}.")

    n = K.shape[0]
    if n == 0:
        raise ValueError("K must contain at least one sample.")

    K_tilde = np.asarray(K, dtype=np.float64) / float(n)
    K_tilde = 0.5 * (K_tilde + K_tilde.T)

    eigenvalues = np.linalg.eigvalsh(K_tilde)
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)

    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    vendi_score = float(np.exp(entropy))
    return vendi_score, eigenvalues.astype(np.float64)


def _build_inception_feature_extractor(
    device: str,
    inception_weights_path: str = DEFAULT_INCEPTION_WEIGHTS_PATH,
) -> nn.Module:
    weights_path = Path(inception_weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            "Local Inception-V3 weights not found for offline evaluation: "
            f"{weights_path}"
        )

    model = inception_v3(weights=None, transform_input=False, init_weights=False)
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.fc = nn.Identity()
    model.aux_logits = False
    if hasattr(model, "AuxLogits"):
        model.AuxLogits = None
    model.eval()
    model.to(device)
    return model


def extract_inception_features(
    image_paths: list[str],
    batch_size: int = 32,
    device: str = "cuda",
    inception_weights_path: str = DEFAULT_INCEPTION_WEIGHTS_PATH,
) -> np.ndarray:
    if len(image_paths) == 0:
        raise ValueError("image_paths must not be empty.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, but got {batch_size}.")

    cache_key = f"{device}:{inception_weights_path}"
    if cache_key not in _INCEPTION_CACHE:
        _INCEPTION_CACHE[cache_key] = _build_inception_feature_extractor(
            device=device,
            inception_weights_path=inception_weights_path,
        )
    model = _INCEPTION_CACHE[cache_key]

    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    features: list[np.ndarray] = []
    for batch_paths in chunked(image_paths, batch_size):
        batch_images = []
        for image_path in batch_paths:
            with Image.open(image_path) as image:
                batch_images.append(preprocess(image.convert("RGB")))

        batch = torch.stack(batch_images, dim=0).to(device)
        with torch.no_grad():
            batch_features = model(batch)
            if isinstance(batch_features, tuple):
                batch_features = batch_features[0]
            batch_features = batch_features.view(batch_features.shape[0], -1)
        features.append(batch_features.detach().cpu().float().numpy())

    return np.concatenate(features, axis=0).astype(np.float32)


class SD3ResidualBatchGenerator:
    def __init__(
        self,
        model_dir: str,
        load_ckpt_path: str | None = None,
        device: str = "cuda",
        residual_target_layers: Sequence[int] | None = None,
        residual_origin_layer: int | None = None,
        residual_origin_layers: Sequence[int] | None = None,
        residual_weights: Sequence[float] | torch.Tensor | None = None,
        residual_weights_path: str | None = None,
        residual_procrustes_path: str | None = None,
        residual_use_layernorm: bool = True,
        timestep_residual_weight_fn: str = "constant",
        timestep_residual_weight_power: float = 1.0,
        timestep_residual_weight_exp_alpha: float = 1.5,
        timestep_stage: int = 0,
    ) -> None:
        from sampler import SD3Euler, build_timestep_residual_weight_fn
        from util import (
            load_residual_procrustes,
            load_residual_weights,
            resolve_origin_layers,
            resolve_rotation_bucket,
            select_residual_rotations,
        )

        self.device = device
        self.model_dir = model_dir
        self.load_ckpt_path = load_ckpt_path
        self.sampler = SD3Euler(
            model_key=model_dir,
            device=device,
            use_8bit=False,
            load_ckpt_path=load_ckpt_path,
        )

        self.residual_rotation_matrices = None
        self.residual_rotation_meta = None
        self.residual_target_layers = list(residual_target_layers) if residual_target_layers is not None else None

        if residual_procrustes_path is not None:
            rotation_matrices, target_layers, meta = load_residual_procrustes(
                residual_procrustes_path
            )
            rotation_matrices, self.residual_target_layers = select_residual_rotations(
                rotation_matrices,
                target_layers,
                self.residual_target_layers,
            )
            self.residual_rotation_matrices = rotation_matrices
            self.residual_rotation_meta = meta

        self.residual_origin_layers = resolve_origin_layers(
            origin_layer=residual_origin_layer,
            origin_layers=residual_origin_layers,
            meta=self.residual_rotation_meta,
        )
        self.residual_origin_layer = (
            self.residual_origin_layers[0]
            if self.residual_origin_layers is not None and len(self.residual_origin_layers) == 1
            else None
        )

        if residual_weights is None and residual_weights_path is not None:
            residual_weights = load_residual_weights(residual_weights_path)
        if residual_weights is not None and not torch.is_tensor(residual_weights):
            residual_weights = torch.tensor(list(residual_weights), dtype=torch.float32)

        self.residual_weights = residual_weights
        self.residual_use_layernorm = bool(residual_use_layernorm)
        self.residual_timestep_weight_fn = build_timestep_residual_weight_fn(
            timestep_residual_weight_fn,
            power=timestep_residual_weight_power,
            exp_alpha=timestep_residual_weight_exp_alpha,
        )
        self.residual_timestep_stage = int(timestep_stage)
        self.use_residual = self.residual_origin_layers is not None
        self._resolve_rotation_bucket = resolve_rotation_bucket

    @property
    def model_id_or_name(self) -> str:
        return self.load_ckpt_path or self.model_dir

    def generation_config(self) -> dict[str, Any]:
        residual_weights = self.residual_weights
        if torch.is_tensor(residual_weights):
            residual_weights = residual_weights.detach().cpu().tolist()

        return {
            "model_id_or_name": self.model_id_or_name,
            "base_model_dir": self.model_dir,
            "load_ckpt_path": self.load_ckpt_path,
            "scheduler": type(self.sampler.scheduler).__name__,
            "residual_target_layers": self.residual_target_layers,
            "residual_origin_layer": self.residual_origin_layer,
            "residual_origin_layers": self.residual_origin_layers,
            "residual_weights": residual_weights,
            "residual_use_layernorm": self.residual_use_layernorm,
            "residual_timestep_stage": self.residual_timestep_stage,
        }

    def _latent_shape(self, image_size: tuple[int, int], batch_size: int) -> tuple[int, int, int, int]:
        height, width = image_size
        latent_h = height // self.sampler.vae_scale_factor
        latent_w = width // self.sampler.vae_scale_factor

        denoiser = getattr(self.sampler.denoiser, "module", self.sampler.denoiser)
        base = getattr(denoiser, "base_model", denoiser)
        base = getattr(base, "module", base)

        in_channels = None
        if hasattr(base, "config") and hasattr(base.config, "in_channels"):
            in_channels = int(base.config.in_channels)
        elif hasattr(base, "in_channels"):
            in_channels = int(base.in_channels)
        else:
            inner = getattr(base, "model", None) or getattr(base, "transformer", None)
            inner = getattr(inner, "module", inner)
            if inner is not None and hasattr(inner, "config") and hasattr(inner.config, "in_channels"):
                in_channels = int(inner.config.in_channels)

        if in_channels is None:
            raise AttributeError(
                "Cannot resolve denoiser in_channels for latent initialization."
            )
        return batch_size, in_channels, latent_h, latent_w

    def _make_seeded_latents(
        self,
        seeds: Sequence[int],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        latent_shape = self._latent_shape(image_size=image_size, batch_size=1)
        generator_device = self.sampler.device_diff.type
        latents = []
        for seed in seeds:
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(int(seed))
            latent = torch.randn(
                latent_shape,
                generator=generator,
                device=self.sampler.device_diff,
                dtype=self.sampler.dtype,
            )
            latents.append(latent)
        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def _sample_batch(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        latents: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        prompt_list = list(prompts)
        negative_prompt_list = list(negative_prompts)
        batch_size = len(prompt_list)

        prompt_emb, pooled_emb, _ = self.sampler.encode_prompt(prompt_list, batch_size=batch_size)
        negative_prompt_emb, negative_pooled_emb, _ = self.sampler.encode_prompt(
            negative_prompt_list,
            batch_size=batch_size,
        )

        z = latents
        self.sampler.scheduler.set_timesteps(num_inference_steps, device=self.sampler.device)
        timesteps = self.sampler.scheduler.timesteps
        steps = timesteps / self.sampler.scheduler.config.num_train_timesteps

        for step_index, t in enumerate(timesteps):
            timestep = t.expand(z.shape[0]).to(self.sampler.device)

            if not self.use_residual:
                pred_v = self.sampler.predict_vector(z, timestep, prompt_emb, pooled_emb)
                pred_neg_v = (
                    self.sampler.predict_vector(z, timestep, negative_prompt_emb, negative_pooled_emb)
                    if guidance_scale != 1.0
                    else 0.0
                )
            else:
                stage_active = self.sampler._is_residual_stage_active(
                    timestep,
                    self.residual_timestep_stage,
                )

                if stage_active:
                    timestep_weight = self.sampler._resolve_timestep_residual_weight(
                        timestep,
                        self.sampler.scheduler.config.num_train_timesteps,
                        self.residual_timestep_weight_fn if self.residual_weights is not None else None,
                    )
                    effective_residual_weights = self.sampler._scale_residual_weights(
                        self.residual_weights,
                        timestep_weight,
                        device=self.sampler.device_diff,
                        dtype=self.sampler.dtype,
                    )
                    selected_rotations = self._resolve_rotation_bucket(
                        self.residual_rotation_matrices,
                        self.residual_rotation_meta,
                        timestep,
                    )
                else:
                    effective_residual_weights = None
                    selected_rotations = None

                pred_v = self.sampler.predict_vector_residual(
                    z,
                    timestep,
                    prompt_emb,
                    pooled_emb,
                    residual_target_layers=self.residual_target_layers,
                    residual_origin_layer=self.residual_origin_layer,
                    residual_origin_layers=self.residual_origin_layers,
                    residual_weights=effective_residual_weights,
                    residual_use_layernorm=self.residual_use_layernorm,
                    residual_rotation_matrices=selected_rotations,
                )
                pred_neg_v = (
                    self.sampler.predict_vector_residual(
                        z,
                        timestep,
                        negative_prompt_emb,
                        negative_pooled_emb,
                        residual_target_layers=self.residual_target_layers,
                        residual_origin_layer=self.residual_origin_layer,
                        residual_origin_layers=self.residual_origin_layers,
                        residual_weights=effective_residual_weights,
                        residual_use_layernorm=self.residual_use_layernorm,
                        residual_rotation_matrices=selected_rotations,
                    )
                    if guidance_scale != 1.0
                    else 0.0
                )

            step = steps[step_index]
            next_step = steps[step_index + 1] if step_index + 1 < num_inference_steps else 0.0
            z = z + (next_step - step) * (pred_neg_v + guidance_scale * (pred_v - pred_neg_v))

        return self.sampler.decode(z)

    def generate(
        self,
        prompt: str,
        seeds: Sequence[int],
        image_size: tuple[int, int] = (512, 512),
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str | None = None,
        batch_size: int = 8,
    ) -> list[Image.Image]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, but got {batch_size}.")
        if negative_prompt is None:
            negative_prompt = ""

        images: list[Image.Image] = []
        for seed_chunk in chunked(list(seeds), batch_size):
            prompt_batch = [prompt] * len(seed_chunk)
            negative_prompt_batch = [negative_prompt] * len(seed_chunk)
            latents = self._make_seeded_latents(seed_chunk, image_size=image_size)
            image_batch = self._sample_batch(
                prompts=prompt_batch,
                negative_prompts=negative_prompt_batch,
                latents=latents,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            for img_tensor in image_batch:
                images.append(tensor_to_pil(img_tensor))
        return images


def evaluate_conditional_vendi(
    model: SD3ResidualBatchGenerator,
    prompt: str,
    seeds: list[int],
    output_dir: str,
    image_size: tuple[int, int] = (512, 512),
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: str | None = None,
    batch_size: int = 8,
    feature_batch_size: int = 32,
    inception_weights_path: str = DEFAULT_INCEPTION_WEIGHTS_PATH,
    device: str = "cuda",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pil_images = model.generate(
        prompt=prompt,
        seeds=seeds,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
    )
    if len(pil_images) != len(seeds):
        raise RuntimeError(
            f"Expected {len(seeds)} generated images, but got {len(pil_images)}."
        )

    image_paths: list[str] = []
    for idx, (seed, image) in enumerate(zip(seeds, pil_images)):
        image_path = images_dir / f"{idx:04d}_seed{seed}.png"
        image.save(image_path)
        image_paths.append(str(image_path.resolve()))

    features = extract_inception_features(
        image_paths=image_paths,
        batch_size=min(len(image_paths), feature_batch_size),
        device=device,
        inception_weights_path=inception_weights_path,
    )
    feature_path = output_path / "inception_features.npy"
    np.save(feature_path, features.astype(np.float32))

    similarity_matrix = compute_cosine_similarity_matrix(features)
    similarity_matrix_path = output_path / "similarity_matrix.npy"
    np.save(similarity_matrix_path, similarity_matrix.astype(np.float32))

    vendi_score, eigenvalues = compute_vendi_score_from_similarity_matrix(similarity_matrix)
    num_samples = len(seeds)
    result = {
        "prompt": prompt,
        "num_samples": num_samples,
        "seeds": list(seeds),
        "image_paths": image_paths,
        "feature_path": str(feature_path.resolve()),
        "similarity_matrix_path": str(similarity_matrix_path.resolve()),
        "vendi_score": vendi_score,
        "normalized_vendi_score": vendi_score / float(num_samples),
        "eigenvalues": eigenvalues.tolist(),
        "generation_config": {
            "image_size": [int(image_size[1]), int(image_size[0])],
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "negative_prompt": negative_prompt,
            **model.generation_config(),
        },
    }
    save_json(result, output_path / "result.json")
    return result


def aggregate_prompt_results(
    output_root: str | Path,
    expected_num_prompts: int,
    num_images_per_prompt: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    output_root = Path(output_root)
    result_paths = sorted(output_root.glob("per_prompt/**/result.json"))
    results = [load_json(path) for path in result_paths]

    if len(results) != expected_num_prompts:
        raise RuntimeError(
            f"Expected {expected_num_prompts} prompt result files, but found {len(results)} under {output_root}."
        )

    vendi_scores = np.array([float(item["vendi_score"]) for item in results], dtype=np.float64)
    normalized_scores = np.array(
        [float(item["normalized_vendi_score"]) for item in results],
        dtype=np.float64,
    )

    summary = {
        "num_prompts_requested": int(expected_num_prompts),
        "num_prompts_completed": int(len(results)),
        "num_images_per_prompt": int(num_images_per_prompt),
        "seeds": [int(seed) for seed in seeds],
        "mean_vendi_score": float(vendi_scores.mean()),
        "std_vendi_score": float(vendi_scores.std(ddof=0)),
        "min_vendi_score": float(vendi_scores.min()),
        "max_vendi_score": float(vendi_scores.max()),
        "mean_normalized_vendi_score": float(normalized_scores.mean()),
        "std_normalized_vendi_score": float(normalized_scores.std(ddof=0)),
        "min_normalized_vendi_score": float(normalized_scores.min()),
        "max_normalized_vendi_score": float(normalized_scores.max()),
        "result_files": [str(path.resolve()) for path in result_paths],
    }
    save_json(summary, output_root / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate conditional Vendi Score for SD3 residual on COCO5k."
    )
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3",
    )
    parser.add_argument("--load_ckpt_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_prompts", type=int, default=5000)
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed_offset", type=int, default=42)

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--generation_batch_size", type=int, default=1)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument(
        "--inception_weights_path",
        type=str,
        default=DEFAULT_INCEPTION_WEIGHTS_PATH,
    )
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_origin_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument("--residual_use_layernorm", type=int, default=1)
    parser.add_argument("--timestep_residual_weight_fn", type=str, default="constant")
    parser.add_argument("--timestep_residual_weight_power", type=float, default=1.0)
    parser.add_argument("--timestep_residual_weight_exp_alpha", type=float, default=1.5)
    parser.add_argument(
        "--timestep_stage",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(
            f"rank must be in [0, world_size-1], but got rank={args.rank}, world_size={args.world_size}."
        )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.startswith("cuda"):
        cuda_device = torch.device(device)
        torch.cuda.set_device(0 if cuda_device.index is None else cuda_device.index)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = prepare_seeds(
        num_images_per_prompt=args.num_images_per_prompt,
        seeds=args.seeds,
        seed_offset=args.seed_offset,
    )

    all_entries = load_coco_prompts(args.datadir, num_prompts=args.num_prompts)
    sharded_entries = [
        entry for entry in all_entries if entry.global_index % args.world_size == args.rank
    ]

    run_config = {
        **vars(args),
        "resolved_seeds": seeds,
        "resolved_device": device,
    }
    if args.rank == 0 and not args.aggregate_only:
        save_json(run_config, output_root / "run_config.json")
        save_json([asdict(entry) for entry in all_entries], output_root / "selected_prompts.json")

    if args.aggregate_only:
        summary = aggregate_prompt_results(
            output_root=output_root,
            expected_num_prompts=args.num_prompts,
            num_images_per_prompt=args.num_images_per_prompt,
            seeds=seeds,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    model = SD3ResidualBatchGenerator(
        model_dir=args.model_dir,
        load_ckpt_path=args.load_ckpt_path,
        device=device,
        residual_target_layers=args.residual_target_layers,
        residual_origin_layer=args.residual_origin_layer,
        residual_origin_layers=args.residual_origin_layers,
        residual_weights=args.residual_weights,
        residual_weights_path=args.residual_weights_path,
        residual_procrustes_path=args.residual_procrustes_path,
        residual_use_layernorm=bool(args.residual_use_layernorm),
        timestep_residual_weight_fn=args.timestep_residual_weight_fn,
        timestep_residual_weight_power=args.timestep_residual_weight_power,
        timestep_residual_weight_exp_alpha=args.timestep_residual_weight_exp_alpha,
        timestep_stage=args.timestep_stage,
    )

    progress = tqdm(sharded_entries, desc=f"rank {args.rank}", total=len(sharded_entries))
    for entry in progress:
        prompt_output_dir = build_prompt_output_dir(output_root, entry)
        result_path = prompt_output_dir / "result.json"

        if result_path.exists() and not args.overwrite:
            continue

        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        save_json(asdict(entry), prompt_output_dir / "prompt_meta.json")

        try:
            result = evaluate_conditional_vendi(
                model=model,
                prompt=entry.prompt,
                seeds=seeds,
                output_dir=str(prompt_output_dir),
                image_size=(args.height, args.width),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                batch_size=args.generation_batch_size,
                feature_batch_size=args.feature_batch_size,
                inception_weights_path=args.inception_weights_path,
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed while evaluating prompt index={entry.global_index}, image_id={entry.image_id}, "
                f"prompt={entry.prompt!r}"
            ) from exc

        progress.set_postfix(
            vendi=f"{result['vendi_score']:.4f}",
            norm=f"{result['normalized_vendi_score']:.4f}",
        )


if __name__ == "__main__":
    main()
