import argparse
import os
import glob
import re
import torch
from tqdm import tqdm

from generate_image_res import SD35PipelineWithRES
from sd35_transformer_res import SD35Transformer2DModel_RES
from util import load_residual_procrustes, select_residual_rotations, set_seed, load_residual_weights


torch.set_grad_enabled(False)


class SD35ImageGenerator:
    def __init__(
        self,
        model_path,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_rotation_matrices=None,
        residual_rotation_meta=None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = SD35PipelineWithRES.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.pipe.transformer = SD35Transformer2DModel_RES(self.pipe.transformer)

        self.residual_target_layers = residual_target_layers
        self.residual_origin_layer = residual_origin_layer
        self.residual_weights = residual_weights
        self.residual_rotation_matrices = residual_rotation_matrices
        self.residual_rotation_meta = residual_rotation_meta

    def generate(
        self,
        prompt,
        seed,
        img_size,
        steps,
        scale,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_rotation_matrices=None,
        residual_rotation_meta=None,
    ):
        rt = residual_target_layers if residual_target_layers is not None else self.residual_target_layers
        ro = residual_origin_layer if residual_origin_layer is not None else self.residual_origin_layer
        rw = residual_weights if residual_weights is not None else self.residual_weights
        rr = residual_rotation_matrices if residual_rotation_matrices is not None else self.residual_rotation_matrices
        rr_meta = residual_rotation_meta if residual_rotation_meta is not None else self.residual_rotation_meta

        set_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                height=img_size,
                width=img_size,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator,
                output_type="pt",
                residual_target_layers=rt,
                residual_origin_layer=ro,
                residual_weights=rw,
                residual_rotation_matrices=rr,
                residual_rotation_meta=rr_meta,
            )
        image = result.images[0] if hasattr(result, "images") else result[0]
        if image.dim() == 4:
            image = image[0]
        return image


def clean_prompt_for_filename(prompt):
    illegal_chars = r'[\/\:*?"<>|]'
    cleaned = re.sub(illegal_chars, '_', prompt.strip())
    if len(cleaned) > 3000:
        cleaned = cleaned[:3000] + "_truncated"
    return cleaned


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench/examples/dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5-large",
    )

    parser.add_argument(
        "--outdir_base",
        type=str,
        default="/inspire/.../examples",
    )

    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=7.0)

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627],
        metavar="SEED",
    )

    parser.add_argument("--output_prefix", type=str, default="sd35_multigpu")

    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    return parser.parse_args()


def main(opt):
    os.makedirs(opt.outdir_base, exist_ok=True)

    residual_rotation_matrices = None
    residual_rotation_meta = None
    if opt.residual_procrustes_path is not None:
        residual_rotation_matrices, target_layers, meta = load_residual_procrustes(
            opt.residual_procrustes_path
        )
        residual_rotation_matrices, opt.residual_target_layers = select_residual_rotations(
            residual_rotation_matrices, target_layers, opt.residual_target_layers
        )
        if opt.residual_origin_layer is None and isinstance(meta, dict):
            opt.residual_origin_layer = meta.get("origin_layer")
        residual_rotation_meta = meta

    if opt.residual_weights is None and opt.residual_weights_path is not None:
        opt.residual_weights = load_residual_weights(opt.residual_weights_path)

    generator = SD35ImageGenerator(
        model_path=opt.model,
        residual_target_layers=opt.residual_target_layers,
        residual_origin_layer=opt.residual_origin_layer,
        residual_weights=opt.residual_weights,
        residual_rotation_matrices=residual_rotation_matrices,
        residual_rotation_meta=residual_rotation_meta,
    )

    prompt_files = sorted(glob.glob(os.path.join(opt.dataset_dir, "*.txt")))
    total_prompts = len(prompt_files)

    if opt.world_size > 1:
        prompt_files = [f for i, f in enumerate(prompt_files) if i % opt.world_size == opt.rank]

    for prompt_idx, prompt_file in enumerate(tqdm(prompt_files, desc="Prompts")):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        cleaned_prompt = clean_prompt_for_filename(prompt)
        prompt_dir = os.path.join(opt.outdir_base, cleaned_prompt)
        os.makedirs(prompt_dir, exist_ok=True)

        for i in range(opt.n_samples):
            seed = opt.seeds[i % len(opt.seeds)]
            output_name = f"{opt.output_prefix}_s{seed:06d}_i{i:02d}.png"
            output_path = os.path.join(prompt_dir, output_name)

            if os.path.exists(output_path):
                continue

            image = generator.generate(
                prompt=prompt,
                seed=seed,
                img_size=opt.H,
                steps=opt.steps,
                scale=opt.scale,
                residual_target_layers=opt.residual_target_layers,
                residual_origin_layer=opt.residual_origin_layer,
                residual_weights=opt.residual_weights,
                residual_rotation_matrices=residual_rotation_matrices,
            )

            if image.dim() == 3 and image.shape[0] != 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)

            from torchvision.utils import save_image

            save_image(image, output_path, normalize=True)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
