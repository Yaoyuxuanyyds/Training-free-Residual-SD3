import argparse
import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as torch_transforms
import tqdm
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from dataset.datasets import get_target_dataset
from flux_transformer_res import FluxTransformer2DModel_RES
from generate_image_res import FluxPipelineWithRES
from util import load_residual_procrustes, select_residual_rotations, load_residual_weights


INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize(size, interpolation=interpolation),
            torch_transforms.CenterCrop(size),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def pil_to_tensor(img):
    arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (arr * 2.0) - 1.0


class FluxSampler:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = FluxPipelineWithRES.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.pipe.transformer = FluxTransformer2DModel_RES(self.pipe.transformer)
        self.pipe.to(self.device)

    def sample(
        self,
        prompts,
        NFE,
        img_shape,
        cfg_scale,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        residual_rotation_matrices=None,
        residual_rotation_meta=None,
    ):
        height, width = img_shape
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompts,
                height=height,
                width=width,
                guidance_scale=cfg_scale,
                num_inference_steps=NFE,
                residual_target_layers=residual_target_layers,
                residual_origin_layer=residual_origin_layer,
                residual_weights=residual_weights,
                residual_rotation_matrices=residual_rotation_matrices,
                residual_rotation_meta=residual_rotation_meta,
            )
        images = result.images if hasattr(result, "images") else result[0]
        tensors = [pil_to_tensor(img) for img in images]
        return torch.stack(tensors, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=1024, choices=[256, 512, 768, 1024])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.5,
        help="0 for null prompt, 1 for only using conditional prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_dir", type=str, default=None, help="replace it with your checkpoint")
    parser.add_argument("--save_dir", type=str, default=None, help="default savedir is set to under load_dir")
    parser.add_argument("--datadir", type=str, default="", help="data path")
    parser.add_argument("--model", type=str, default="flux", choices=["flux"])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--save_name", type=str, default="image_flux")
    parser.add_argument("--num", type=int, default=-1, help="number of sampling images. -1 for whole dataset")
    parser.add_argument("--dataset", type=str, nargs="+", default=None, choices=["coco"])
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    interpolation = INTERPOLATIONS["bilinear"]
    transform = get_transform(interpolation, 1024)

    sampler = FluxSampler(model_path=args.load_dir)

    residual_rotation_matrices = None
    residual_rotation_meta = None
    if args.residual_procrustes_path is not None:
        residual_rotation_matrices, target_layers, meta = load_residual_procrustes(
            args.residual_procrustes_path
        )
        residual_rotation_matrices, args.residual_target_layers = select_residual_rotations(
            residual_rotation_matrices, target_layers, args.residual_target_layers
        )
        if args.residual_origin_layer is None and isinstance(meta, dict):
            args.residual_origin_layer = meta.get("origin_layer")
        residual_rotation_meta = meta

    if args.residual_weights is None and args.residual_weights_path is not None:
        args.residual_weights = load_residual_weights(args.residual_weights_path)
        print(f"Residual weights: {args.residual_weights}")
        print(f"Num res weights: {len(args.residual_weights)}")
        print(f"Num res targets: {len(args.residual_target_layers)}")

    if args.dataset is not None:
        config = f'{"-".join(args.dataset)}-cfg{args.cfg_scale}-nfe{args.NFE}'
        if args.save_dir is not None:
            args.load_dir = args.save_dir
        savedir = os.path.join(args.load_dir, config)
        os.makedirs(savedir, exist_ok=True)

        train_datasets = [
            get_target_dataset(ds, args.datadir, train=False, transform=transform)
            for ds in args.dataset
        ]
        train_dataset = ConcatDataset(train_datasets)
        num = args.num if args.num != -1 else len(train_dataset)

        indices = list(range(num))
        if args.world_size > 1:
            indices = [idx for idx in indices if idx % args.world_size == args.rank]

        class IndexedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                _, label = self.dataset[idx]
                return idx, label

        train_dataset = Subset(IndexedDataset(train_dataset), indices)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
        pbar = tqdm.tqdm(train_dataloader)
        results = []

        for batch_indices, label in pbar:
            ids = [int(i) for i in batch_indices]
            missing_ids = [t for t in ids if not os.path.exists(os.path.join(savedir, f"{t:04d}.png"))]
            if not missing_ids:
                continue

            index_map = {idx: pos for pos, idx in enumerate(ids)}
            sub_labels = [label[index_map[t]] for t in missing_ids]

            images = sampler.sample(
                sub_labels,
                NFE=args.NFE,
                img_shape=(args.img_size, args.img_size),
                cfg_scale=args.cfg_scale,
                residual_target_layers=args.residual_target_layers,
                residual_origin_layer=args.residual_origin_layer,
                residual_weights=args.residual_weights,
                residual_rotation_matrices=residual_rotation_matrices,
                residual_rotation_meta=residual_rotation_meta,
            )

            for bi, t in enumerate(missing_ids):
                imgname = f"{t:04d}.png"
                save_image(images[bi], os.path.join(savedir, imgname), normalize=True)
                results.append({"prompt": sub_labels[bi], "img_path": imgname})
                pbar.set_description(f"Flux Sampling [{t}/{num}]")

        results_path = os.path.join(args.load_dir, f"results-{config}.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as file:
                results_all = json.load(file)
                if isinstance(results_all, list):
                    results_all.extend(results)
                else:
                    results_all = [results_all] + results
        else:
            results_all = results
        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(results_all, file, indent=4)
    else:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        images = sampler.sample(
            [args.prompt],
            NFE=args.NFE,
            img_shape=(args.img_size, args.img_size),
            cfg_scale=args.cfg_scale,
            residual_target_layers=args.residual_target_layers,
            residual_origin_layer=args.residual_origin_layer,
            residual_weights=args.residual_weights,
            residual_rotation_matrices=residual_rotation_matrices,
            residual_rotation_meta=residual_rotation_meta,
        )

        save_image(images[0], os.path.join(args.save_dir, f"{args.save_name}.png"), normalize=True)
