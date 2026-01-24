import argparse
import numpy as np
import random, os
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms
import torch.nn as nn
import tqdm

from sampler import SD3Euler, build_timestep_residual_weight_fn
from util import load_residual_procrustes, select_residual_rotations, load_residual_weights
from dataset.datasets import get_target_dataset
import json

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def tensor_to_pil(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img_tensor = img_tensor.detach().cpu().clamp(0, 1)
    img = (img_tensor * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sampling config
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img_size', type=int, default=1024, choices=[256,512,768,1024])
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='0 for null prompt, 1 for only using conditional prompt')
    parser.add_argument('--batch_size', type=int, default=1)
    # path
    parser.add_argument('--load_dir', type=str, default=None, help="replace it with your checkpoint")
    parser.add_argument('--save_dir', type=str, default=None, help="default savedir is set to under load_dir")
    parser.add_argument('--datadir', type=str, default='', help='data path')
    # model config
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'sdxl', 'sd1.5'], help='Model to use')

    # one sample generation
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--save_name', type=str, default="image_sd3")
    # set generation
    parser.add_argument('--num', type=int, default=-1, help='number of sampling images. -1 for whole dataset')
    parser.add_argument('--dataset', type=str, nargs='+', default=None, choices=['coco'])


    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    parser.add_argument("--residual_weights_path", type=str, default=None)
    parser.add_argument("--residual_procrustes_path", type=str, default=None)
    parser.add_argument(
        "--timestep_residual_weight_fn",
        type=str,
        default="constant",
        help="Mapping from timestep (0-1000) to residual weight multiplier.",
    )
    parser.add_argument(
        "--timestep_residual_weight_power",
        type=float,
        default=1.0,
        help="Optional power for timestep residual weight mapping.",
    )
    parser.add_argument(
        "--timestep_residual_weight_exp_alpha",
        type=float,
        default=1.5,
        help="Exponent alpha for exponential timestep residual weight mapping.",
    )


    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)


    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    interpolation = INTERPOLATIONS['bilinear']
    transform = get_transform(interpolation, 1024)

    # load model
    if args.model == 'sd3':
        sampler = SD3Euler(use_8bit=False, load_ckpt_path=args.load_dir)
    else:
        raise ValueError('args.model should be one of [sd3, sdxl, sd1.5]')

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

    # sample set
    if args.dataset is not None:
        # save dir
        config=f'{"-".join(args.dataset)}-cfg{args.cfg_scale}-nfe{args.NFE}'
        if args.save_dir is not None:
            args.load_dir = args.save_dir
        savedir = os.path.join(args.load_dir, config)
        if not os.path.exists(savedir):
            os.makedirs(savedir)


        sampler.denoiser.to(torch.float32)
        torch.set_default_dtype(torch.float32)
        
        train_datasets = []
        for ds in args.dataset:
            train_datasets.append(get_target_dataset(ds, args.datadir, train=False, transform=transform))

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

            # 找出缺失样本（不重复生成）
            missing_ids = [t for t in ids if not os.path.exists(os.path.join(savedir, f'{t:04d}.png'))]
            if not missing_ids:
                continue

            # 当前批次的 prompts
            index_map = {idx: pos for pos, idx in enumerate(ids)}
            sub_labels = [label[index_map[t]] for t in missing_ids]

            # -----------  🔥 统一 residual 采样逻辑 🔥 -----------
            with torch.inference_mode():
                if args.residual_origin_layer is None:
                    # 普通采样
                    img = sampler.sample(
                        sub_labels,
                        NFE=args.NFE,
                        img_shape=(args.img_size, args.img_size),
                        cfg_scale=args.cfg_scale,
                        batch_size=len(sub_labels),
                    )
                else:
                    # residual 采样
                    img = sampler.sample_residual(
                        sub_labels,
                        NFE=args.NFE,
                        img_shape=(args.img_size, args.img_size),
                        cfg_scale=args.cfg_scale,
                        batch_size=len(sub_labels),
                        residual_target_layers=args.residual_target_layers,
                        residual_origin_layer=args.residual_origin_layer,
                        residual_weights=args.residual_weights,
                        residual_rotation_matrices=residual_rotation_matrices,
                        residual_rotation_meta=residual_rotation_meta,
                        residual_timestep_weight_fn=build_timestep_residual_weight_fn(
                            args.timestep_residual_weight_fn,
                            power=args.timestep_residual_weight_power,
                            exp_alpha=args.timestep_residual_weight_exp_alpha,
                        ),
                    )
            # ----------------------------------------------------

            # 保存图片
            for bi, t in enumerate(missing_ids):
                imgname = f'{t:04d}.png'
                save_image(img[bi], os.path.join(savedir, imgname), normalize=True)
                results.append({"prompt": sub_labels[bi], "img_path": imgname})
                pbar.set_description(f'SD Sampling [{t}/{num}]')
            
        # save config
        if os.path.exists(os.path.join(args.load_dir, f"results-{config}.json")):
            with open(os.path.join(args.load_dir, f"results-{config}.json"), 'r', encoding='utf-8') as file:
                results_all = json.load(file)
                if isinstance(results_all, list):
                    results_all.extend(results)
                else:
                    results_all = [results_all] + results
        else:
            results_all = results
        with open(os.path.join(args.load_dir, f"results-{config}.json"), "w", encoding="utf-8") as file:
            json.dump(results_all, file, indent=4)  # `indent=4` makes the JSON more readable

    # sample image
    else:
        # save dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        # with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        #     img = sampler.sample([args.prompt], NFE=args.NFE, img_shape=(args.img_size, args.img_size), cfg_scale=args.cfg_scale, batch_size=1)
        sampler.denoiser.to(torch.float32)
        torch.set_default_dtype(torch.float32)
        with torch.inference_mode():
            if args.residual_origin_layer is None:
                img = sampler.sample([args.prompt], NFE=args.NFE, img_shape=(args.img_size, args.img_size),
                                    cfg_scale=args.cfg_scale, batch_size=1)
            else:
                img = sampler.sample_residual([args.prompt], NFE=args.NFE, img_shape=(args.img_size, args.img_size),
                                        cfg_scale=args.cfg_scale, 
                                        batch_size=1,
                                        residual_target_layers=args.residual_target_layers,
                                        residual_origin_layer=args.residual_origin_layer,
                                        residual_weights=args.residual_weights,
                                        residual_rotation_matrices=residual_rotation_matrices,
                                        residual_rotation_meta=residual_rotation_meta,
                                        residual_timestep_weight_fn=build_timestep_residual_weight_fn(
                                            args.timestep_residual_weight_fn,
                                            power=args.timestep_residual_weight_power,
                                            exp_alpha=args.timestep_residual_weight_exp_alpha,
                                        ),
                                    )       

        save_image(img, os.path.join(args.save_dir, f'{args.save_name}.png'), normalize=True)
        # fpath = f'{args.save_name}.png'
        # tensor_to_pil(img).save(fpath)
