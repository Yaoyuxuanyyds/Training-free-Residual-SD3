import torch
import random
from typing import Optional

import numpy as np
import torch

import os
import sys
import torch
import torch.distributed as dist
import torchvision.transforms as torch_transforms
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_residual_procrustes(
    path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    data = torch.load(path, map_location="cpu")
    target_layers = None

    if isinstance(data, dict):
        if "rotation_matrices" in data:
            rotation_matrices = data["rotation_matrices"]
        elif "R" in data:
            rotation_matrices = data["R"]
        else:
            raise KeyError("Procrustes file missing rotation_matrices/R key.")
        target_layers = data.get("target_layers")
    else:
        rotation_matrices = data

    if not torch.is_tensor(rotation_matrices):
        rotation_matrices = torch.tensor(rotation_matrices)

    if device is not None or dtype is not None:
        rotation_matrices = rotation_matrices.to(
            device=device if device is not None else rotation_matrices.device,
            dtype=dtype if dtype is not None else rotation_matrices.dtype,
        )

    return rotation_matrices, target_layers, data['meta']

path = "/inspire/hdd/project/chineseculture/public/yuxuan/Training-free-Residual-SD3/SD3.5-Residual/logs/procrustes_rotations/procrustes_rotations_coco5k_ln_t1_o2.pt"
residual_rotation_matrices, target_layers, meta = load_residual_procrustes(
    path
)
obj = torch.load(path, map_location="cpu")

print(type(obj))
