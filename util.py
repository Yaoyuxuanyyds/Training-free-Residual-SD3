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

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(size=512, interpolation=InterpolationMode.BICUBIC):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform

def get_qwen_transform(size=512):
    def pil_to_tensor_255(pic):
        """将 PIL Image 转换为 torch.Tensor，保留像素范围为 [0, 255]"""
        np_img = np.array(pic, dtype=np.uint8)  # shape: [H, W, 3], dtype: uint8
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()  # [3, H, W]
        return tensor.to(torch.uint8)

    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        pil_to_tensor_255,
    ])
    return transform


def build_text_token_nonpad_mask(
    feats: torch.Tensor,
    *,
    clip_token_length: int = 77,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a boolean mask indicating non-padding tokens for SD3 text features."""

    if feats.dim() == 2:
        feats_3d = feats.unsqueeze(0)
        squeeze = True
    elif feats.dim() == 3:
        feats_3d = feats
        squeeze = False
    else:
        raise ValueError(
            f"Expected text features of shape (L, D) or (B, L, D), got {feats.shape}"
        )

    # Padding tokens share identical embeddings with the final token for the T5
    # branch and with the last CLIP token inside the 77-token window. We detect
    # such duplicates using a small numerical tolerance.
    B, L, _ = feats_3d.shape
    ref_last = feats_3d[:, -1:, :]
    diff_last = torch.amax(torch.abs(feats_3d - ref_last), dim=-1)
    pad_mask = diff_last < eps

    if clip_token_length is not None and clip_token_length > 0 and L >= clip_token_length:
        ref_clip = feats_3d[:, clip_token_length - 1 : clip_token_length, :]
        diff_clip = torch.amax(
            torch.abs(feats_3d[:, :clip_token_length, :] - ref_clip), dim=-1
        )
        pad_mask[:, :clip_token_length] |= diff_clip < eps

    nonpad_mask = ~pad_mask

    # Fallback: keep all tokens if the heuristic masks everything out to avoid
    # downstream division-by-zero issues.
    flat_counts = nonpad_mask.view(B, -1).sum(dim=-1)
    fallback_rows = flat_counts == 0
    if fallback_rows.any():
        nonpad_mask[fallback_rows] = True

    if squeeze:
        return nonpad_mask[0]
    return nonpad_mask


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

    return rotation_matrices, target_layers, data


def denormalize(imgs: torch.Tensor) -> torch.Tensor:
    """
    将像素值范围为 [-1, 1] 的图像张量转换为 [0, 255] 的 uint8 类型张量。

    参数:
        imgs (torch.Tensor): 输入张量，形状为 (B, 3, H, W)，像素范围为 [-1, 1]

    返回:
        torch.Tensor: 输出张量，形状不变，像素范围为 [0, 255]，类型为 uint8
    """
    imgs = (imgs * 0.5 + 0.5) * 255.0  # 将 [-1, 1] 映射到 [0, 255]
    imgs = imgs.clamp(0, 255)  # 截断非法值
    return imgs.to(torch.float32)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = rank  # Now matches the index inside CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(local_rank)  # Set correct GPU
    setup_for_distributed(local_rank == 0)
    return local_rank, dist.get_world_size()

def cleanup():
    dist.destroy_process_group()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def gather_from_all_gpus(data):
    if not is_dist_avail_and_initialized():
        return data
    world_size = dist.get_world_size()
    if world_size == 1:
        return data
    data_list = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(data_list, data.contiguous())
    data_gathered = torch.cat(data_list, dim=0)
    return data_gathered


def build_prompts_from_captions_cot(captions: list[str]) -> list[str]:
    """
    构造符合 Qwen2VL 格式的 prompt，结合图片和 caption，引导模型对图片和描述之间的关系进行推理。

    参数:
        captions: List[str]，长度为 B，每个是图像对应的文本描述

    返回:
        prompts: List[str]，每个 prompt 包含 vision token 和描述，引导模型理解图像与 caption 的关系
    """
    template = (
        "<|im_start|>system\n"
        "You are a helpful assistant. When the user requests an image, the assistant first thinks about the reasoning process in the mind and then provides the user with concise prompt as the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Given the previous image as the ground truth generated result. Describe in detail the image you would generate according to the following instructions: \"{caption}\". Please return the contents of 'think' and 'answer' as required.<|im_end|>"
    )

    return [template.format(caption=cap) for cap in captions]



def build_prompts_from_captions(captions: list[str]) -> list[str]:
    """
    构造符合 Qwen2VL 格式的 prompt，结合图片和 caption，引导模型对图片和描述之间的关系进行推理。

    参数:
        captions: List[str]，长度为 B，每个是图像对应的文本描述

    返回:
        prompts: List[str]，每个 prompt 包含 vision token 和描述，引导模型理解图像与 caption 的关系
    """
    template = (
        "<|im_start|>system\n"
        "You are a helpful assistant. When the user requests an image, the assistant first thinks about the reasoning process in the mind and then provides the user with concise prompt as the answer. Given the following image as the ground truth generated result. Describe in detail the image you would generate according to the instructions given by user:\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "\"{caption}\". <|im_end|>"
    )

    return [template.format(caption=cap) for cap in captions]




def list_all_submodules(model, show_params=False, keyword=None):
    """
    列出模型的所有子模块名称和类型。
    show_params=True 时会显示参数数量
    keyword 用于只显示包含该字符串的模块（可选）
    """
    for name, module in model.named_modules():
        if keyword is None or keyword in name:
            if show_params:
                params = sum(p.numel() for p in module.parameters())
                print(f"{name:60s} | {type(module).__name__:25s} | params: {params}")
            else:
                print(f"{name:60s} | {type(module).__name__}")
                
                
       
       
       
# =========================
# 时间采样函数
# =========================
def sample_timesteps(batch_size, num_steps, device, mode="uniform", **kwargs):
    if mode == "uniform":
        t = torch.randint(0, num_steps, (batch_size,), device=device)
    elif mode == "gaussian":
        mu_ratio = kwargs.get("mu_ratio", 0.6)
        sigma_ratio = kwargs.get("sigma_ratio", 0.1)
        mu = int(num_steps * mu_ratio)
        sigma = int(num_steps * sigma_ratio)
        t = torch.normal(mean=torch.full((batch_size,), mu, device=device, dtype=torch.float), std=sigma)
        t = t.clamp(0, num_steps - 1).long()
    elif mode == "beta":
        alpha = kwargs.get("alpha", 5.0)
        beta = kwargs.get("beta", 2.0)
        u = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        t = (u * (num_steps - 1)).long()
    elif mode == "logitnorm":
        mu = kwargs.get("mu", 0.5)
        sigma = kwargs.get("sigma", 1.0)
        z = torch.normal(mean=torch.full((batch_size,), mu, device=device, dtype=torch.float), std=sigma)
        u = torch.sigmoid(z)
        t = (u * (num_steps - 1)).long()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return t
         


import torch
import torchaudio.functional as TAF

import numpy as np
from sklearn.cross_decomposition import CCA

try:
    import pymp
    pymp_available = True
except ImportError:
    pymp_available = False
    print("Please install the pymp library using `pip install pymp` to speed up non-batched metrics")


class AlignmentMetrics:

    SUPPORTED_METRICS = [
        "cycle_knn",
        "mutual_knn",
        "lcs_knn",
        "cka",
        "unbiased_cka",
        "cknna",
        "svcca",
        "edit_distance_knn",
    ]

    @staticmethod
    def measure(metric, *args, **kwargs):
        """ metric is a string for the function """

        if metric not in AlignmentMetrics.SUPPORTED_METRICS:
            raise ValueError(f"Unrecognized metric: {metric}")

        return getattr(AlignmentMetrics, metric)(*args, **kwargs)


    @staticmethod
    def cycle_knn(feats_A, feats_B, topk):
        """
        LLM nearest neighbors -> Query Language Pair -> LVM nearest neighbors
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            acc: a float representing the accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   
        return compute_knn_accuracy(knn_A[knn_B]).item()


    @staticmethod
    def mutual_knn(feats_A, feats_B, topk):
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0
        
        acc = (lvm_mask * llm_mask).sum(dim=1) / topk
        
        return acc.mean().item()
    
    
    @staticmethod
    def lcs_knn(feats_A, feats_B, topk):
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)        
        score = longest_ordinal_sequence(knn_A, knn_B).float().mean()
        return score
    
    
    @staticmethod
    def cka(feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=False):
        """Computes the unbiased Centered Kernel Alignment (CKA) between features."""
        
        if kernel_metric == 'ip':
            # Compute kernel matrices for the linear case
            K = torch.mm(feats_A, feats_A.T)
            L = torch.mm(feats_B, feats_B.T)
        elif kernel_metric == 'rbf':
            # COMPUTES RBF KERNEL
            K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
            L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))
        else:
            raise ValueError(f"Invalid kernel metric {kernel_metric}")

        # Compute HSIC values
        hsic_fn = hsic_unbiased if unbiased else hsic_biased
        hsic_kk = hsic_fn(K, K)
        hsic_ll = hsic_fn(L, L)
        hsic_kl = hsic_fn(K, L)

        # Compute CKA
        #print('hsic', hsic_kl)
        cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)        
        return cka_value.item()
    
    
    @staticmethod
    def unbiased_cka(*args, **kwargs):
        kwargs['unbiased'] = True
        return AlignmentMetrics.cka(*args, **kwargs)
    
    
    @staticmethod
    def svcca(feats_A, feats_B, cca_dim=10):

        # Center and scale the activations
        def preprocess_activations(act):
            act = act - torch.mean(act, axis=0)
            act = act / (torch.std(act, axis=0) + 1e-8)
            return act

        feats_A = preprocess_activations(feats_A)
        feats_B = preprocess_activations(feats_B)

        # Compute SVD
        U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
        U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)
        
        U1 = U1.cpu().detach().numpy()
        U2 = U2.cpu().detach().numpy()

        # Compute CCA
        cca = CCA(n_components=cca_dim)
        cca.fit(U1, U2)
        U1_c, U2_c = cca.transform(U1, U2)

        # sometimes it goes to nan, this is just to avoid that
        U1_c += 1e-10 * np.random.randn(*U1_c.shape)
        U2_c += 1e-10 * np.random.randn(*U2_c.shape)

        # Compute SVCCA similarity
        svcca_similarity = np.mean(
            [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
        )
        return svcca_similarity
    
    
    @staticmethod
    def edit_distance_knn(feats_A, feats_B, topk):
        """
        Computes the edit distance between the nearest neighbors of feats_A and feats_B.
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)
        
        # given N x topk with integer entries, compute edit distance
        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        edit_distance = compute_distance(knn_A, knn_B, TAF.edit_distance)
        return 1 - torch.mean(edit_distance) / topk
    
    
    @staticmethod
    def cknna(feats_A, feats_B, topk=None, distance_agnostic=False, unbiased=True):
        """ similarity only cka variant """
        n = feats_A.shape[0]
                
        if topk < 2:
            raise ValueError("CKNNA requires topk >= 2")
        
        if topk is None:
            topk = feats_A.shape[0] - 1
                            
        K = feats_A @ feats_A.T
        L = feats_B @ feats_B.T
        device = feats_A.device

        def similarity(K, L, topk):                         
            if unbiased:            
                K_hat = K.clone().fill_diagonal_(float("-inf"))
                L_hat = L.clone().fill_diagonal_(float("-inf"))
            else:
                K_hat, L_hat = K, L

            # get topk indices for each row
            # if unbiased we cannot attend to the diagonal unless full topk
            # else we can attend to the diagonal
            _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
            _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
            
            # create masks for nearest neighbors
            mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
            mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)
            
            # intersection of nearest neighbors
            mask = mask_K * mask_L
                        
            if distance_agnostic:
                sim = mask * 1.0
            else:
                if unbiased:
                    sim = hsic_unbiased(mask * K, mask * L)
                else:
                    sim = hsic_biased(mask * K, mask * L)
            return sim

        sim_kl = similarity(K, L, topk)
        sim_kk = similarity(K, K, topk)
        sim_ll = similarity(L, L, topk)
                
        return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """ Compute the biased HSIC (the original CKA) """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)

    
def compute_knn_accuracy(knn):
    """
    Compute the accuracy of the nearest neighbors. Assumes index is the gt label.
    Args:
        knn: a torch tensor of shape N x topk
    Returns:
        acc: a float representing the accuracy
    """
    n = knn.shape[0]
    acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
    acc = acc.float().view(n, -1).max(dim=1).values.mean()
    return acc
    

def compute_nearest_neighbors(feats, topk=1):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn


def longest_ordinal_sequence(X, Y):
    """ For each pair in X and Y, compute the length of the longest sub-sequence (LCS) """
    
    def lcs_length(x, y):
        """
        Compute the length of the longest common subsequence between two sequences.
        This is a classic dynamic programming implementation.
        """
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs = compute_distance(X, Y, lcs_length)
    return lcs


def compute_distance(X, Y, dist_fn):
    """ compute distance in parallel"""
    B, N = X.shape
    distances = np.zeros(B)
    X, Y = X.cpu().numpy(), Y.cpu().numpy()

    if pymp_available:
        with pymp.Parallel(4) as p:
            for i in p.range(B):
                distances[i] = dist_fn(X[i], Y[i])
    else:
        for i in range(B):
            distances[i] = dist_fn(X[i], Y[i])
    return torch.tensor(distances)


def remove_outliers(feats, q, exact=False, max_threshold=None):
    if q == 1:
        return feats

    if exact:
        # sorts the whole tensor and gets the q-th percentile
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        # quantile for element in the tensor and take the average
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()

    if max_threshold is not None:
        max_threshold = max(max_threshold, q_val)

    return feats.clamp(-q_val, q_val)



if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    feats_A = torch.randn(64, 8192)
    feats_B = torch.randn(64, 8192)
    feats_A = F.normalize(feats_A, dim=-1)
    feats_B = F.normalize(feats_B, dim=-1)

    import time 
    trials = 10

    t0 = time.time()
    for metric in AlignmentMetrics.SUPPORTED_METRICS:

        scores, times = [], []
        for t in range(trials):
            t_st = time.time()

            kwargs = {}
            if 'nn' in metric:
                kwargs['topk'] = 10
            if 'cca' in metric:
                kwargs['cca_dim'] = 10
            if 'kernel' in metric:
                kwargs['dist'] = 'sample'

            score = AlignmentMetrics.measure(metric, feats_A, feats_B, **kwargs)
            scores.append(score)
            times.append(time.time() - t_st)        
        print(f"{metric.rjust(20)}: {np.mean(scores):1.3f} [elapsed: {np.mean(times):.2f}s]")

    print(f'Total time: {time.time() - t0:.2f}s')
