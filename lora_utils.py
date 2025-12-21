import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    Deterministic LoRALinear:
    - 不在构造函数中随机初始化 A/B；
    - init_lora=True 时仅在训练阶段初始化；
    - 推理加载时不触发随机数，确保可复现。
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16,
                 dropout: float = 0.0, init_lora: bool = False):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(1, rank)

        # 冻结原始线性层权重
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        # 仅在需要时初始化 LoRA 参数
        if rank > 0:
            self.A = nn.Parameter(torch.empty(rank, self.in_features))
            self.B = nn.Parameter(torch.empty(self.out_features, rank))
            if init_lora:
                nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
                nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            x_d = self.dropout(x)
            lora_out = F.linear(F.linear(x_d, self.A), self.B)
            out = out + self.scale * lora_out
        return out



def _name_match(full_name: str, pattern: str) -> bool:
    """在模块路径 token 上匹配。"""
    full = full_name.strip(".")
    pat  = pattern.strip(".")
    if full == pat:
        return True
    if full.endswith("." + pat) or full.startswith(pat + ".") or (("." + pat + ".") in full):
        return True
    tokens = full.split(".")
    return pat in tokens



def _inject_lora_recursive(module: nn.Module, rank: int, alpha: int, target,
                           dropout: float, init_lora: bool, prefix: str = "") -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        # 深度递归
        replaced += _inject_lora_recursive(child, rank, alpha, target, dropout, init_lora, prefix=full_name)

        # 匹配 Linear 层
        if isinstance(child, nn.Linear):
            ok = False
            if target == "all_linear":
                ok = True
            elif isinstance(target, (tuple, list, set)):
                ok = any(_name_match(full_name, pat) for pat in target)

            if ok:
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha,
                                                 dropout=dropout, init_lora=init_lora))
                replaced += 1
    return replaced


def inject_lora(module: nn.Module, rank: int, alpha: int, target="all_linear",
                dropout: float = 0.0, verbose: bool = True, is_train: bool = True):
    """
    自动区分训练 / 推理模式的注入函数。
    - is_train=True: 初始化 A/B（用于训练或继续训练）
    - is_train=False: 不初始化 A/B，仅创建空参数（用于加载推理）
    """
    init_lora = is_train  # 自动控制初始化行为
    replaced = _inject_lora_recursive(module, rank, alpha, target, dropout, init_lora)
    if verbose:
        mode = "train(init)" if is_train else "inference(no-init)"
        print(f"[LoRA] injected {replaced} Linear layers ({mode}, target={target})")
    return module



def extract_lora_state_dict(module: nn.Module):
    lora_sd = {}
    for name, sub in module.named_modules():
        if isinstance(sub, LoRALinear):
            lora_sd[f"{name}.A"] = sub.A.detach().cpu()
            lora_sd[f"{name}.B"] = sub.B.detach().cpu()
            lora_sd[f"{name}.meta.alpha"] = torch.tensor(sub.alpha)
            lora_sd[f"{name}.meta.rank"] = torch.tensor(sub.rank)
    return lora_sd


def load_lora_state_dict(module: nn.Module, lora_sd: dict, strict: bool = False):
    missing = []
    for name, sub in module.named_modules():
        if isinstance(sub, LoRALinear):
            keyA = f"{name}.A"
            keyB = f"{name}.B"
            if keyA in lora_sd and keyB in lora_sd:
                with torch.no_grad():
                    sub.A.copy_(lora_sd[keyA].to(sub.A.device))
                    sub.B.copy_(lora_sd[keyB].to(sub.B.device))
            else:
                missing.append(name)
    if strict and missing:
        raise RuntimeError(f"Missing LoRA weights for modules: {missing}")
    if missing:
        print(f"[LoRA] Warning: missing weights for {len(missing)} modules (not strict).")


def preview_targets(module, target):
    print(">> Will inject LoRA into these Linear layers:")
    for full_name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            if target == "all_linear" or any(_name_match(full_name, pat) for pat in target):
                print(full_name)
