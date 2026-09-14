import torch
from torch import Tensor


def _constant_feature_mask(
    var: Tensor,
    mean: Tensor,
    num_samples: int | Tensor,
) -> Tensor:
    eps = torch.finfo(var.dtype).eps
    upper_bound = num_samples * eps * var + (num_samples * mean * eps) ** 2
    return var <= upper_bound
