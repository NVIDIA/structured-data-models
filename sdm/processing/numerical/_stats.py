import torch
from torch import Tensor


def _constant_feature_mask(
    var: Tensor,
    mean: Tensor,
    n_samples: int,
) -> Tensor:
    eps = torch.finfo(var.dtype).eps
    upper_bound = (n_samples * mean).mul_(eps).square_()
    upper_bound.add_(n_samples * eps * var)
    return var <= upper_bound
