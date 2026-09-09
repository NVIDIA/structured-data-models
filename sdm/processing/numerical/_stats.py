import torch
from torch import Tensor


def _constant_feature_mask(
    var: Tensor,
    mean: Tensor,
    n_samples: int,
) -> Tensor:
    eps = torch.finfo(var.dtype).eps
    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound


def _nanmean_var(inp: Tensor) -> tuple[Tensor, Tensor]:
    """Return population statistics over finite values, or zero if absent."""
    safe = inp.masked_fill(~inp.isfinite(), float("nan"))
    mean = safe.nanmean(dim=-2, keepdim=True)
    mean = torch.where(mean.isnan(), torch.zeros_like(mean), mean)
    var = (safe - mean).square().nanmean(dim=-2, keepdim=True)
    var = torch.where(var.isnan(), torch.zeros_like(var), var)
    return mean, var
