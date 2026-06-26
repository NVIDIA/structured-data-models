import torch
from torch import Tensor


def _ensure_floating(input: Tensor) -> Tensor:
    """Return a floating-point view of ``input`` for numeric transforms."""
    if input.is_floating_point():
        return input
    return input.to(dtype=torch.get_default_dtype())
