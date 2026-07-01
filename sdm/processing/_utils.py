"""Shared tensor conversion helpers for processing transforms."""

import torch
from torch import Tensor


def _as_float(input: Tensor) -> Tensor:
    """Return a floating-point tensor for numeric transforms.

    Floating-point inputs are returned unchanged; integer inputs are promoted
    to the default floating dtype.
    """
    if input.is_floating_point():
        return input
    return input.to(dtype=torch.get_default_dtype())
