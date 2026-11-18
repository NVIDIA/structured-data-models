# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor


def identity(x: Tensor) -> Tensor:
    """Return the input values unchanged."""
    return x


def heaviside(x: Tensor) -> Tensor:
    """Map positive values to ``1`` and others to ``0`` elementwise."""
    return x.heaviside(torch.zeros((), dtype=x.dtype, device=x.device))


def gaussian(x: Tensor) -> Tensor:
    """Apply the Gaussian bump ``exp(-x ** 2)`` elementwise."""
    return (-x.pow(2)).exp()


def unit_indicator(x: Tensor) -> Tensor:
    """Map values in ``[0, 1]`` to ``1`` and others to ``0`` elementwise."""
    return ((x >= 0) & (x <= 1)).to(x.dtype)


def softmax(x: Tensor) -> Tensor:
    """Apply the softmax along the last dimension."""
    return x.softmax(dim=-1)


def one_hot_argmax(x: Tensor) -> Tensor:
    """One-hot encode the argmax along the last dimension.

    Ties select the first maximum.
    """
    return F.one_hot(x.argmax(dim=-1), num_classes=x.size(-1)).to(x.dtype)


def argsort(x: Tensor) -> Tensor:
    """Return sorting indices along the last dimension."""
    return x.argsort(dim=-1).to(x.dtype)


def log_abs(x: Tensor) -> Tensor:
    """Apply ``log(max(abs(x), 1e-6))`` elementwise."""
    return x.abs().clamp(min=1e-6).log()


def rank(x: Tensor) -> Tensor:
    """Return zero-based ranks along the last dimension."""
    return x.argsort(dim=-1).argsort(dim=-1).to(x.dtype)


def modulo(x: Tensor) -> Tensor:
    """Return the fractional part ``x - floor(x)`` elementwise."""
    return x % 1.0
