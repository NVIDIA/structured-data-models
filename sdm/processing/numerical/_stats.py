# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor


def _ndtri(x: Tensor) -> Tensor:
    # MPS does not implement 'torch.special.ndtri', so use 'erfinv' instead.
    if x.device.type == "mps":
        return x.mul(2.0).sub_(1.0).erfinv_().mul_(math.sqrt(2.0))
    return torch.special.ndtri(x)


def _isfinite(x: Tensor) -> Tensor:
    # Equal to 'x.isfinite()', which allocates 'x.abs()' on the way.
    return x.gt(-math.inf).logical_and_(x.lt(math.inf))


def _constant_feature_mask(
    var: Tensor,
    mean: Tensor,
    num_samples: int | Tensor,
) -> Tensor:
    eps = torch.finfo(var.dtype).eps
    upper_bound = num_samples * eps * var + (num_samples * mean * eps) ** 2
    return var <= upper_bound


@torch.library.custom_op("sdm::fitting_nansum", mutates_args=())
def _fitting_nansum(inp: Tensor) -> Tensor:
    return inp.nansum(dim=-2, keepdim=True)


@_fitting_nansum.register_fake
def _fitting_nansum_fake(inp: Tensor) -> Tensor:
    return inp.new_empty((*inp.shape[:-2], 1, inp.size(-1)))


@torch.library.custom_op("sdm::fitting_nanmean", mutates_args=())
def _fitting_nanmean(inp: Tensor) -> Tensor:
    return inp.nanmean(dim=-2, keepdim=True)


@_fitting_nanmean.register_fake
def _fitting_nanmean_fake(inp: Tensor) -> Tensor:
    return inp.new_empty((*inp.shape[:-2], 1, inp.size(-1)))


def _nansum_rows(inp: Tensor) -> Tensor:
    # Native reductions keep fitted power-search comparisons consistent.
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        return _fitting_nansum(inp)
    return inp.nansum(dim=-2, keepdim=True)


def _nanmean_rows(inp: Tensor) -> Tensor:
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        return _fitting_nanmean(inp)
    return inp.nanmean(dim=-2, keepdim=True)
