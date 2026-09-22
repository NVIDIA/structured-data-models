# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

_TOLERANCE = 1e-6


def _make_safe_for_division(values: Tensor) -> Tensor:
    return torch.where(values < _TOLERANCE, 1.0, values)


def _make_safe_for_sqrt(values: Tensor) -> Tensor:
    is_zero = values == 0
    safe_values = torch.where(is_zero, 1.0, values)
    return torch.where(is_zero, 0.0, safe_values.sqrt())


def update_running_stats(
    n: Tensor,
    mu: Tensor,
    sigma: Tensor,
    x: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Update running statistics with a new patch.

    Args:
        n: Counts of valid values with shape ``[..., V]``, where ``V`` is
            the number of variates.
        mu: Running means with shape ``[..., V]``.
        sigma: Running population standard deviations with shape
            ``[..., V]``.
        x: New values with shape ``[..., V, P]``, where ``P`` is the patch
            length.
        mask: Invalid-value mask with shape ``[..., V, P]``.

    Returns:
        Updated counts, means, and population standard deviations, each with
        shape ``[..., V]``.
    """
    is_valid = ~mask
    valid = is_valid.float()
    inc_n = valid.sum(dim=-1)
    safe_inc_n = _make_safe_for_division(inc_n)

    inc_sum = torch.where(is_valid, x, 0.0).sum(dim=-1)
    inc_mu = torch.where(inc_n == 0, 0.0, inc_sum / safe_inc_n)
    inc_var = torch.where(
        inc_n == 0,
        0.0,
        torch.where(is_valid, (x - inc_mu.unsqueeze(-1)).square(), 0.0).sum(
            dim=-1
        )
        / safe_inc_n,
    )
    inc_sigma = _make_safe_for_sqrt(inc_var)

    new_n = n + inc_n
    safe_new_n = _make_safe_for_division(new_n)
    new_mu = torch.where(
        new_n == 0,
        0.0,
        (n * mu + inc_n * inc_mu) / safe_new_n,
    )
    new_var = torch.where(
        new_n == 0,
        0.0,
        (
            n * sigma * sigma
            + inc_n * inc_sigma * inc_sigma
            + n * (mu - new_mu) * (mu - new_mu)
            + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)
        )
        / safe_new_n,
    )
    new_sigma = _make_safe_for_sqrt(new_var)
    return new_n, new_mu, new_sigma


def get_running_stats(
    values: Tensor,
    masks: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute cumulative statistics patch by patch.

    Args:
        values: Input values with shape ``[B, V, N, P]``, where ``B`` is
            the batch size, ``V`` is the number of variates, ``N`` is the
            number of patches, and ``P`` is the patch length.
        masks: Invalid-value mask with shape ``[B, V, N, P]``.

    Returns:
        Cumulative counts, means, and population standard deviations, each
        with shape ``[B, V, N]``.
    """
    batch_size, num_variates, num_patches, _ = values.size()
    current_n = torch.zeros(
        (batch_size, num_variates),
        dtype=torch.float32,
        device=values.device,
    )
    current_mu = torch.zeros_like(current_n)
    current_sigma = torch.zeros_like(current_n)

    all_n = []
    all_mu = []
    all_sigma = []
    for index in range(num_patches):
        current_n, current_mu, current_sigma = update_running_stats(
            current_n,
            current_mu,
            current_sigma,
            values[:, :, index, :],
            masks[:, :, index, :],
        )
        all_n.append(current_n)
        all_mu.append(current_mu)
        all_sigma.append(current_sigma)

    return (
        torch.stack(all_n, dim=2),
        torch.stack(all_mu, dim=2),
        torch.stack(all_sigma, dim=2),
    )


def revin(
    x: Tensor,
    mu: Tensor,
    sigma: Tensor,
    reverse: bool = False,
) -> Tensor:
    r"""Apply the statistical transform described by the `RevIN paper`_.

    This helper uses supplied statistics and omits RevIN's learnable affine
    transformation.

    Args:
        x: Input values with shape ``[..., D]`` or ``[..., D, Q]``, where
            ``D`` is the number of values and ``Q`` is the optional number of
            prediction channels.
        mu: Means with one or two fewer dimensions than ``x``.
        sigma: Standard deviations with the same shape as ``mu``.
        reverse: Whether to denormalize instead of normalize.

    Returns:
        Transformed values with the same shape as ``x``.

    .. _RevIN paper: https://openreview.net/forum?id=cGDAkQo1C0p
    """
    if mu.shape != sigma.shape:
        raise ValueError(
            "mu and sigma must have the same shape, got "
            f"{mu.shape} and {sigma.shape}."
        )

    if mu.dim() not in (x.dim() - 1, x.dim() - 2):
        raise ValueError(
            f"Unsupported shapes for x and mu: {x.shape}, {mu.shape}."
        )
    if mu.shape != x.shape[: mu.dim()]:
        raise ValueError(
            "mu and sigma must match the leading dimensions of x, got "
            f"x.shape={x.shape} and stats shape={mu.shape}."
        )

    if mu.dim() == x.dim() - 1:
        mu = mu.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    else:
        mu = mu.unsqueeze(-1).unsqueeze(-1)
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)

    if reverse:
        return x * sigma + mu
    return (x - mu) / _make_safe_for_division(sigma)
