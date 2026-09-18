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

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

_TOLERANCE = 1e-6


def _make_safe_for_division(values: Tensor) -> Tensor:
    return torch.where(values < _TOLERANCE, 1.0, values)


@dataclass
class DecodeCache:
    """Store TimesFM-3 state for incremental temporal attention.

    Args:
        next_index: Next insertion index for each flattened batch/variate
            sequence with shape ``[L]``.
        num_front_masked: Number of leading masked patches for each sequence
            with shape ``[L]``.
        key: Key projections with shape ``[L, C, H, D]``, where ``C`` is the
            cache capacity, ``H`` is the number of heads, and ``D`` is the
            head dimension.
        value: Value projections with the same shape as ``key``.
        patch_mask: Cached invalid-patch mask with shape ``[L, C]``.
        segment_ids: Optional cached segment identifiers with shape
            ``[L, C]``.
        filled: Length of the populated cache prefix. ``None`` uses the full
            cache capacity.
    """

    next_index: Tensor
    num_front_masked: Tensor
    key: Tensor
    value: Tensor
    patch_mask: Tensor | None = None
    segment_ids: Tensor | None = None
    filled: int | None = None

    @classmethod
    def init_decode_cache(
        cls,
        num_layers: int,
        batch_size: int,
        num_variates: int,
        num_total_input_patches: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> list["DecodeCache"]:
        """Initialize one decode cache per transformer layer.

        Args:
            num_layers: Number of transformer layers.
            batch_size: Batch size ``B``.
            num_variates: Number of variates ``V``.
            num_total_input_patches: Cache capacity ``C``.
            num_heads: Number of attention heads ``H``.
            head_dim: Dimension ``D`` of each attention head.
            device: Device on which to create the cache tensors.
            dtype: Data type of the key and value tensors.

        Returns:
            Independent caches whose leading dimension is ``B * V``.
        """
        leading_size = batch_size * num_variates
        return [
            cls(
                next_index=torch.zeros(
                    leading_size,
                    dtype=torch.int32,
                    device=device,
                ),
                num_front_masked=torch.zeros(
                    leading_size,
                    dtype=torch.int32,
                    device=device,
                ),
                key=torch.zeros(
                    leading_size,
                    num_total_input_patches,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                value=torch.zeros(
                    leading_size,
                    num_total_input_patches,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                patch_mask=torch.ones(
                    leading_size,
                    num_total_input_patches,
                    dtype=torch.bool,
                    device=device,
                ),
                filled=0,
            )
            for _ in range(num_layers)
        ]

    def append(
        self,
        key: Tensor,
        value: Tensor,
        patch_mask: Tensor | None = None,
        segment_ids: Tensor | None = None,
    ) -> "DecodeCache":
        """Append key/value projections and advance each insertion index.

        The method updates the preallocated key and value tensors in place and
        returns new cursor metadata referencing the same storage.

        Args:
            key: New key projections with shape ``[L, Q, H, D]``.
            value: New value projections with the same shape as ``key``.
            patch_mask: Invalid-patch mask with shape ``[L, Q]``.
            segment_ids: Optional segment identifiers with shape ``[L, Q]``.

        Returns:
            Cache with every insertion index advanced by ``Q``.
        """
        num_sequences, query_length = key.shape[:2]
        sequence_indices = torch.arange(
            num_sequences,
            device=self.next_index.device,
        )
        if patch_mask is None:
            patch_mask = torch.zeros(
                num_sequences,
                query_length,
                dtype=torch.bool,
                device=self.next_index.device,
            )

        cache_patch_mask = self.patch_mask
        if cache_patch_mask is None:
            cache_patch_mask = torch.ones(
                self.key.shape[:2],
                dtype=torch.bool,
                device=self.key.device,
            )

        cache_segment_ids = self.segment_ids
        if segment_ids is not None and cache_segment_ids is None:
            cache_segment_ids = torch.zeros(
                self.key.shape[:2],
                dtype=segment_ids.dtype,
                device=segment_ids.device,
            )

        if query_length == 1:
            self.key[sequence_indices, self.next_index] = key[:, 0]
            self.value[sequence_indices, self.next_index] = value[:, 0]
            cache_patch_mask[sequence_indices, self.next_index] = patch_mask[
                :, 0
            ]
            if segment_ids is not None:
                assert cache_segment_ids is not None
                cache_segment_ids[sequence_indices, self.next_index] = (
                    segment_ids[:, 0]
                )
        else:
            positions = self.next_index[:, None] + torch.arange(
                query_length,
                device=self.next_index.device,
            )
            cache_indices = (sequence_indices[:, None], positions)
            self.key[cache_indices] = key
            self.value[cache_indices] = value
            cache_patch_mask[cache_indices] = patch_mask
            if segment_ids is not None:
                assert cache_segment_ids is not None
                cache_segment_ids[cache_indices] = segment_ids

        leading_masked = patch_mask.int().cumprod(dim=-1).sum(dim=-1)
        leading_masked = leading_masked.to(self.num_front_masked.dtype)
        num_front_masked = torch.where(
            self.next_index == self.num_front_masked,
            self.num_front_masked + leading_masked,
            self.num_front_masked,
        )
        return DecodeCache(
            next_index=self.next_index + query_length,
            num_front_masked=num_front_masked,
            key=self.key,
            value=self.value,
            patch_mask=cache_patch_mask,
            segment_ids=cache_segment_ids,
            filled=None if self.filled is None else self.filled + query_length,
        )


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

    inc_sum = torch.where(is_valid, x, 0.0).sum(dim=-1)
    inc_mu = torch.where(inc_n == 0, 0.0, inc_sum / inc_n)
    inc_var = torch.where(
        inc_n == 0,
        0.0,
        torch.where(is_valid, (x - inc_mu.unsqueeze(-1)).square(), 0.0).sum(
            dim=-1
        )
        / inc_n,
    )
    inc_sigma = inc_var.sqrt()

    new_n = n + inc_n
    new_mu = torch.where(
        new_n == 0,
        0.0,
        (n * mu + inc_n * inc_mu) / new_n,
    )
    new_sigma = torch.where(
        new_n == 0,
        0.0,
        (
            n * sigma * sigma
            + inc_n * inc_sigma * inc_sigma
            + n * (mu - new_mu) * (mu - new_mu)
            + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)
        )
        / new_n,
    ).sqrt()
    return new_n, new_mu, new_sigma


def get_running_stats(
    values: Tensor,
    masks: Tensor,
    *,
    segment_ids: Tensor | None = None,
    initial_stats: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute cumulative statistics patch by patch.

    Statistics reset to ``initial_stats`` at segment boundaries. Segment
    identifiers support packed inputs; callers must apply the same boundaries
    to transformer attention. Standard TimesFM-3 inference does not provide
    segment identifiers.

    Args:
        values: Input values with shape ``[B, V, N, P]``, where ``B`` is
            the batch size, ``V`` is the number of variates, ``N`` is the
            number of patches, and ``P`` is the patch length.
        masks: Invalid-value mask with shape ``[B, V, N, P]``.
        segment_ids: Optional segment identifiers with shape ``[B, N]``.
        initial_stats: Optional initial counts, means, and population
            standard deviations, each with shape ``[B, V]``.

    Returns:
        Cumulative counts, means, and population standard deviations, each
        with shape ``[B, V, N]``.
    """
    batch_size, num_variates, num_patches, _ = values.size()
    if initial_stats is None:
        initial_stats = (
            torch.zeros(
                (batch_size, num_variates),
                dtype=torch.float32,
                device=values.device,
            ),
            torch.zeros(
                (batch_size, num_variates),
                dtype=torch.float32,
                device=values.device,
            ),
            torch.zeros(
                (batch_size, num_variates),
                dtype=torch.float32,
                device=values.device,
            ),
        )

    if segment_ids is None:
        is_new_segment = torch.zeros(
            batch_size,
            num_patches,
            dtype=torch.bool,
            device=values.device,
        )
    else:
        shifted = F.pad(segment_ids[:, :-1], (1, 0), value=-1)
        is_new_segment = segment_ids != shifted

    all_n = []
    all_mu = []
    all_sigma = []
    current_n, current_mu, current_sigma = initial_stats
    initial_n, initial_mu, initial_sigma = initial_stats
    for index in range(num_patches):
        reset = is_new_segment[:, index].unsqueeze(-1)
        current_n = torch.where(reset, initial_n, current_n)
        current_mu = torch.where(reset, initial_mu, current_mu)
        current_sigma = torch.where(reset, initial_sigma, current_sigma)
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


def get_output_patch_via_roll(
    x: Tensor,
    rolls: int,
) -> tuple[Tensor, Tensor]:
    """Create output patches by rolling patched inputs.

    Args:
        x: Patched inputs with shape ``[B, V, N, P]``.
        rolls: Number of future patches in each output patch.

    Returns:
        Rolled values with shape ``[B, V, N, P * rolls]`` and a wrap-around
        mask with shape ``[1, 1, N, P * rolls]``.
    """
    batch_size, num_variates, num_patches, patch_len = x.size()
    patch_indices = torch.arange(num_patches, device=x.device)
    roll_indices = torch.arange(1, rolls + 1, device=x.device)
    source_indices = patch_indices[:, None] + roll_indices[None, :]
    wrap_mask = (source_indices >= num_patches).repeat_interleave(
        patch_len,
        dim=-1,
    )
    source_indices = source_indices % num_patches
    result = x.index_select(2, source_indices.flatten()).reshape(
        batch_size,
        num_variates,
        num_patches,
        rolls * patch_len,
    )

    return result, wrap_mask.unsqueeze(0).unsqueeze(0)


_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "none": lambda x: x,
}


def get_activation_fn(
    activation_name: str,
) -> Callable[[Tensor], Tensor]:
    """Return an activation function by name.

    Args:
        activation_name: Activation name: ``"relu"``, ``"swish"``,
            ``"silu"``, or ``"none"``.

    Returns:
        The corresponding tensor operation.
    """
    try:
        return _ACTIVATIONS[activation_name]
    except KeyError:
        raise ValueError(
            f"Activation: {activation_name} not supported. Supported "
            f"activations: {list(_ACTIVATIONS)}"
        ) from None


def stitch_patches(
    patch_preds: Tensor,
    patch_len: int,
) -> Tensor:
    """Stitch overlapping patch predictions.

    Args:
        patch_preds: Predictions with shape ``[B, V, N, P + O, Q]``, where
            ``O`` is the overlap and ``Q`` is the number of quantiles.
        patch_len: Non-overlapping patch length ``P``.

    Returns:
        Stitched predictions with shape ``[B, V, N * P + O, Q]``.
    """
    batch_size, num_variates, num_patches, total_len, num_quantiles = (
        patch_preds.size()
    )
    overlap = total_len - patch_len
    if num_patches == 1:
        return patch_preds[:, :, 0, :, :]

    stitch_weights = torch.linspace(
        1.0,
        0.0,
        overlap,
        device=patch_preds.device,
        dtype=patch_preds.dtype,
    )[None, None, None, :, None]
    first = patch_preds[:, :, 0, :patch_len, :]
    previous = patch_preds[:, :, :-1, patch_len:, :]
    following = patch_preds[:, :, 1:, :overlap, :]
    stitched = stitch_weights * previous + (1.0 - stitch_weights) * following
    middles = patch_preds[:, :, 1:, overlap:patch_len, :]
    middle = torch.cat((stitched, middles), dim=3).reshape(
        batch_size,
        num_variates,
        (num_patches - 1) * patch_len,
        num_quantiles,
    )
    tail = patch_preds[:, :, -1, patch_len:, :]
    return torch.cat((first, middle, tail), dim=2)
