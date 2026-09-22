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

from sdm.models.timesfm3.util import revin, update_running_stats


def cpm_iterative_revin_refine(
    raw_logits: Tensor,
    revin_n: Tensor,
    revin_mu: Tensor,
    revin_sigma: Tensor,
    patch_cpm_mask: Tensor,
    median_q_idx: int,
    rolls: int,
    patch_len: int,
    num_quantiles: int,
    value_clip: float = 1e9,
) -> tuple[Tensor, Tensor]:
    """Refine RevIN statistics for Contiguous Patch Masking (CPM).

    Each masked position incorporates model-estimated values from preceding
    CPM patches in its block. Statistics at unmasked positions are unchanged.

    Args:
        raw_logits: Normalized predictions with shape ``[B, V, N, O * Q]``,
            where ``B`` is the batch size, ``V`` is the number of variates,
            ``N`` is the number of patches, ``O`` is the output patch length,
            and ``Q`` is the number of quantiles.
        revin_n: Running counts with shape ``[B, V, N]``.
        revin_mu: Running means with shape ``[B, V, N]``.
        revin_sigma: Running standard deviations with shape ``[B, V, N]``.
        patch_cpm_mask: Contiguous Patch Masking indicator with shape
            ``[B, N]``.
        median_q_idx: Index of the median quantile used as the point estimate.
        rolls: Number of input patches covered by each output patch.
        patch_len: Input patch length.
        num_quantiles: Number of predicted quantiles.
        value_clip: Absolute bound applied to estimated values.

    Returns:
        Refined means and standard deviations, each with shape ``[B, V, N]``.
    """
    batch_size, num_variates, num_patches, _ = raw_logits.shape
    device = raw_logits.device

    # [B, V, N, O * Q] -> [B, V, N, R, P]
    median_logits = raw_logits.reshape(
        batch_size,
        num_variates,
        num_patches,
        rolls,
        patch_len,
        num_quantiles,
    )[..., median_q_idx]

    carry_n = torch.zeros(
        (batch_size, num_variates),
        dtype=torch.float32,
        device=device,
    )
    carry_mu = torch.zeros_like(carry_n)
    carry_sigma = torch.zeros_like(carry_n)
    anchor_predicted_values = torch.zeros(
        (batch_size, num_variates, rolls, patch_len),
        dtype=torch.float32,
        device=device,
    )
    block_offset = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device,
    )
    step_masks = torch.zeros(
        (batch_size, num_variates, patch_len),
        dtype=torch.bool,
        device=device,
    )

    refined_mu = []
    refined_sigma = []
    for index in range(num_patches):
        actual_n = revin_n[:, :, index]
        actual_mu = revin_mu[:, :, index]
        actual_sigma = revin_sigma[:, :, index]
        current_step_logits = median_logits[:, :, index]
        is_cpm = patch_cpm_mask[:, index : index + 1]

        offset_index = block_offset.view(batch_size, 1, 1, 1).expand(
            -1,
            num_variates,
            1,
            patch_len,
        )
        predicted_values_step = anchor_predicted_values.gather(
            2,
            offset_index,
        ).squeeze(2)
        new_n, new_mu, new_sigma = update_running_stats(
            carry_n,
            carry_mu,
            carry_sigma,
            predicted_values_step,
            step_masks,
        )

        out_n = torch.where(is_cpm, new_n, actual_n)
        out_mu = torch.where(is_cpm, new_mu, actual_mu)
        out_sigma = torch.where(is_cpm, new_sigma, actual_sigma)

        new_block_offset = torch.where(
            is_cpm.squeeze(-1),
            (block_offset + 1) % rolls,
            torch.zeros_like(block_offset),
        )
        should_update_anchor = new_block_offset == 0

        step_predicted_values = revin(
            current_step_logits,
            out_mu,
            out_sigma,
            reverse=True,
        ).clamp(-value_clip, value_clip)
        anchor_predicted_values = torch.where(
            should_update_anchor.view(batch_size, 1, 1, 1),
            step_predicted_values,
            anchor_predicted_values,
        )

        carry_n = out_n
        carry_mu = out_mu
        carry_sigma = out_sigma
        block_offset = new_block_offset
        refined_mu.append(out_mu)
        refined_sigma.append(out_sigma)

    return torch.stack(refined_mu, dim=2), torch.stack(refined_sigma, dim=2)
