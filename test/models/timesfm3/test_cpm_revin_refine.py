# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.timesfm3.cpm_revin_refine import (
    cpm_iterative_revin_refine,
)
from sdm.testing import withCUDA


def _raw_logits(
    median_logits: torch.Tensor,
    num_quantiles: int,
    median_q_idx: int,
) -> torch.Tensor:
    *leading, rolls, patch_len = median_logits.shape
    logits = median_logits.new_zeros(
        *leading,
        rolls,
        patch_len,
        num_quantiles,
    )
    logits[..., median_q_idx] = median_logits
    return logits.flatten(start_dim=-3)


@withCUDA
def test_cpm_revin_refine_without_mask_is_identity(
    device: torch.device,
) -> None:
    batch_size, num_variates, num_patches = 2, 3, 5
    rolls, patch_len, num_quantiles = 2, 4, 3
    output_dims = rolls * patch_len * num_quantiles
    raw_logits = torch.arange(
        batch_size * num_variates * num_patches * output_dims,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, num_variates, num_patches, output_dims)
    revin_n = torch.full(
        (batch_size, num_variates, num_patches),
        4.0,
        device=device,
    )
    revin_mu = torch.arange(
        batch_size * num_variates * num_patches,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, num_variates, num_patches)
    revin_sigma = revin_mu + 1.0
    patch_cpm_mask = torch.zeros(
        batch_size,
        num_patches,
        dtype=torch.bool,
        device=device,
    )

    refined_mu, refined_sigma = cpm_iterative_revin_refine(
        raw_logits=raw_logits,
        revin_n=revin_n,
        revin_mu=revin_mu,
        revin_sigma=revin_sigma,
        patch_cpm_mask=patch_cpm_mask,
        median_q_idx=1,
        rolls=rolls,
        patch_len=patch_len,
        num_quantiles=num_quantiles,
    )

    torch.testing.assert_close(refined_mu, revin_mu)
    torch.testing.assert_close(refined_sigma, revin_sigma)


@withCUDA
def test_cpm_revin_refine_updates_from_anchor_predictions(
    device: torch.device,
) -> None:
    median_logits = torch.tensor(
        [
            [
                [
                    [[2.0, 4.0], [6.0, 8.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ]
        ],
        device=device,
    )
    raw_logits = _raw_logits(
        median_logits,
        num_quantiles=3,
        median_q_idx=1,
    )
    revin_n = torch.tensor(
        [[[2.0, 2.0, 2.0, 1.0, 1.0]]],
        device=device,
    )
    revin_mu = torch.tensor(
        [[[0.0, 20.0, 30.0, 10.0, 40.0]]],
        device=device,
    )
    revin_sigma = torch.tensor(
        [[[1.0, 2.0, 3.0, 2.0, 4.0]]],
        device=device,
    )
    patch_cpm_mask = torch.tensor(
        [[False, True, True, False, True]],
        device=device,
    )

    refined_mu, refined_sigma = cpm_iterative_revin_refine(
        raw_logits=raw_logits,
        revin_n=revin_n,
        revin_mu=revin_mu,
        revin_sigma=revin_sigma,
        patch_cpm_mask=patch_cpm_mask,
        median_q_idx=1,
        rolls=2,
        patch_len=2,
        num_quantiles=3,
    )

    expected_mu = torch.tensor(
        [[[0.0, 1.5, 10.0 / 3.0, 10.0, 34.0 / 3.0]]],
        device=device,
    )
    expected_sigma = torch.tensor(
        [
            [
                [
                    1.0,
                    3.25**0.5,
                    (83.0 / 9.0) ** 0.5,
                    2.0,
                    (20.0 / 9.0) ** 0.5,
                ]
            ]
        ],
        device=device,
    )
    torch.testing.assert_close(refined_mu, expected_mu)
    torch.testing.assert_close(refined_sigma, expected_sigma)


@withCUDA
def test_cpm_revin_refine_clips_anchor_predictions(
    device: torch.device,
) -> None:
    median_logits = torch.tensor(
        [[[[[100.0]], [[0.0]]]]],
        device=device,
    )
    raw_logits = _raw_logits(
        median_logits,
        num_quantiles=1,
        median_q_idx=0,
    )
    revin_n = torch.ones(1, 1, 2, device=device)
    revin_mu = torch.zeros(1, 1, 2, device=device)
    revin_sigma = torch.ones(1, 1, 2, device=device)
    patch_cpm_mask = torch.tensor([[False, True]], device=device)

    refined_mu, refined_sigma = cpm_iterative_revin_refine(
        raw_logits=raw_logits,
        revin_n=revin_n,
        revin_mu=revin_mu,
        revin_sigma=revin_sigma,
        patch_cpm_mask=patch_cpm_mask,
        median_q_idx=0,
        rolls=1,
        patch_len=1,
        num_quantiles=1,
        value_clip=5.0,
    )

    expected_mu = torch.tensor([[[0.0, 2.5]]], device=device)
    expected_sigma = torch.tensor(
        [[[1.0, 6.75**0.5]]],
        device=device,
    )
    torch.testing.assert_close(refined_mu, expected_mu)
    torch.testing.assert_close(refined_sigma, expected_sigma)


@withCUDA
def test_cpm_revin_refine_tracks_each_batch_independently(
    device: torch.device,
) -> None:
    batch_size, num_patches = 2, 6
    rolls, patch_len, num_quantiles = 2, 2, 3
    output_dims = rolls * patch_len * num_quantiles
    raw_logits = torch.arange(
        batch_size * num_patches * output_dims,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, 1, num_patches, output_dims)
    raw_logits = raw_logits / 10.0
    revin_n = torch.arange(
        1,
        num_patches + 1,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, num_patches)
    revin_n = revin_n.expand(batch_size, -1, -1)
    revin_mu = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]],
        device=device,
    ).expand(batch_size, -1, -1)
    revin_sigma = revin_mu + 1.0
    patch_cpm_mask = torch.tensor(
        [
            [False, True, True, False, True, True],
            [False, False, True, True, True, False],
        ],
        device=device,
    )
    kwargs = {
        "median_q_idx": 1,
        "rolls": rolls,
        "patch_len": patch_len,
        "num_quantiles": num_quantiles,
    }

    batched = cpm_iterative_revin_refine(
        raw_logits=raw_logits,
        revin_n=revin_n,
        revin_mu=revin_mu,
        revin_sigma=revin_sigma,
        patch_cpm_mask=patch_cpm_mask,
        **kwargs,
    )
    independent = [
        cpm_iterative_revin_refine(
            raw_logits=raw_logits[index : index + 1],
            revin_n=revin_n[index : index + 1],
            revin_mu=revin_mu[index : index + 1],
            revin_sigma=revin_sigma[index : index + 1],
            patch_cpm_mask=patch_cpm_mask[index : index + 1],
            **kwargs,
        )
        for index in range(batch_size)
    ]

    expected_mu = torch.cat([result[0] for result in independent])
    expected_sigma = torch.cat([result[1] for result in independent])
    torch.testing.assert_close(batched[0], expected_mu)
    torch.testing.assert_close(batched[1], expected_sigma)
