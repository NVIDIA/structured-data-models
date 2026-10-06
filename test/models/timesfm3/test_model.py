# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.models.timesfm3 import TimesFM3
from sdm.models.timesfm3.model import _TimesFM3
from sdm.testing import withCUDA


def test_forward() -> None:
    model = TimesFM3(pretrained=False)

    # Past and past-and-future covariates:
    x_context = TableTensor.from_tensor(
        torch.randn(5, 4),
        columns=["x1", "x2", "x3", "x4"],
    )
    x_query = TableTensor.from_tensor(
        torch.randn(3, 2),
        columns=["x2", "x4"],
    )

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(5, 3),
        columns=["y1", "y2", "y3"],
    )

    out = model(x_context, y_context, x_query)
    assert out.size() == (3, 3 * 9)
    assert "y1__q10" in out.columns["numerical"]
    assert "y2__q50" in out.columns["numerical"]
    assert "y3__q90" in out.columns["numerical"]

    model.fit(x_context, y_context)
    out = model.predict(x_query)
    assert out.allclose(model.predict(x_query))


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_preprocess_freezes_running_statistics(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model = _TimesFM3(
        input_patch_len=2,
        output_patch_len=4,
        channels=8,
        num_layers=1,
        num_heads=2,
        device=device,
        dtype=dtype,
    )
    values = torch.tensor(
        [[[[1.0, 3.0], [10.0, 14.0], [100.0, 200.0]]]],
        device=device,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.ones(1, 1, 3, dtype=torch.bool, device=device)

    embeddings, residual_input, patch_mask, stats = model._preprocess(
        values, masks, patch_is_target, freeze_after=0
    )

    counts, running_mean, running_std = stats
    assert residual_input.shape == (1, 1, 3, 12)
    assert residual_input.dtype == dtype
    assert embeddings.shape == (1, 1, 3, 8)
    assert embeddings.dtype == dtype
    assert embeddings.isfinite().all()
    assert not patch_mask.any()
    torch.testing.assert_close(
        running_mean, torch.tensor([[[2.0, 2.0, 2.0]]], device=device)
    )
    torch.testing.assert_close(
        running_std, torch.tensor([[[1.0, 1.0, 1.0]]], device=device)
    )
    torch.testing.assert_close(
        counts, torch.tensor([[[2, 4, 6]]], device=device)
    )
    torch.testing.assert_close(
        residual_input[0, 0, 1, :2], residual_input.new_tensor([8.0, 12.0])
    )
    assert not residual_input[..., 2:6].bool().any()


@withCUDA
def test_preprocess_masks_targets_but_keeps_future_covariates(
    device: torch.device,
) -> None:
    model = _TimesFM3(
        input_patch_len=2,
        output_patch_len=4,
        channels=8,
        num_layers=1,
        num_heads=2,
        device=device,
    )
    values = torch.tensor(
        [
            [
                [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]],
                [[10.0, 12.0], [14.0, 16.0], [18.0, 20.0]],
            ]
        ],
        device=device,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    masks[0, 1, 1, 1] = True
    patch_is_target = torch.tensor([[[True] * 3, [False] * 3]], device=device)
    cpm_mask = torch.tensor([[False, True, False]], device=device)

    _, residual_input, patch_mask, stats = model._preprocess(
        values, masks, patch_is_target, cpm_mask=cpm_mask
    )

    counts, running_mean, _ = stats
    assert running_mean[0, 0, 1] == 3.0
    assert counts[0, 0, 1] == 4
    assert counts[0, 1, 1] == 3
    assert not residual_input[0, 0, 1, :6].bool().any()
    assert not residual_input[0, 0, :, 2:6].bool().any()
    torch.testing.assert_close(
        residual_input[0, 1, 0, 2:6],
        torch.tensor([3.0, 0.0, 7.0, 9.0], device=device),
    )
    assert torch.equal(
        residual_input[0, 1, 0, 8:12].bool(),
        torch.tensor([False, True, False, False], device=device),
    )
    assert residual_input[0, 1, 1, 10:12].bool().all()
    assert residual_input[0, 1, 2, 8:12].bool().all()
    assert torch.equal(
        patch_mask,
        torch.tensor(
            [[[False, True, False], [False, False, False]]],
            device=device,
        ),
    )
