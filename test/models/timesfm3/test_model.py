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
def test_preprocess_embeds_current_and_known_next_patches(
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
    x = torch.tensor(
        [
            [
                [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]],
                [[10.0, 12.0], [14.0, 16.0], [18.0, 20.0]],
            ]
        ],
        device=device,
    )
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, 0, 2] = True
    mask[0, 1, 1, 1] = True
    context_only = torch.tensor([[[True] * 3, [False] * 3]], device=device)

    embeddings, patch_features, empty_patch_mask, (count, mean, std) = (
        model._preprocess(x=x, mask=mask, context_only=context_only)
    )

    assert embeddings.shape == (1, 2, 3, 8)
    assert embeddings.dtype == dtype
    assert embeddings.isfinite().all()
    assert patch_features.shape == (1, 2, 3, 12)
    assert patch_features.dtype == dtype
    assert (count[0, 0] == torch.tensor([2, 4, 4], device=device)).all()
    assert mean[0, 0, 1] == 3.0
    assert std.isfinite().all()
    assert patch_features[0, 0, 1, :2].abs().sum() > 0
    assert not patch_features[0, 0, :, 2:6].bool().any()
    assert patch_features[0, 0, :, 8:12].bool().all()
    torch.testing.assert_close(
        patch_features[0, 1, 0, 2:6],
        torch.tensor([3.0, 0.0, 7.0, 9.0], device=device, dtype=dtype),
    )
    assert torch.equal(
        patch_features[0, 1, 0, 8:12].bool(),
        torch.tensor([False, True, False, False], device=device),
    )
    assert patch_features[0, 1, 1, 10:12].bool().all()
    assert patch_features[0, 1, 2, 8:12].bool().all()
    assert torch.equal(
        empty_patch_mask,
        torch.tensor(
            [[[False, False, True], [False, False, False]]],
            device=device,
        ),
    )


def test_preprocess_preserves_input_gradients() -> None:
    model = _TimesFM3(
        input_patch_len=2,
        output_patch_len=4,
        channels=8,
        num_layers=1,
        num_heads=2,
    )
    values = torch.tensor(
        [[[[1.0, 3.0], [2.0, 6.0], [5.0, 7.0]]]],
        requires_grad=True,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    context_only = torch.ones_like(values[..., 0], dtype=torch.bool)

    embeddings, *_ = model._preprocess(
        x=values, mask=masks, context_only=context_only
    )
    (grad,) = torch.autograd.grad(embeddings.square().sum(), values)

    assert grad.isfinite().all()


def test_preprocess_accepts_leading_batch_dimensions() -> None:
    model = _TimesFM3(
        input_patch_len=2,
        output_patch_len=4,
        channels=8,
        num_layers=1,
        num_heads=2,
    )
    x = torch.tensor([[[[1.0, 3.0], [10.0, 14.0], [100.0, 200.0]]]]).expand(
        2, 3, -1, -1, -1
    )
    mask = torch.zeros_like(x, dtype=torch.bool)
    context_only = torch.ones_like(x[..., 0], dtype=torch.bool)

    embeddings, patch_features, empty_patch_mask, (count, mean, std) = (
        model._preprocess(x=x, mask=mask, context_only=context_only)
    )

    assert embeddings.shape == (2, 3, 1, 3, 8)
    assert patch_features.shape == (2, 3, 1, 3, 12)
    assert empty_patch_mask.shape == (2, 3, 1, 3)
    assert not empty_patch_mask.any()
    assert (count[..., 2] == 6).all()
    assert mean.shape == std.shape == count.shape
