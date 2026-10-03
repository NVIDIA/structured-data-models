# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from sdm.models.timesfm3.core import _TimesFM3Model
from sdm.testing import withCUDA


def _residual_config(output_dims: int = 8) -> dict[str, Any]:
    return {
        "hidden_dims": 8,
        "output_dims": output_dims,
        "use_bias": False,
    }


def _transformer_config(model_dims: int = 8) -> dict[str, Any]:
    return {
        "num_layers": 2,
        "transformer": {
            "model_dims": model_dims,
            "hidden_dims": 12,
            "num_heads": 2,
            "qk_norm": "rms",
            "use_rope_seq": True,
            "use_rope_var": False,
            "use_bias": False,
        },
    }


def _internal_model(
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    *,
    use_iterative_cpm_revin: bool = True,
    use_linear_detrending: bool = True,
) -> _TimesFM3Model:
    return _TimesFM3Model(
        input_patch_len=2,
        output_patch_len=4,
        quantiles=[0.1, 0.5, 0.9],
        residual_block_config=_residual_config(),
        transformer_config=_transformer_config(),
        use_iterative_cpm_revin=use_iterative_cpm_revin,
        use_linear_detrending=use_linear_detrending,
        device=device,
        dtype=dtype,
    )


@withCUDA
def test_internal_model_assembles_checkpoint_parameters(
    device: torch.device,
) -> None:
    source = _internal_model(device)
    state = source.state_dict()
    assert state["pre_transformer_resblock.hidden_layer.weight"].shape == (
        8,
        12,
    )
    assert state[
        "transformer_stack.layers.0.seq_attn.qkv_lin.weight"
    ].shape == (
        24,
        8,
    )
    assert state["output_head.weight"].shape == (12, 8)

    loaded = _internal_model("meta")
    loaded.load_state_dict(state, strict=True, assign=True)
    assert all(parameter.device == device for parameter in loaded.parameters())
    torch.testing.assert_close(
        loaded.output_head.weight, source.output_head.weight
    )


@withCUDA
def test_internal_model_preprocesses_patches(device: torch.device) -> None:
    model = _internal_model(device).eval()
    values = torch.tensor(
        [[[[1.0, 3.0], [10.0, 14.0], [100.0, 200.0]]]],
        device=device,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.ones(1, 1, 3, dtype=torch.bool, device=device)

    residual_input, transformer_input, patch_mask, stats, counts = (
        model._preprocess(
            values,
            masks,
            patch_is_target,
            freeze_after=0,
        )
    )

    running_mean, running_std = stats
    assert residual_input.shape == (1, 1, 3, 12)
    assert transformer_input.shape == (1, 1, 3, 8)
    assert patch_mask.shape == (1, 1, 3)
    torch.testing.assert_close(
        running_mean,
        torch.tensor([[[2.0, 2.0, 2.0]]], device=device),
    )
    torch.testing.assert_close(
        running_std,
        torch.tensor([[[1.0, 1.0, 1.0]]], device=device),
    )
    torch.testing.assert_close(
        counts,
        torch.tensor([[[2.0, 4.0, 6.0]]], device=device),
    )
    torch.testing.assert_close(
        residual_input[0, 0, 1, :2],
        torch.tensor([8.0, 12.0], device=device),
    )
    assert residual_input[..., 8:].bool().all()


@withCUDA
def test_internal_model_preserves_covariates_under_cpm(
    device: torch.device,
) -> None:
    model = _internal_model(device).eval()
    values = torch.tensor(
        [
            [
                [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]],
                [[10.0, 12.0], [10.0, 12.0], [10.0, 12.0]],
            ]
        ],
        device=device,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.tensor([[[True] * 3, [False] * 3]], device=device)
    patch_cpm_mask = torch.tensor([[False, True, False]], device=device)

    residual_input, _, patch_mask, stats, counts = model._preprocess(
        values,
        masks,
        patch_is_target,
        patch_cpm_mask=patch_cpm_mask,
    )
    running_mean, _ = stats

    assert running_mean[0, 0, 1] == 3.0
    assert counts[0, 0, 1] == 4.0
    # Current values, future values, current masks, then future masks.
    assert not residual_input[0, 0, 1, :6].bool().any()
    torch.testing.assert_close(
        residual_input[0, 1, 0, 2:6],
        torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device),
    )
    torch.testing.assert_close(
        residual_input[0, 1, 1, 2:6],
        torch.tensor([-1.0, 1.0, 0.0, 0.0], device=device),
    )
    assert residual_input[0, 0, :, 8:12].bool().all()
    assert not residual_input[0, 1, 1, 8:10].bool().any()
    assert residual_input[0, 1, 1, 10:12].bool().all()
    assert residual_input[0, 1, 2, 8:12].bool().all()
    expected_patch_mask = torch.tensor(
        [[[False, True, False], [False, False, False]]], device=device
    )
    torch.testing.assert_close(patch_mask, expected_patch_mask)
