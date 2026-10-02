# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.timesfm3.configs import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TransformerConfig,
)
from sdm.models.timesfm3.core import _TimesFM3Model
from sdm.models.timesfm3.util import get_running_stats
from sdm.testing import withCUDA


def _residual_config(output_dims: int = 8) -> ResidualBlockConfig:
    return ResidualBlockConfig(
        hidden_dims=8,
        output_dims=output_dims,
        use_bias=False,
        activation="relu",
    )


def _transformer_config(model_dims: int = 8) -> StackedTransformersConfig:
    return StackedTransformersConfig(
        num_layers=2,
        transformer=TransformerConfig(
            model_dims=model_dims,
            hidden_dims=12,
            num_heads=2,
            qk_norm="rms",
            use_rope_seq=True,
            use_rope_var=False,
            use_bias=False,
            ff_activation="relu",
        ),
    )


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
        "transformer_stack.layers.0.seq_attn.query_proj.weight"
    ].shape == (
        8,
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


@withCUDA
def test_internal_model_forward(device: torch.device) -> None:
    model = _internal_model(device).eval()
    values = torch.arange(
        2 * 3 * 4 * 2,
        device=device,
        dtype=torch.float32,
    ).reshape(2, 3, 4, 2)
    masks = torch.zeros_like(values, dtype=torch.bool)
    masks[:, :, 0] = True
    masks[:, :, 2] = True
    patch_is_target = torch.ones(2, 3, 4, dtype=torch.bool, device=device)

    with torch.inference_mode():
        outputs = model(
            values,
            masks,
            patch_is_target,
            return_aux_outputs=True,
        )

    assert outputs["logits"].shape == (2, 3, 4, 4, 3)
    assert torch.isfinite(outputs["logits"]).all()
    assert outputs["__call__:resblock_input"].shape == (2, 3, 4, 12)
    assert outputs["__call__:transformer_input"].shape == (2, 3, 4, 8)
    assert outputs["__call__:transformer_output"].shape == (2, 3, 4, 8)

    attention_masks = outputs["__call__:seq_attn_mask"]
    assert len(attention_masks) == 2
    expected_mask = torch.tensor(
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, True, True, False],
            [False, True, True, True],
        ],
        device=device,
    )
    torch.testing.assert_close(attention_masks[0][0, 0], expected_mask)


@withCUDA
def test_internal_model_denormalizes_output_head(
    device: torch.device,
) -> None:
    model = _internal_model(
        device,
        use_iterative_cpm_revin=False,
    ).eval()
    normalized_logits = (
        torch.arange(
            12,
            device=device,
            dtype=torch.float32,
        ).reshape(4, 3)
        / 10.0
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output_head.bias.copy_(normalized_logits.flatten())

    values = torch.tensor(
        [[[[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]]],
        device=device,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.ones(1, 1, 3, dtype=torch.bool, device=device)

    with torch.inference_mode():
        logits = model(values, masks, patch_is_target)["logits"]

    _, running_mean, running_std = get_running_stats(values, masks)
    expected = (
        normalized_logits[None, None, None] * running_std[..., None, None]
        + running_mean[..., None, None]
    )
    torch.testing.assert_close(logits, expected)


@withCUDA
def test_internal_model_bfloat16(device: torch.device) -> None:
    model = _internal_model(device, dtype=torch.bfloat16).eval()
    values = torch.arange(
        2 * 2 * 3 * 2,
        device=device,
        dtype=torch.float32,
    ).reshape(2, 2, 3, 2)
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.ones(2, 2, 3, dtype=torch.bool, device=device)

    with torch.inference_mode():
        outputs = model(
            values,
            masks,
            patch_is_target,
            return_aux_outputs=True,
        )

    assert outputs["__call__:resblock_input"].dtype == torch.bfloat16
    assert outputs["__call__:transformer_output"].dtype == torch.bfloat16
    assert torch.isfinite(outputs["logits"]).all()


@withCUDA
def test_internal_model_loads_from_meta(device: torch.device) -> None:
    expected = _internal_model(device).eval()
    model = _internal_model("meta").eval()
    assert all(
        parameter.device.type == "meta" for parameter in model.parameters()
    )

    model.load_state_dict(expected.state_dict(), assign=True)
    model.eval()
    values = torch.arange(
        2 * 2 * 3 * 2,
        device=device,
        dtype=torch.float32,
    ).reshape(2, 2, 3, 2)
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.ones(2, 2, 3, dtype=torch.bool, device=device)

    with torch.inference_mode():
        actual_logits = model(values, masks, patch_is_target)["logits"]
        expected_logits = expected(values, masks, patch_is_target)["logits"]

    assert all(parameter.device == device for parameter in model.parameters())
    assert all(buffer.device == device for buffer in model.buffers())
    torch.testing.assert_close(actual_logits, expected_logits)


def test_internal_model_rejects_incompatible_configuration() -> None:
    with pytest.raises(ValueError, match="must be a multiple"):
        _TimesFM3Model(
            input_patch_len=2,
            output_patch_len=3,
            residual_block_config=_residual_config(),
            transformer_config=_transformer_config(),
            use_stitching=False,
        )

    with pytest.raises(ValueError, match="dimensions must match"):
        _TimesFM3Model(
            input_patch_len=2,
            output_patch_len=4,
            residual_block_config=_residual_config(output_dims=4),
            transformer_config=_transformer_config(model_dims=8),
        )

    with pytest.raises(ValueError, match="Stitching requires"):
        _TimesFM3Model(
            input_patch_len=2,
            output_patch_len=2,
            residual_block_config=_residual_config(),
            transformer_config=_transformer_config(),
        )

    model = _internal_model("cpu")
    with pytest.raises(ValueError, match="does not match"):
        model(
            torch.zeros(1, 1, 1, 3),
            torch.zeros(1, 1, 1, 3, dtype=torch.bool),
            torch.ones(1, 1, 1, dtype=torch.bool),
        )


@withCUDA
def test_internal_model_sanitizes_inputs(device: torch.device) -> None:
    model = _TimesFM3Model(
        input_patch_len=2,
        output_patch_len=4,
        quantiles=[0.1, 0.5, 0.9],
        residual_block_config=_residual_config(),
        transformer_config=_transformer_config(),
        value_clip=5.0,
        device=device,
    ).eval()
    values = torch.tensor(
        [[[[float("nan"), float("inf")], [float("-inf"), 10.0]]]],
        device=device,
    )
    sanitized = torch.tensor(
        [[[[0.0, 5.0], [-5.0, 5.0]]]],
        device=device,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.ones(1, 1, 2, dtype=torch.bool, device=device)

    with torch.inference_mode():
        actual = model(
            values,
            masks,
            patch_is_target,
            return_aux_outputs=True,
        )
        expected = model(
            sanitized,
            masks,
            patch_is_target,
            return_aux_outputs=True,
        )

    torch.testing.assert_close(actual["logits"], expected["logits"])
    torch.testing.assert_close(
        actual["__call__:resblock_input"],
        expected["__call__:resblock_input"],
    )


@withCUDA
def test_internal_model_applies_cpm_revin_refinement(
    device: torch.device,
) -> None:
    refined_model = _internal_model(
        device,
        use_iterative_cpm_revin=True,
    ).eval()
    frozen_model = _internal_model(
        device,
        use_iterative_cpm_revin=False,
    ).eval()
    normalized_logits = (
        torch.arange(
            12,
            device=device,
            dtype=torch.float32,
        )
        / 10.0
    )
    with torch.no_grad():
        for parameter in refined_model.parameters():
            parameter.zero_()
        refined_model.output_head.bias.copy_(normalized_logits)
    frozen_model.load_state_dict(refined_model.state_dict())

    values = torch.tensor(
        [[[[1.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]],
        device=device,
    )
    masks = torch.tensor(
        [[[[False, False], [True, True], [True, True], [True, True]]]],
        device=device,
    )
    patch_is_target = torch.ones(1, 1, 4, dtype=torch.bool, device=device)
    patch_cpm_mask = torch.tensor(
        [[False, True, True, True]],
        device=device,
    )

    with torch.inference_mode():
        refined = refined_model(
            values,
            masks,
            patch_is_target,
            patch_cpm_mask=patch_cpm_mask,
        )["logits"]
        frozen = frozen_model(
            values,
            masks,
            patch_is_target,
            patch_cpm_mask=patch_cpm_mask,
        )["logits"]

    torch.testing.assert_close(refined[:, :, 0], frozen[:, :, 0])
    assert not torch.equal(refined[:, :, 1:], frozen[:, :, 1:])
