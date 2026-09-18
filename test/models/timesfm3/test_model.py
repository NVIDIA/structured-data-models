# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.models.timesfm3 import TimesFM3
from sdm.models.timesfm3.configs import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TransformerConfig,
)
from sdm.models.timesfm3.model import _TimesFM3
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
            attention_norm="rms",
            feedforward_norm="rms",
            qk_norm="rms",
            use_rope_seq=True,
            use_rope_var=False,
            use_bias=False,
            ff_activation="relu",
            deterministic=True,
        ),
    )


def _internal_model(
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    *,
    use_iterative_cpm_revin: bool = True,
) -> _TimesFM3:
    return _TimesFM3(
        input_patch_len=2,
        output_patch_len=4,
        quantiles=[0.1, 0.5, 0.9],
        residual_block_config=_residual_config(),
        transformer_config=_transformer_config(),
        use_iterative_cpm_revin=use_iterative_cpm_revin,
        device=device,
        dtype=dtype,
    )


def _golden_model(
    device: torch.device,
    *,
    use_stitching: bool = True,
    use_linear_detrending: bool = True,
    use_frozen_running_stats: bool = False,
) -> _TimesFM3:
    model = _TimesFM3(
        input_patch_len=2,
        output_patch_len=4,
        quantiles=[0.5],
        residual_block_config=ResidualBlockConfig(
            hidden_dims=4,
            output_dims=4,
            use_bias=False,
            activation="relu",
        ),
        transformer_config=StackedTransformersConfig(
            num_layers=1,
            transformer=TransformerConfig(
                model_dims=4,
                hidden_dims=6,
                num_heads=2,
                attention_norm="rms",
                feedforward_norm="rms",
                qk_norm="rms",
                use_rope_seq=True,
                use_rope_var=False,
                use_bias=False,
                ff_activation="relu",
                deterministic=True,
            ),
        ),
        use_variate_attention=False,
        use_stitching=use_stitching,
        use_linear_detrending=use_linear_detrending,
        use_frozen_running_stats=use_frozen_running_stats,
        device=device,
    ).eval()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = (
                (
                    torch.arange(parameter.numel()).reshape(parameter.shape)
                    + 3 * index
                )
                % 19
                - 9
            ) / 32
            parameter.copy_(values.to(device))
    return model


def test_forward() -> None:
    model = TimesFM3(pretrained=False, device="meta")

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
    assert model.predict(x_query).size() == (3, 3 * 9)


@pytest.mark.parametrize(
    ("patch_cpm_mask", "expected_values"),
    [
        (
            None,
            [
                2.1499414443969727,
                2.198551654815674,
                2.5312085151672363,
                2.213311195373535,
                2.203420400619507,
                2.3626081943511963,
                2.4190330505371094,
                2.0732791423797607,
                2.19990611076355,
                2.355543375015259,
                2.4288644790649414,
                2.0759785175323486,
            ],
        ),
        (
            [[False, True, True]],
            [
                2.1499414443969727,
                2.198551654815674,
                2.5312085151672363,
                2.213311195373535,
                2.232093095779419,
                2.3455402851104736,
                2.385751962661743,
                2.1393465995788574,
                2.3029558658599854,
                2.3969945907592773,
                2.4412965774536133,
                2.228076457977295,
            ],
        ),
    ],
)
@withCUDA
def test_internal_model_matches_pinned_upstream(
    device: torch.device,
    patch_cpm_mask: list[list[bool]] | None,
    expected_values: list[float],
) -> None:
    model = _golden_model(device)
    values = torch.tensor(
        [[[[1.0, 3.0], [0.0, 0.0], [0.0, 0.0]]]],
        device=device,
    )
    masks = torch.tensor(
        [[[[False, False], [True, True], [True, True]]]],
        device=device,
    )
    patch_is_target = torch.ones(1, 1, 3, dtype=torch.bool, device=device)
    cpm_mask = (
        None
        if patch_cpm_mask is None
        else torch.tensor(patch_cpm_mask, device=device)
    )

    with torch.inference_mode():
        actual = model(
            values,
            masks,
            patch_is_target,
            patch_cpm_mask=cpm_mask,
        )["logits"]

    # Generated by google-research/timesfm@e31dadd with identical weights.
    expected = actual.new_tensor(expected_values).reshape(1, 1, 3, 4, 1)
    torch.testing.assert_close(actual, expected)


@withCUDA
def test_internal_model_decode_matches_pinned_upstream(
    device: torch.device,
) -> None:
    model = _golden_model(device)
    target = torch.tensor([[[1.0, 2.0, 5.0, 3.0, 8.0]]], device=device)
    past_only_covariates = torch.tensor(
        [[[3.0, 1.0, 4.0, 1.0, 5.0]]],
        device=device,
    )
    past_future_covariates = torch.tensor(
        [[[5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 6.0, 10.0]]],
        device=device,
    )
    mask = torch.tensor(
        [[False, False, False, True, False]],
        device=device,
    )
    past_only_mask = torch.tensor(
        [[[False, True, False, False, False]]],
        device=device,
    )
    past_future_mask = torch.tensor(
        [[[False, False, False, False, False, False, True, False]]],
        device=device,
    )

    result = model.decode(
        target,
        horizon=99,
        past_only_covariates=past_only_covariates,
        past_future_covariates=past_future_covariates,
        past_only_mask=past_only_mask,
        past_future_mask=past_future_mask,
        mask=mask,
        return_aux_outputs=True,
    )

    assert isinstance(result, tuple)
    actual, auxiliary = result
    expected = actual.new_tensor(
        [
            10.019998550415039,
            11.902892112731934,
            13.782285690307617,
            5.5,
            6.0,
            6.5,
            5.632140159606934,
            5.58524227142334,
            6.072176456451416,
        ]
    ).reshape(1, 3, 3, 1)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert auxiliary["logits"].shape == (1, 3, 5, 4, 1)
    assert auxiliary["__call__:resblock_input"].shape == (1, 3, 5, 12)


@withCUDA
def test_internal_model_decode_without_stitching_matches_pinned_upstream(
    device: torch.device,
) -> None:
    model = _golden_model(
        device,
        use_stitching=False,
        use_linear_detrending=False,
        use_frozen_running_stats=True,
    )
    target = torch.tensor([[[1.0, 3.0, 2.0, 5.0]]], device=device)
    target_mask = torch.tensor(
        [[[False, False, True, False]]],
        device=device,
    )

    actual = model.decode(target, horizon=5, target_mask=target_mask)

    assert isinstance(actual, torch.Tensor)
    expected = actual.new_tensor(
        [
            3.2529115676879883,
            3.369866371154785,
            3.794128179550171,
            3.2906386852264404,
            3.4670321941375732,
        ]
    ).reshape(1, 1, 5, 1)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_internal_model_decode_rejects_nonpositive_horizon() -> None:
    model = _golden_model(torch.device("cpu"))

    with pytest.raises(ValueError, match="horizon > 0"):
        model.decode(torch.zeros(1, 1, 2))


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
def test_internal_model_freezes_running_statistics(
    device: torch.device,
) -> None:
    model = _internal_model(device).eval()
    values = torch.tensor(
        [[[[1.0, 3.0], [10.0, 14.0], [100.0, 200.0]]]],
        device=device,
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.ones(1, 1, 3, dtype=torch.bool, device=device)

    with torch.inference_mode():
        outputs = model(
            values,
            masks,
            patch_is_target,
            freeze_after=0,
        )

    running_mean, running_std = outputs["revin_stats"]
    torch.testing.assert_close(
        running_mean,
        torch.tensor([[[2.0, 2.0, 2.0]]], device=device),
    )
    torch.testing.assert_close(
        running_std,
        torch.tensor([[[1.0, 1.0, 1.0]]], device=device),
    )


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


def test_internal_model_configuration_round_trip() -> None:
    model = _internal_model("cpu")
    config = model.to_dict()
    restored = _TimesFM3(**config)

    assert restored.to_dict() == config
    state_keys = set(restored.state_dict())
    assert "pre_transformer_resblock.hidden_layer.weight" in state_keys
    assert (
        "transformer_stack.layers.1.seq_attn.query_proj.weight" in state_keys
    )
    assert "output_head.weight" in state_keys


def test_internal_model_rejects_incompatible_configuration() -> None:
    with pytest.raises(ValueError, match="must be a multiple"):
        _TimesFM3(
            input_patch_len=2,
            output_patch_len=3,
            residual_block_config=_residual_config(),
            transformer_config=_transformer_config(),
            use_stitching=False,
        )

    with pytest.raises(ValueError, match="dimensions must match"):
        _TimesFM3(
            input_patch_len=2,
            output_patch_len=4,
            residual_block_config=_residual_config(output_dims=4),
            transformer_config=_transformer_config(model_dims=8),
        )

    with pytest.raises(ValueError, match="Stitching requires"):
        _TimesFM3(
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
def test_internal_model_masks_target_lookahead(
    device: torch.device,
) -> None:
    model = _internal_model(device).eval()
    values = torch.arange(
        1,
        1 + 2 * 3 * 2,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 2, 3, 2)
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.zeros(1, 2, 3, dtype=torch.bool, device=device)
    patch_is_target[:, 0] = True

    with torch.inference_mode():
        residual_input = model(
            values,
            masks,
            patch_is_target,
            return_aux_outputs=True,
        )["__call__:resblock_input"]

    # [current values (2), lookahead values (4), current masks (2),
    #  lookahead masks (4)]
    assert torch.count_nonzero(residual_input[:, 0, :, 2:6]) == 0
    assert residual_input[:, 0, :, 8:12].bool().all()
    assert not residual_input[:, 1, 0, 8:12].bool().any()
    torch.testing.assert_close(
        residual_input[:, 1, 1, 8:12].bool(),
        torch.tensor([[False, False, True, True]], device=device),
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


@withCUDA
def test_internal_model_sanitizes_inputs(device: torch.device) -> None:
    model = _TimesFM3(
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
