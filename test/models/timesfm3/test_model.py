# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file

import sdm
from sdm import Stype, TableTensor
from sdm.models import TimesFM3
from sdm.models.timesfm3 import model as timesfm_module
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


def _write_pretrained_fixture(
    tmp_path: Path,
) -> tuple[_TimesFM3, dict[str, Path]]:
    expected = _internal_model("cpu").eval()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(expected.to_dict()))
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(expected.state_dict(), checkpoint_path)
    return expected, {
        "config.json": config_path,
        "model.safetensors": checkpoint_path,
    }


def _reference_fixture() -> dict[str, Any]:
    path = Path(__file__).with_name("reference") / "golden.json"
    return cast(dict[str, Any], json.loads(path.read_text()))


def _fixture_tensor(
    values: list[list[float | None]],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            [float("nan") if value is None else value for value in row]
            for row in values
        ],
        device=device,
    )


def _load_reference_weights(
    model: _TimesFM3,
    model_fixture: dict[str, Any],
    recipe: dict[str, int],
) -> None:
    state = model.state_dict()
    actual_state = [
        {"key": key, "shape": list(tensor.shape)}
        for key, tensor in sorted(state.items())
    ]
    assert actual_state == model_fixture["state"]

    for index, key in enumerate(sorted(state)):
        tensor = state[key]
        values = (
            (
                torch.arange(tensor.numel(), device=tensor.device).reshape(
                    tensor.shape
                )
                + recipe["offset_step"] * index
            )
            % recipe["modulus"]
            - recipe["center"]
        ) / recipe["divisor"]
        state[key] = values.to(dtype=tensor.dtype)
    model.load_state_dict(state, strict=True)


def _golden_model(device: torch.device) -> _TimesFM3:
    fixture = _reference_fixture()
    model_fixture = fixture["internal"]
    model = _TimesFM3(**model_fixture["config"], device=device).eval()
    _load_reference_weights(model, model_fixture, fixture["weight_recipe"])
    return model


@pytest.mark.parametrize("accept_license", [False, True])
@withCUDA
def test_pretrained_loading(
    device: torch.device,
    accept_license: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, paths = _write_pretrained_fixture(tmp_path)
    calls: list[tuple[str, str, str, str | None]] = []

    def download(
        repo_id: str,
        filename: str,
        **kwargs: object,
    ) -> str:
        revision = kwargs.get("revision")
        assert isinstance(revision, str)
        license_prompt = kwargs.get("license_prompt")
        assert license_prompt is None or isinstance(license_prompt, str)
        calls.append((repo_id, filename, revision, license_prompt))
        return str(paths[filename])

    monkeypatch.setattr(timesfm_module, "download_checkpoint", download)
    model = TimesFM3(
        pretrained=True,
        accept_license=accept_license,
        device=device,
    )
    expected = expected.to(device)
    values = torch.arange(12, device=device, dtype=torch.float32).reshape(
        1, 1, 6, 2
    )
    masks = torch.zeros_like(values, dtype=torch.bool)
    patch_is_target = torch.ones(1, 1, 6, dtype=torch.bool, device=device)

    with torch.inference_mode():
        actual_logits = model.model(
            values,
            masks,
            patch_is_target,
        )["logits"]
        expected_logits = expected(
            values,
            masks,
            patch_is_target,
        )["logits"]

    assert model.model.to_dict() == expected.to_dict()
    assert all(parameter.device == device for parameter in model.parameters())
    assert all(buffer.device == device for buffer in model.buffers())
    torch.testing.assert_close(actual_logits, expected_logits)
    assert [call[:3] for call in calls] == [
        (
            "google/timesfm-3.0-pytorch",
            "config.json",
            "43046b85ec22d584a13f8098c2ed39c889e129c2",
        ),
        (
            "google/timesfm-3.0-pytorch",
            "model.safetensors",
            "43046b85ec22d584a13f8098c2ed39c889e129c2",
        ),
    ]
    assert calls[0][3] is None
    if accept_license:
        assert calls[1][3] is None
    else:
        assert calls[1][3] is not None


def test_pretrained_loading_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, paths = _write_pretrained_fixture(tmp_path)
    checkpoint = expected.state_dict()
    checkpoint.pop("output_head.bias")
    save_file(checkpoint, paths["model.safetensors"])

    def download(
        repo_id: str,
        filename: str,
        **kwargs: object,
    ) -> str:
        return str(paths[filename])

    monkeypatch.setattr(timesfm_module, "download_checkpoint", download)
    with pytest.raises(RuntimeError, match="Missing key"):
        TimesFM3(pretrained=True, accept_license=True)


def _public_model(device: torch.device) -> TimesFM3:
    model = TimesFM3(pretrained=False, device="meta")
    model.model = _internal_model(device)
    return model.eval()


@withCUDA
def test_forward_matches_internal_decode(device: torch.device) -> None:
    model = _public_model(device)
    context_values = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 11.0],
            [float("nan"), 12.0],
            [4.0, 13.0],
            [5.0, 14.0],
        ],
        device=device,
    )
    query_values = torch.tensor(
        [[15.0], [float("nan")], [17.0]],
        device=device,
    )
    target_values = torch.tensor(
        [
            [2.0, 20.0],
            [3.0, 21.0],
            [4.0, float("nan")],
            [5.0, 23.0],
            [6.0, 24.0],
        ],
        device=device,
    )
    x_context = TableTensor.from_tensor(
        context_values,
        columns=["past", "known"],
    )
    x_query = TableTensor.from_tensor(query_values, columns=["known"])
    y_context = TableTensor.from_tensor(
        target_values,
        columns=["first", "second"],
    )

    out = model(x_context, y_context, x_query)

    target = target_values.T.unsqueeze(0)
    past_only = context_values[:, :1].T.unsqueeze(0)
    past_future = torch.cat(
        [context_values[:, 1], query_values[:, 0]]
    ).reshape(1, 1, -1)
    expected = model.model.decode(
        target,
        horizon=len(query_values),
        past_only_covariates=past_only,
        past_future_covariates=past_future,
        target_mask=~target.isfinite(),
        past_only_mask=~past_only.isfinite(),
        past_future_mask=~past_future.isfinite(),
    )
    assert isinstance(expected, torch.Tensor)
    expected = expected[:, :2].permute(0, 2, 1, 3).reshape(3, 6)

    assert out.columns[Stype.numerical] == (
        "first__q10",
        "first__q50",
        "first__q90",
        "second__q10",
        "second__q50",
        "second__q90",
    )
    torch.testing.assert_close(out.numerical, expected)

    model.fit(x_context, y_context)
    cached = model.predict(x_query)
    torch.testing.assert_close(cached.numerical, out.numerical)
    assert cached.columns == out.columns

    past_only_query = TableTensor.from_tensor(
        torch.empty(3, 0, device=device),
        columns=[],
    )
    past_only_out = model(x_context, y_context, past_only_query)
    cached_past_only = model.predict(past_only_query)
    torch.testing.assert_close(
        cached_past_only.numerical,
        past_only_out.numerical,
    )
    cached_again = model.predict(x_query)
    torch.testing.assert_close(cached_again.numerical, out.numerical)


@withCUDA
def test_forward_without_covariates(device: torch.device) -> None:
    model = _public_model(device)
    context = TableTensor.from_tensor(
        torch.empty(5, 0, device=device),
        columns=[],
    )
    query = TableTensor.from_tensor(
        torch.empty(3, 0, device=device),
        columns=[],
    )
    target_values = torch.arange(
        5, dtype=torch.float32, device=device
    ).unsqueeze(-1)
    target = TableTensor.from_tensor(target_values, columns=["target"])

    out = model(context, target, query)
    expected = model.model.decode(
        target_values.T.unsqueeze(0),
        horizon=3,
    )
    assert isinstance(expected, torch.Tensor)
    expected = expected[0, 0]

    assert out.size() == (3, 3)
    torch.testing.assert_close(out.numerical, expected)


@withCUDA
def test_forward_preserves_batch_dimensions(device: torch.device) -> None:
    model = _public_model(device)
    context_values = torch.arange(
        24, dtype=torch.float32, device=device
    ).reshape(2, 6, 2)
    query_values = torch.arange(6, dtype=torch.float32, device=device).reshape(
        2, 3, 1
    )
    target_values = torch.arange(
        24, dtype=torch.float32, device=device
    ).reshape(2, 6, 2)
    x_context = TableTensor.from_tensor(
        context_values,
        columns=["past", "known"],
    )
    x_query = TableTensor.from_tensor(query_values, columns=["known"])
    y_context = TableTensor.from_tensor(
        target_values,
        columns=["first", "second"],
    )

    out = model(
        x_context,
        y_context,
        x_query,
        num_estimators=1,
    )

    target = target_values.movedim(-1, -2)
    past_only = context_values[..., :1].movedim(-1, -2)
    past_future = torch.cat(
        [context_values[..., 1:], query_values],
        dim=-2,
    ).movedim(-1, -2)
    expected = model.model.decode(
        target,
        horizon=3,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
    )
    assert isinstance(expected, torch.Tensor)
    expected = expected[:, :2].permute(0, 2, 1, 3).reshape(2, 3, 6)

    assert out.size() == (2, 3, 6)
    torch.testing.assert_close(out.numerical, expected)


@withCUDA
def test_default_recipe_averages_estimators(device: torch.device) -> None:
    model = _public_model(device)
    context_values = torch.arange(
        24, dtype=torch.float32, device=device
    ).reshape(2, 6, 2)
    query_values = torch.arange(6, dtype=torch.float32, device=device).reshape(
        2, 3, 1
    )
    target_values = torch.arange(
        6, dtype=torch.float32, device=device
    ).reshape(1, 6, 1)
    target_values = torch.cat([target_values, target_values + 2.0])
    contexts = TableTensor.from_tensor(
        context_values,
        columns=["past", "known"],
    )
    queries = TableTensor.from_tensor(query_values, columns=["known"])
    targets = TableTensor.from_tensor(target_values, columns=["target"])

    out = model(contexts, targets, queries)
    member_outputs = [
        model(contexts[index], targets[index], queries[index]).numerical
        for index in range(2)
    ]
    expected = torch.stack(member_outputs).mean(dim=0)

    assert out.size() == (3, 3)
    torch.testing.assert_close(out.numerical, expected)

    model.fit(contexts, targets)
    cached = model.predict(queries)
    torch.testing.assert_close(cached.numerical, out.numerical)


def test_forecasting_example(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def model_factory(*, device: torch.device) -> TimesFM3:
        return _public_model(device)

    monkeypatch.setattr(sdm.models, "TimesFM3", model_factory)
    path = Path(__file__).parents[3] / "examples/timesfm3/forecast.py"

    namespace = runpy.run_path(str(path))

    output = capsys.readouterr().out
    forecast = namespace["forecast"]
    assert isinstance(forecast, TableTensor)
    assert forecast.size(0) == 3
    assert "ice_cream__q50" in forecast.columns[Stype.numerical]
    assert "cold_drinks__q50" in forecast.columns[Stype.numerical]
    assert "ice_cream__q50" in output
    assert forecast.numerical.isfinite().all()


@withCUDA
def test_public_model_matches_pinned_upstream(
    device: torch.device,
) -> None:
    fixture = _reference_fixture()
    model_fixture = fixture["public"]
    model = TimesFM3(pretrained=False, device="meta")
    model.model = _TimesFM3(**model_fixture["config"], device=device).eval()
    _load_reference_weights(
        model.model, model_fixture, fixture["weight_recipe"]
    )

    inputs = model_fixture["inputs"]
    target_values = _fixture_tensor(inputs["targets"]["context"], device)
    past_values = _fixture_tensor(inputs["past_only"]["context"], device)
    known_context_values = _fixture_tensor(
        inputs["future_known"]["context"], device
    )
    query_values = _fixture_tensor(inputs["future_known"]["query"], device)
    x_context = TableTensor.from_tensor(
        torch.cat([past_values, known_context_values], dim=-1),
        columns=(
            inputs["past_only"]["columns"] + inputs["future_known"]["columns"]
        ),
    )
    x_query = TableTensor.from_tensor(
        query_values,
        columns=inputs["future_known"]["columns"],
    )
    y_context = TableTensor.from_tensor(
        target_values,
        columns=inputs["targets"]["columns"],
    )

    actual = model(x_context, y_context, x_query)

    expected = actual.numerical.new_tensor(model_fixture["expected"]["values"])
    assert actual.columns[Stype.numerical] == tuple(
        model_fixture["expected"]["columns"]
    )
    torch.testing.assert_close(actual.numerical, expected)


@pytest.mark.parametrize(
    ("context_length", "horizon"),
    [(1, 1), (2, 4), (3, 5)],
)
@withCUDA
def test_public_forecast_geometry(
    device: torch.device,
    context_length: int,
    horizon: int,
) -> None:
    model = _public_model(device)
    x_context = TableTensor.from_tensor(
        torch.empty(context_length, 0, device=device),
        columns=[],
    )
    x_query = TableTensor.from_tensor(
        torch.empty(horizon, 0, device=device),
        columns=[],
    )
    y_context = TableTensor.from_tensor(
        torch.arange(
            1,
            2 * context_length + 1,
            dtype=torch.float32,
            device=device,
        ).reshape(context_length, 2),
        columns=["first", "second"],
    )

    out = model(x_context, y_context, x_query)

    assert out.size() == (horizon, 6)
    assert out.numerical.isfinite().all()


@withCUDA
def test_fit_snapshots_and_replaces_context(device: torch.device) -> None:
    model = _public_model(device)
    context_values = torch.arange(
        12, dtype=torch.float32, device=device
    ).reshape(6, 2)
    target_values = torch.arange(
        12, dtype=torch.float32, device=device
    ).reshape(6, 2)
    query_values = torch.arange(
        3, dtype=torch.float32, device=device
    ).unsqueeze(-1)
    x_context = TableTensor.from_tensor(
        context_values,
        columns=["past", "known"],
    )
    y_context = TableTensor.from_tensor(
        target_values,
        columns=["first", "second"],
    )
    x_query = TableTensor.from_tensor(query_values, columns=["known"])

    model.fit(x_context, y_context)
    expected = model.predict(x_query)
    context_values.add_(1000.0)
    target_values.neg_()

    actual = model.predict(x_query)
    repeated = model.predict(x_query)
    torch.testing.assert_close(actual.numerical, expected.numerical)
    torch.testing.assert_close(repeated.numerical, expected.numerical)

    next_context = TableTensor.from_tensor(
        torch.arange(12, dtype=torch.float32, device=device).reshape(6, 2),
        columns=["past", "known"],
    )
    next_target = TableTensor.from_tensor(
        torch.arange(12, dtype=torch.float32, device=device).reshape(6, 2)
        + 20.0,
        columns=["first", "second"],
    )
    model.fit(next_context, next_target)
    refitted = model.predict(x_query)
    one_shot = model(next_context, next_target, x_query)
    torch.testing.assert_close(refitted.numerical, one_shot.numerical)


@withCUDA
def test_future_covariate_order_is_semantic(device: torch.device) -> None:
    model = _public_model(device)
    x_context = TableTensor.from_tensor(
        torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3),
        columns=["past", "known_a", "known_b"],
    )
    y_context = TableTensor.from_tensor(
        torch.arange(6, dtype=torch.float32, device=device).unsqueeze(-1),
        columns=["target"],
    )
    query_values = torch.arange(6, dtype=torch.float32, device=device).reshape(
        3, 2
    )
    query_ab = TableTensor.from_tensor(
        query_values,
        columns=["known_a", "known_b"],
    )
    query_ba = TableTensor.from_tensor(
        query_values.flip(-1),
        columns=["known_b", "known_a"],
    )

    out_ab = model(x_context, y_context, query_ab)
    out_ba = model(x_context, y_context, query_ba)

    torch.testing.assert_close(out_ab.numerical, out_ba.numerical)
    unknown = TableTensor.from_tensor(
        torch.zeros(3, 1, device=device),
        columns=["unknown"],
    )
    with pytest.raises(ValueError, match="subset"):
        model(x_context, y_context, unknown)


@withCUDA
def test_public_model_masks_missing_series(device: torch.device) -> None:
    model = _public_model(device)
    x_context = TableTensor.from_tensor(
        torch.tensor(
            [
                [float("nan"), 1.0],
                [float("nan"), 2.0],
                [float("nan"), float("inf")],
                [float("nan"), 4.0],
                [float("nan"), 5.0],
            ],
            device=device,
        ),
        columns=["missing_past", "known"],
    )
    x_query = TableTensor.from_tensor(
        torch.tensor([[6.0], [float("nan")], [8.0]], device=device),
        columns=["known"],
    )
    y_context = TableTensor.from_tensor(
        torch.tensor(
            [
                [float("nan"), float("nan")],
                [float("nan"), float("nan")],
                [3.0, float("nan")],
                [4.0, float("nan")],
                [5.0, float("nan")],
            ],
            device=device,
        ),
        columns=["partial", "missing"],
    )

    out = model(x_context, y_context, x_query)

    assert out.size() == (3, 6)
    assert out.numerical.isfinite().all()


@withCUDA
def test_public_model_bfloat16(device: torch.device) -> None:
    model = TimesFM3(pretrained=False, device="meta")
    model.model = _internal_model(device, dtype=torch.bfloat16).eval()
    x_context = TableTensor.from_tensor(
        torch.arange(12, device=device, dtype=torch.bfloat16).reshape(6, 2),
        columns=["past", "known"],
    )
    x_query = TableTensor.from_tensor(
        torch.arange(3, device=device, dtype=torch.bfloat16).unsqueeze(-1),
        columns=["known"],
    )
    y_context = TableTensor.from_tensor(
        torch.arange(6, device=device, dtype=torch.bfloat16).unsqueeze(-1),
        columns=["target"],
    )

    one_shot = model(x_context, y_context, x_query)
    model.fit(x_context, y_context)
    cached = model.predict(x_query)

    assert one_shot.dtype == torch.bfloat16
    assert one_shot.device == device
    assert one_shot.numerical.isfinite().all()
    torch.testing.assert_close(cached.numerical, one_shot.numerical)


@pytest.mark.parametrize("case_index", [0, 1])
@withCUDA
def test_internal_model_matches_pinned_upstream(
    device: torch.device,
    case_index: int,
) -> None:
    fixture = _reference_fixture()["internal"]["forward"]
    inputs = fixture["inputs"]
    model = _golden_model(device)
    values = torch.tensor(inputs["values"], device=device)
    masks = torch.tensor(inputs["masks"], device=device)
    patch_is_target = torch.tensor(inputs["patch_is_target"], device=device)
    patch_cpm_mask = inputs["patch_cpm_masks"][case_index]
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

    expected = actual.new_tensor(fixture["outputs"][case_index])
    torch.testing.assert_close(actual, expected)


@withCUDA
def test_internal_model_decode_matches_pinned_upstream(
    device: torch.device,
) -> None:
    fixture = _reference_fixture()["internal"]["decode"]
    inputs = fixture["inputs"]
    model = _golden_model(device)

    result = model.decode(
        torch.tensor(inputs["target"], device=device),
        horizon=inputs["horizon"],
        past_only_covariates=torch.tensor(
            inputs["past_only_covariates"], device=device
        ),
        past_future_covariates=torch.tensor(
            inputs["past_future_covariates"], device=device
        ),
        past_only_mask=torch.tensor(inputs["past_only_mask"], device=device),
        past_future_mask=torch.tensor(
            inputs["past_future_mask"], device=device
        ),
        mask=torch.tensor(inputs["mask"], device=device),
        return_aux_outputs=True,
    )

    assert isinstance(result, tuple)
    actual, auxiliary = result
    expected = actual.new_tensor(fixture["output"])
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert auxiliary["logits"].shape == (1, 3, 5, 4, 1)
    assert auxiliary["__call__:resblock_input"].shape == (1, 3, 5, 12)


@withCUDA
def test_internal_model_decode_without_stitching_matches_pinned_upstream(
    device: torch.device,
) -> None:
    reference = _reference_fixture()
    fixture = reference["internal"]["decode_without_stitching"]
    inputs = fixture["inputs"]
    model = _TimesFM3(**fixture["config"], device=device).eval()
    _load_reference_weights(
        model, reference["internal"], reference["weight_recipe"]
    )
    actual = model.decode(
        torch.tensor(inputs["target"], device=device),
        horizon=inputs["horizon"],
        target_mask=torch.tensor(inputs["target_mask"], device=device),
    )

    assert isinstance(actual, torch.Tensor)
    expected = actual.new_tensor(fixture["output"])
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
