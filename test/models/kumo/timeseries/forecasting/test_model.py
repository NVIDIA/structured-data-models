# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5EncoderModel

import sdm.processing as sp
from sdm import TableTensor
from sdm.models import KumoForecasting
from sdm.models.kumo.timeseries.forecasting import model as forecasting


@pytest.fixture
def small_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        forecasting,
        "MODEL_KWARGS",
        {
            "context_length": 32,
            "prediction_length": 6,
            "patch_len": 4,
            "channels": 32,
            "hidden_channels": 48,
            "num_layers": 2,
            "num_heads": 4,
            "head_channels": 8,
            "cross_channel_heads": 4,
        },
    )


@pytest.fixture
def model(small_config: None) -> KumoForecasting:
    return KumoForecasting(pretrained=False)


def test_forward_and_fit_predict(model: KumoForecasting) -> None:
    x = TableTensor.from_tensor(torch.randn(32, 2), columns=["a", "b"])
    y = TableTensor.from_tensor(torch.randn(32, 2), columns=["u", "v"])
    query = TableTensor.from_tensor(
        torch.full((6, 2), float("nan")), columns=["a", "b"]
    )
    direct = model(x, y, query)
    model.fit(x, y)
    cached = model.predict(query)
    torch.testing.assert_close(cached.numerical, direct.numerical)
    assert direct.shape == (6, 2)
    assert direct.columns["numerical"] == ("u", "v")
    assert direct.numerical.isfinite().all()
    assert torch.is_inference(direct)
    torch.testing.assert_close(
        model.predict(query[:3]).numerical, direct.numerical[:3]
    )
    # Query values cannot affect a model that only consumes past covariates.
    changed = query.replace_blocks(numerical=torch.randn(6, 2))
    torch.testing.assert_close(
        model(x, y, changed).numerical, direct.numerical
    )


def test_univariate_without_covariates(model: KumoForecasting) -> None:
    y = torch.randn(32, 1)
    direct = model(torch.empty(32, 0), y, torch.empty(3, 0))
    model.fit(torch.empty(32, 0), y)
    cached = model.predict(torch.empty(3, 0))
    assert direct.shape == (3, 1)
    torch.testing.assert_close(cached.numerical, direct.numerical)


def test_batch_dimensions(model: KumoForecasting) -> None:
    x, y, query = (
        torch.randn(2, 32, 1),
        torch.randn(2, 32, 2),
        torch.empty(2, 4, 1),
    )
    actual = model(x, y, query, num_estimators=1).numerical
    expected = torch.stack(
        [model(x[i], y[i], query[i]).numerical for i in range(2)]
    )
    assert actual.shape == (2, 4, 2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    model.fit(x, y, num_estimators=1)
    torch.testing.assert_close(model.predict(query).numerical, actual)


def test_uses_last_context_window(model: KumoForecasting) -> None:
    x, y, query = torch.randn(40, 1), torch.randn(40, 1), torch.empty(3, 1)
    torch.testing.assert_close(
        model(x, y, query).numerical,
        model(x[-32:], y[-32:], query).numerical,
    )


def test_history_affects_forecast(model: KumoForecasting) -> None:
    x, y, query = torch.randn(32, 2), torch.randn(32, 1), torch.empty(3, 2)
    prediction = model(x, y, query).numerical
    assert not torch.allclose(model(x.flip(0), y, query).numerical, prediction)
    assert not torch.allclose(model(x, y.flip(0), query).numerical, prediction)


def test_missing_observations(model: KumoForecasting) -> None:
    x, y = torch.randn(32, 2), torch.randn(32, 1)
    x[0, 0], y[4, 0] = float("nan"), float("nan")
    assert model(x, y, torch.empty(3, 2)).numerical.isfinite().all()


def test_fully_missing_series(model: KumoForecasting) -> None:
    """Keep forecasts finite with empty target and covariate histories."""
    x, y = torch.randn(2, 32, 2), torch.randn(2, 32, 2)
    x[0, :, 1] = float("nan")
    y[0, :, 0] = float("nan")
    y[1] = float("nan")
    query = torch.full((2, 3, 2), float("nan"))

    direct = model(x, y, query, num_estimators=1).numerical
    model.fit(x, y, num_estimators=1)
    cached = model.predict(query).numerical

    assert direct.shape == (2, 3, 2)
    assert direct.isfinite().all()
    torch.testing.assert_close(cached, direct)
    for i in range(2):
        torch.testing.assert_close(
            direct[i],
            model(x[i], y[i], query[i]).numerical,
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("horizon", [0, 7])
def test_rejects_unsupported_horizon(
    model: KumoForecasting, horizon: int
) -> None:
    with pytest.raises(ValueError, match="Forecast length"):
        model(torch.empty(32, 0), torch.randn(32, 1), torch.empty(horizon, 0))
    model.fit(torch.empty(32, 0), torch.randn(32, 1))
    with pytest.raises(ValueError, match="Forecast length"):
        model.predict(torch.empty(horizon, 0))


def test_rejects_short_history(model: KumoForecasting) -> None:
    with pytest.raises(ValueError, match="at least 32 history rows"):
        model(torch.empty(31, 0), torch.randn(31, 1), torch.empty(3, 0))


def test_rejects_mismatched_batches(model: KumoForecasting) -> None:
    with pytest.raises(ValueError, match="batch dimensions"):
        model(
            torch.empty(2, 32, 0),
            torch.randn(2, 32, 1),
            torch.empty(3, 4, 0),
            num_estimators=1,
        )


def test_custom_recipe(model: KumoForecasting) -> None:
    x, y, query = torch.randn(32, 1), torch.randn(32, 1), torch.empty(3, 1)
    recipe = sp.Recipe(
        target=sp.Standardize(), output=sp.ReduceEstimators(method="mean")
    )
    direct = model(x, y, query, recipe=recipe)
    model.fit(x, y, recipe=recipe)
    torch.testing.assert_close(
        model.predict(query).numerical, direct.numerical
    )


@pytest.mark.parametrize("use_cross_channel", [False, True])
def test_checkpoint_loading(
    small_config: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_cross_channel: bool,
) -> None:
    source = KumoForecasting(
        pretrained=False, use_cross_channel=use_cross_channel
    )
    encoder = T5EncoderModel(
        T5Config(
            vocab_size=16,
            d_model=32,
            d_ff=48,
            num_layers=2,
            num_heads=4,
            d_kv=8,
            feed_forward_proj="gated-gelu",
        )
    ).encoder.eval()
    ckpt = {
        key: value
        for key, value in source.model.state_dict().items()
        if not key.startswith("encoder.")
    }
    ckpt.update(
        {
            f"encoder.{key}": value
            for key, value in encoder.state_dict().items()
        }
    )
    path = tmp_path / "forecast.pt"
    torch.save(ckpt, path)
    restored = KumoForecasting(
        checkpoint=path, use_cross_channel=use_cross_channel
    )
    assert not any(
        t.is_meta for t in (*restored.parameters(), *restored.buffers())
    )
    assert not restored.training
    embeddings = torch.randn(2, 8, 32)
    with torch.inference_mode():
        torch.testing.assert_close(
            restored.model.encoder(embeddings),
            encoder(inputs_embeds=embeddings).last_hidden_state,
            atol=2e-5,
            rtol=2e-5,
        )

    def download_checkpoint(**kwargs: object) -> str:
        assert kwargs["repo_id"] == "nvidia/Kumo-Forecast"
        assert kwargs["revision"] == "abff20a58834638b28227ff4ab934f26206e4b09"
        assert kwargs["filename"] == (
            "run8_best_model_cr.pt"
            if use_cross_channel
            else "moment_head_512_6hr.pt"
        )
        return str(path)

    monkeypatch.setattr(
        forecasting, "download_checkpoint", download_checkpoint
    )
    downloaded = KumoForecasting(use_cross_channel=use_cross_channel)
    x, y, query = torch.randn(32, 1), torch.randn(32, 1), torch.empty(6, 1)
    torch.testing.assert_close(
        downloaded(x, y, query).numerical, restored(x, y, query).numerical
    )
    # Distributed training prefixes are accepted; every other key is strict.
    torch.save({f"module.{key}": value for key, value in ckpt.items()}, path)
    KumoForecasting(checkpoint=path, use_cross_channel=use_cross_channel)
    ckpt["unexpected.weight"] = torch.ones(1)
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        KumoForecasting(checkpoint=path, use_cross_channel=use_cross_channel)
    del ckpt["unexpected.weight"]
    del ckpt["encoder.block.0.layer.0.SelfAttention.q.weight"]
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        KumoForecasting(checkpoint=path, use_cross_channel=use_cross_channel)


def test_meta_construction() -> None:
    model = KumoForecasting(pretrained=False, device="meta")
    assert all(t.is_meta for t in (*model.parameters(), *model.buffers()))
