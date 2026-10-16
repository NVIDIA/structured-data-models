# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.models import KumoAnomaly
from sdm.models.kumo.timeseries.anomaly.ckpt import model_kwargs, remap_ckpt
from sdm.models.kumo.timeseries.anomaly.model import _KumoAnomaly
from sdm.models.kumo.timeseries.anomaly.sampling import DPMSolver
from sdm.testing import withCUDA


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """Save a tiny upstream-format checkpoint without network access."""
    config = {
        "model": {
            "target_dim": 3,
            "timeemb": 8,
            "featureemb": 4,
            "is_unconditional": 0,
        },
        "diffusion": {
            "channels": 8,
            "layers": 1,
            "nheads": 2,
            "diffusion_embedding_dim": 8,
            "num_steps": 50,
            "beta_start": 0.0001,
            "beta_end": 0.01,
            "schedule": "linear",
        },
        "dataset": {"scale_factor": 1.0, "split": 10},
    }
    model = _KumoAnomaly(**model_kwargs(config))
    state = {}
    for key, value in model.state_dict().items():
        key = key.replace("denoiser.", "diffmodel.")
        key = key.replace("time_layer.", "time_layer.layers.0.")
        key = key.replace("feature_layer.", "feature_layer.layers.0.")
        key = key.removeprefix("side_embedding.")
        state[key] = value
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": state, "config": config}, path)
    return path


@withCUDA
@pytest.mark.parametrize("length", [1, 8, 11, 16])
@pytest.mark.parametrize("features", [1, 3, 5])
def test_anomaly_output(
    checkpoint: Path, device: torch.device, length: int, features: int
) -> None:
    """Score real rows/features, including partial windows and PCA input."""
    model = KumoAnomaly(pretrained=checkpoint, window_length=8, device=device)
    x = torch.randn(length, features, device=device)
    reference = torch.randn(12, features, device=device)
    out = model(
        x, reference=reference, num_samples=2, num_steps=3, batch_size=1
    )
    count = min(features, 3)
    assert out.scores.shape == out.l2_scores.shape == (length,)
    assert out.reconstruction.shape == out.target.shape == (length, count)
    assert out.scores.device == device
    assert out.scores.isfinite().all()
    assert not out.scores.requires_grad
    error = out.reconstruction.numerical - out.target.numerical
    torch.testing.assert_close(out.scores, error.abs().mean(-1))
    torch.testing.assert_close(out.l2_scores, error.norm(dim=-1))


@withCUDA
def test_generator_and_state_dict(
    checkpoint: Path, device: torch.device
) -> None:
    """A supplied generator reproduces output without consuming global RNG."""
    model = KumoAnomaly(pretrained=checkpoint, window_length=8, device=device)
    other = KumoAnomaly(pretrained=checkpoint, window_length=8, device=device)
    other.load_state_dict(model.state_dict())
    x = torch.arange(16, device=device, dtype=torch.float32).reshape(8, 2)
    generator = torch.Generator(device=device).manual_seed(42)
    state = generator.get_state()
    cpu_state = torch.random.get_rng_state()
    device_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    first = model(x, num_samples=2, num_steps=3, generator=generator)
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_state)
    if device_state is not None:
        torch.testing.assert_close(
            torch.cuda.get_rng_state(device), device_state
        )
    generator.set_state(state)
    second = other(x, num_samples=2, num_steps=3, generator=generator)
    torch.testing.assert_close(first.scores, second.scores, rtol=0, atol=0)
    third = model(x, num_samples=2, num_steps=3, generator=generator)
    assert not torch.equal(first.scores, third.scores)


def test_checkpoint_strict_loading(checkpoint: Path, tmp_path: Path) -> None:
    """Reject missing or unknown weights instead of partially initializing."""
    state = torch.load(checkpoint, weights_only=True)
    state["model"]["unknown.weight"] = torch.ones(1)
    path = tmp_path / "unexpected.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        KumoAnomaly(pretrained=path)
    del state["model"]["unknown.weight"]
    del state["model"]["embed_layer.weight"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        KumoAnomaly(pretrained=path)


def test_published_loader(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read architecture from checkpoint metadata and pin the download."""
    calls = []

    def download(**kwargs: object) -> Path:
        """Record the checkpoint request without accessing the network."""
        calls.append(kwargs)
        return checkpoint

    monkeypatch.setattr(
        "sdm.models.kumo.timeseries.anomaly.model.download_checkpoint",
        download,
    )
    model = KumoAnomaly()
    assert calls == [
        {
            "repo_id": "nvidia/Kumo-Anomaly",
            "filename": "final_model.pth",
            "revision": "226c5003b0582adc1cac433123f47324bfae39f2",
        }
    ]
    assert model.model.num_features == 3
    assert model.mask_blocks == 4  # Inference default, not training split=10.
    saved = torch.load(checkpoint, weights_only=True)
    expected = remap_ckpt(saved["model"])
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    saved["config"]["model"]["is_unconditional"] = 1
    with pytest.raises(ValueError, match="conditional checkpoint"):
        model_kwargs(saved["config"])


def test_reference_schema_and_values(checkpoint: Path) -> None:
    """Reference preprocessing preserves names and out-of-range values."""
    model = KumoAnomaly(pretrained=checkpoint, window_length=8)
    columns: dict[str, list[str]] = {Stype.numerical: ["a", "b"]}
    reference = TableTensor(
        columns=columns, numerical=torch.tensor([[0.0, 2.0], [2.0, 4.0]])
    )
    x = TableTensor(columns=columns, numerical=torch.tensor([[4.0, 8.0]]))
    result = model(x, reference=reference, num_samples=1, num_steps=1)
    torch.testing.assert_close(
        result.target.numerical, torch.tensor([[2.0, 3.0]])
    )
    assert result.target.columns == x.columns
    reordered = TableTensor(
        columns={Stype.numerical: ["b", "a"]}, numerical=reference.numerical
    )
    with pytest.raises(ValueError, match="ordered columns"):
        model(x, reference=reordered, num_samples=1, num_steps=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_input(checkpoint: Path, value: float) -> None:
    """Reject missing observations instead of returning misleading scores."""
    model = KumoAnomaly(pretrained=checkpoint)
    with pytest.raises(ValueError, match="non-finite"):
        model(torch.tensor([[value]]), num_samples=1, num_steps=1)


@pytest.mark.parametrize(
    ("length", "blocks"), [(7, 4), (12, 3), (0, 4), (8, 0)]
)
def test_mask_partition(length: int, blocks: int) -> None:
    """Reject masks that would leave rows unconditioned by both strategies."""
    with pytest.raises(ValueError, match="mask_blocks"):
        KumoAnomaly(pretrained=False, window_length=length, mask_blocks=blocks)


def test_complementary_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only each strategy's held-out half contributes to reconstruction."""
    model = _KumoAnomaly(
        num_features=2,
        time_channels=8,
        feature_channels=4,
        channels=8,
        num_layers=1,
        num_heads=2,
        embedding_channels=8,
    ).eval()

    class StrategyPrediction(torch.nn.Module):
        """Return the strategy index as a directly observable prediction."""

        def forward(
            self, x: Tensor, side_info: Tensor, step: Tensor, strategy: Tensor
        ) -> Tensor:
            """Fill the prediction with its masking strategy index."""
            return torch.ones_like(x[:, 0]) * strategy

    class DirectSampler(DPMSolver):
        """Evaluate once without integrating to expose mask composition."""

        calls: int = 0

        def forward(
            self, x: Tensor, predict_noise: Callable[[Tensor, Tensor], Tensor]
        ) -> Tensor:
            """Return the provided denoiser's output."""
            offset = 2 * (self.calls % 2)
            self.calls += 1
            return predict_noise(x, torch.zeros(1, dtype=torch.long)) + offset

    monkeypatch.setattr(model, "denoiser", StrategyPrediction())
    x = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    sampler = DirectSampler(torch.linspace(0.001, 0.01, 10), num_steps=1)
    actual = model(x, sampler=sampler, num_samples=2)
    expected = torch.ones_like(x) * x.mean()
    # Two samples per strategy differ by 2: use the lower median, not mean.
    expected[:, [0, 1, 4, 5]] += x.std() + 1e-5
    torch.testing.assert_close(actual, expected)
