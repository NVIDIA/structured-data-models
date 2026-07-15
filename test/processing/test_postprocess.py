from typing import Literal

import pytest
import torch
from sdm import TableTensor
from sdm.processing import (
    ClassDecode,
    EstimatorMean,
    QuantileDecode,
    RecipeContext,
    SoftmaxTemperature,
    StandardScale,
    TargetDecode,
)
from sdm.testing import withCUDA


@withCUDA
def test_softmax_temperature_controls_sharpness(
    device: torch.device,
) -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]], device=device)

    colder = (
        SoftmaxTemperature(temperature=0.5)
        .transform(TableTensor.from_tensor(logits))
        .numerical
    )
    warmer = (
        SoftmaxTemperature(temperature=2.0)
        .transform(TableTensor.from_tensor(logits))
        .numerical
    )

    assert colder[0, -1] > warmer[0, -1]
    assert colder[0, 0] < warmer[0, 0]


@withCUDA
def test_softmax_temperature_is_numerically_stable(
    device: torch.device,
) -> None:
    logits = torch.tensor(
        [[1000.0, 1001.0], [-1000.0, -1001.0]],
        device=device,
    )

    output = (
        SoftmaxTemperature()
        .transform(TableTensor.from_tensor(logits))
        .numerical
    )

    assert torch.isfinite(output).all()
    assert torch.allclose(output.sum(dim=-1), torch.ones(2, device=device))


@pytest.mark.parametrize("temperature", [0.0, float("inf"), float("nan")])
def test_softmax_temperature_rejects_invalid_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        SoftmaxTemperature(temperature=temperature)


@withCUDA
def test_class_decode_uses_all_estimator_contexts(
    device: torch.device,
) -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [
                [[10.0, 20.0, 30.0]],
                [[40.0, 50.0, 60.0]],
            ],
            device=device,
        )
    )
    contexts = (
        RecipeContext(
            estimator_index=0,
            task="classification",
            class_indices=(2, 0, 1),
        ),
        RecipeContext(
            estimator_index=1,
            task="classification",
            class_indices=(1, 2, 0),
        ),
    )

    actual = ClassDecode().transform(table, context=contexts)

    expected = torch.tensor(
        [
            [[30.0, 10.0, 20.0]],
            [[50.0, 60.0, 40.0]],
        ],
        device=device,
    )
    torch.testing.assert_close(actual.numerical, expected)

    with pytest.raises(ValueError, match="position 0"):
        ClassDecode().transform(table, context=contexts[::-1])
    with pytest.raises(RuntimeError, match="RecipeContext"):
        ClassDecode().transform(table)


@withCUDA
def test_target_decode_uses_each_fitted_target_inverse(
    device: torch.device,
) -> None:
    first = StandardScale().fit(
        TableTensor.from_tensor(torch.tensor([[1.0], [3.0]], device=device))
    )
    second = StandardScale().fit(
        TableTensor.from_tensor(torch.tensor([[10.0], [30.0]], device=device))
    )
    table = TableTensor.from_tensor(torch.zeros(2, 1, 1, device=device))
    contexts = (
        RecipeContext(
            estimator_index=0,
            task="regression",
            target_inverse=first,
        ),
        RecipeContext(
            estimator_index=1,
            task="regression",
            target_inverse=second,
        ),
    )

    decoded = TargetDecode().transform(table, context=contexts)
    actual = EstimatorMean().transform(decoded)

    torch.testing.assert_close(
        decoded.numerical,
        torch.tensor([[[2.0]], [[20.0]]], device=device),
    )
    torch.testing.assert_close(
        actual.numerical,
        torch.tensor([[11.0]], device=device),
    )


@withCUDA
def test_estimator_mean_reduces_third_to_last_dimension(
    device: torch.device,
) -> None:
    values = torch.arange(
        2 * 3 * 4 * 5,
        device=device,
        dtype=torch.float32,
    ).reshape(2, 3, 4, 5)

    actual = EstimatorMean().transform(TableTensor.from_tensor(values))

    torch.testing.assert_close(actual.numerical, values.mean(dim=-3))


@withCUDA
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("median", [[3.0]]),
        ("mean", [[2.5]]),
        ("quantiles", [[1.0, 3.0, 4.0]]),
    ],
)
def test_quantile_decode_matches_rfm_ordering(
    device: torch.device,
    method: Literal["mean", "median", "quantiles"],
    expected: list[list[float]],
) -> None:
    kwargs = {"quantiles": (0.0, 0.5, 1.0)} if method == "quantiles" else {}
    processor = QuantileDecode(method=method, **kwargs)
    table = TableTensor.from_tensor(
        torch.tensor([[4.0, 1.0, 3.0, 2.0]], device=device)
    )

    actual = processor.transform(table)

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor(expected, device=device),
    )
