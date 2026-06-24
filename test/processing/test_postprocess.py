from __future__ import annotations

import pytest
import torch
from schemafm import CategoricalTensor, StringTensor
from schemafm.processing import DecodeLabels, SoftmaxTemperature


def test_softmax_temperature_runs_without_fit() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])

    processor = SoftmaxTemperature()

    assert torch.allclose(
        processor.transform(logits), torch.softmax(logits, dim=-1)
    )


def test_softmax_temperature_controls_sharpness() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])

    colder = SoftmaxTemperature(temperature=0.5).fit(logits).transform(logits)
    warmer = SoftmaxTemperature(temperature=2.0).fit(logits).transform(logits)

    assert colder[0, -1] > warmer[0, -1]
    assert colder[0, 0] < warmer[0, 0]


def test_softmax_temperature_is_numerically_stable() -> None:
    logits = torch.tensor([[1000.0, 1001.0], [-1000.0, -1001.0]])

    output = SoftmaxTemperature().fit(logits).transform(logits)

    assert torch.isfinite(output).all()
    assert torch.allclose(output.sum(dim=-1), torch.ones(2))


def test_softmax_temperature_matches_tabicl_numpy_formula() -> None:
    logits = torch.tensor(
        [
            [1.5, -2.0, 0.25],
            [100.0, 101.0, 99.0],
        ],
        dtype=torch.float64,
    )
    temperature = 0.9
    scaled = logits / temperature
    exp = torch.exp(scaled - scaled.max(dim=-1, keepdim=True).values)
    expected = exp / exp.sum(dim=-1, keepdim=True)

    output = (
        SoftmaxTemperature(temperature=temperature)
        .fit(logits)
        .transform(logits)
    )

    assert torch.allclose(output, expected)


def test_softmax_temperature_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        SoftmaxTemperature(temperature=0.0)


def test_decode_labels_runs_without_fit_and_has_no_vocab_state() -> None:
    categories = StringTensor.from_list(["red", "green", "blue"])
    indices = torch.tensor([2, 0, 1])
    processor = DecodeLabels()

    output = processor.transform(indices, categories)

    assert output.tolist() == ["blue", "red", "green"]
    assert processor.state_dict() == {}
    assert list(processor.named_buffers()) == []


def test_decode_labels_uses_categorical_tensor_categories() -> None:
    target = CategoricalTensor(
        data=torch.tensor([[0], [1], [0]], dtype=torch.int64),
        categories=(torch.tensor([10, 20]),),
    )

    output = DecodeLabels().transform(torch.tensor([1, 0]), target)

    assert torch.equal(output, torch.tensor([20, 10]))


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([-1]),
        torch.tensor([2]),
    ],
)
def test_decode_labels_rejects_out_of_range_indices(
    indices: torch.Tensor,
) -> None:
    categories = torch.tensor([10, 20])

    with pytest.raises(ValueError, match="outside categories"):
        DecodeLabels().transform(indices, categories)


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([0.0]),
        torch.tensor([True]),
    ],
)
def test_decode_labels_rejects_non_integer_indices(
    indices: torch.Tensor,
) -> None:
    categories = torch.tensor([10, 20])

    with pytest.raises(ValueError, match="integer class indices"):
        DecodeLabels().transform(indices, categories)


def test_decode_labels_rejects_multi_column_categorical_context() -> None:
    target = CategoricalTensor(
        data=torch.tensor([[0, 1]], dtype=torch.int64),
        categories=(torch.tensor([10]), torch.tensor([20, 30])),
    )

    with pytest.raises(ValueError, match="single target"):
        DecodeLabels().transform(torch.tensor([0]), target)


def test_processing_api_does_not_export_encoding_vocab_processors() -> None:
    import schemafm.processing as processing

    assert hasattr(processing, "DecodeLabels")
    assert not hasattr(processing, "LabelEncode")
    assert not hasattr(processing, "OrdinalEncode")
