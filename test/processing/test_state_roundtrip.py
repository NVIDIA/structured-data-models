import copy
import io
from collections.abc import Callable

import pytest
import torch

from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    PCA,
    TFIDF,
    AlignCategories,
    ClipQuantiles,
    ClipSigma,
    DropConstantColumns,
    ImputeMean,
    ImputeMode,
    PowerTransform,
    Processor,
    QuantileTransform,
    RandomProjection,
    ShuffleCategories,
    ShuffleColumns,
    Standardize,
)


def _save_and_load(state: dict[str, object]) -> dict[str, object]:
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


def _numerical_table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [1.0, 1.0, 2.0],
                [2.0, 1.0, 4.0],
                [5.0, 1.0, 8.0],
            ]
        )
    )


def _categorical_table() -> TableTensor:
    return TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[0, 1], [1, -1], [0, 0]], dtype=torch.int32),
            categories=(
                StringTensor.from_list(["a", "b"]),
                StringTensor.from_list(["x", "y"]),
            ),
        )
    )


def _text_table() -> TableTensor:
    return TableTensor.from_tensor(
        StringTensor.from_list([["alpha"], ["beta"], ["alpha beta"]])
    )


@pytest.mark.parametrize(
    "factory",
    [
        Standardize,
        ClipQuantiles,
        ClipSigma,
        ImputeMean,
        PowerTransform,
        lambda: QuantileTransform(n_quantiles=4, subsample=None),
        lambda: PCA(2),
        DropConstantColumns,
        ShuffleColumns,
        lambda: RandomProjection(2),
    ],
)
def test_numerical_processor_state_dict_round_trip(
    factory: Callable[[], Processor],
) -> None:
    table = _numerical_table()
    processor = factory().fit(table)
    expected = processor.transform(table)

    restored = factory()
    restored.load_state_dict(
        _save_and_load(copy.deepcopy(processor.state_dict()))
    )

    assert restored.transform(table).equal(expected)


@pytest.mark.parametrize(
    ("factory", "table"),
    [
        (ImputeMode, _categorical_table),
        (AlignCategories, _categorical_table),
        (ShuffleCategories, _categorical_table),
        (lambda: TFIDF(ngram_range=(2, 2)), _text_table),
    ],
)
def test_non_numerical_processor_state_dict_round_trip(
    factory: Callable[[], Processor],
    table: Callable[[], TableTensor],
) -> None:
    input_table = table()
    processor = factory().fit(input_table)
    expected = processor.transform(input_table)

    restored = factory()
    restored.load_state_dict(
        _save_and_load(copy.deepcopy(processor.state_dict()))
    )

    assert restored.transform(input_table).equal(expected)
