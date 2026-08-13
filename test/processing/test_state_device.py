from collections.abc import Callable
from typing import cast

import pytest
import torch

from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    PCA,
    TFIDF,
    AlignCategories,
    Choice,
    ClipQuantiles,
    ClipSigma,
    DropConstantColumns,
    ImputeMean,
    ImputeMode,
    PowerTransform,
    Processor,
    QuantileTransform,
    RandomProjection,
    Sequential,
    ShuffleCategories,
    ShuffleColumns,
    Standardize,
    StypeDispatch,
)
from sdm.tensor import EnsembleTable
from sdm.testing import onlyCUDA


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
    ("factory", "table", "native_atol", "sign_invariant"),
    [
        (Standardize, _numerical_table, 1e-5, False),
        (ClipQuantiles, _numerical_table, 1e-5, False),
        (ClipSigma, _numerical_table, 1e-5, False),
        (ImputeMean, _numerical_table, 1e-5, False),
        (DropConstantColumns, _numerical_table, 1e-5, False),
        (PowerTransform, _numerical_table, 5e-4, False),
        (
            lambda: QuantileTransform(n_quantiles=4, subsample=None),
            _numerical_table,
            1e-5,
            False,
        ),
        (lambda: PCA(2), _numerical_table, 1e-5, True),
        (ImputeMode, _categorical_table, 1e-5, False),
        (AlignCategories, _categorical_table, 1e-5, False),
        (
            lambda: TFIDF(ngram_range=(2, 2)),
            _text_table,
            1e-5,
            False,
        ),
    ],
)
@onlyCUDA
def test_moved_processor_matches_native_cuda_and_moves_back(
    factory: Callable[[], Processor],
    table: Callable[[], TableTensor],
    native_atol: float,
    sign_invariant: bool,
) -> None:
    cpu_table = table()
    moved = factory().fit(cpu_table)
    expected = moved.transform(cpu_table)

    moved.cuda()
    cuda_table = cast(TableTensor, cpu_table.cuda())
    moved_output = moved.transform(cuda_table)
    native_output = factory().fit_transform(cuda_table)

    assert all(buffer.is_cuda for buffer in moved.buffers())
    assert cast(TableTensor, moved_output.cpu()).allclose(expected)
    if sign_invariant:
        assert moved_output.numerical.abs().allclose(
            native_output.numerical.abs(),
            atol=native_atol,
            rtol=1e-5,
        )
    else:
        assert moved_output.allclose(
            native_output,
            atol=native_atol,
            rtol=1e-5,
        )

    moved.cpu()
    assert all(not buffer.is_cuda for buffer in moved.buffers())
    assert moved.transform(cpu_table).allclose(expected)


@pytest.mark.parametrize(
    ("factory", "table"),
    [
        (ShuffleColumns, _numerical_table),
        (ShuffleCategories, _categorical_table),
        (lambda: RandomProjection(2), _numerical_table),
    ],
)
@onlyCUDA
def test_randomized_processor_has_no_hidden_cpu_tensor_state(
    factory: Callable[[], Processor],
    table: Callable[[], TableTensor],
) -> None:
    cpu_table = table()
    processor = factory().fit(cpu_table)
    expected = processor.transform(cpu_table)

    processor.cuda()
    cuda_table = cast(TableTensor, cpu_table.cuda())
    output = processor.transform(cuda_table)
    native = factory().fit(cuda_table)
    native_output = native.transform(cuda_table)

    assert all(buffer.is_cuda for buffer in processor.buffers())
    assert all(buffer.is_cuda for buffer in native.buffers())
    assert native_output.device.type == "cuda"
    assert native_output.size() == output.size()
    assert cast(TableTensor, output.cpu()).allclose(
        expected, atol=1e-5, rtol=1e-5
    )


def _nested_pipeline() -> Sequential:
    return Sequential(
        StypeDispatch(
            numerical=Choice(
                Standardize(),
                ClipSigma(),
                method="round_robin",
            )
        ),
        StypeDispatch(numerical=ImputeMean()),
    )


def _assert_ensemble_close(
    actual: EnsembleTable,
    expected: EnsembleTable,
) -> None:
    assert actual.num_members == expected.num_members
    for member_id in range(actual.num_members):
        actual_table = actual.table(member_id)
        expected_table = expected.table(member_id)
        if actual_table.device != expected_table.device:
            actual_table = cast(
                TableTensor,
                actual_table.to(expected_table.device),
            )
        assert actual_table.allclose(expected_table)


@onlyCUDA
def test_nested_processor_state_moves_between_cpu_and_cuda() -> None:
    first = _numerical_table()
    second = TableTensor.from_tensor(_numerical_table().numerical + 10.0)
    cpu_ensemble = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )
    moved = _nested_pipeline().fit_ensemble(cpu_ensemble)
    expected = moved.transform_ensemble(cpu_ensemble)

    moved.cuda()
    cuda_ensemble = EnsembleTable.from_tables(
        tables=(
            cast(TableTensor, first.cuda()),
            cast(TableTensor, second.cuda()),
        ),
        member_table_ids=(0, 1, 0, 1),
    )
    moved_output = moved.transform_ensemble(cuda_ensemble)
    native_output = _nested_pipeline().fit_transform_ensemble(cuda_ensemble)

    assert all(buffer.is_cuda for buffer in moved.buffers())
    _assert_ensemble_close(moved_output, expected)
    _assert_ensemble_close(moved_output, native_output)

    moved.cpu()
    assert all(not buffer.is_cuda for buffer in moved.buffers())
    _assert_ensemble_close(moved.transform_ensemble(cpu_ensemble), expected)
