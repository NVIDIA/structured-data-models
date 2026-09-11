from typing import Literal

import pytest
import torch

import sdm.processing as sp
from sdm import ColumnarTensor, EnsembleTable, TableTensor
from sdm.testing import withCUDA


@withCUDA
def test_reduce_estimators_mean(device: torch.device) -> None:
    values = torch.arange(
        2 * 3 * 4 * 2,
        dtype=torch.float32,
        device=device,
    ).reshape(2, 3, 4, 2)
    table = TableTensor.from_tensor(values)
    processor = sp.ReduceEstimators(method="mean")

    output = processor.transform(table)
    fit_output = processor.fit_transform(table)

    assert output.size() == (3, 4, 2)
    assert output.schema == table.schema
    assert output.dtype == values.dtype
    torch.testing.assert_close(output.numerical, values.mean(dim=0))
    torch.testing.assert_close(fit_output.numerical, output.numerical)
    assert repr(processor) == "ReduceEstimators(method='mean')"


def test_reduce_estimators_rejects_missing_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.ones(4, 2))

    with pytest.raises(ValueError, match="leading ensemble dimension"):
        sp.ReduceEstimators().transform(table)


def test_reduce_estimators_rejects_empty_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.empty(0, 4, 2))

    with pytest.raises(ValueError, match="at least one ensemble member"):
        sp.ReduceEstimators().transform(table)


def test_reduce_estimators_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be 'mean'"):
        sp.ReduceEstimators(method="median")  # type: ignore


def test_reduce_estimators_rejects_non_numerical_stypes() -> None:
    table = TableTensor(
        numerical=torch.ones(2, 3, 2),
        id=ColumnarTensor((torch.arange(2 * 3).reshape(2, 3),)),
    )

    with pytest.raises(ValueError, match="numerical-only output table"):
        sp.ReduceEstimators().transform(table)


def test_reduce_estimators_rejects_non_numerical_ensemble_stypes() -> None:
    member = TableTensor(
        numerical=torch.ones(2, 1),
        id=ColumnarTensor((torch.arange(2),)),
    )
    table = EnsembleTable.from_tables(
        tables=(member, member),
        member_table_ids=(0, 1),
    )

    with pytest.raises(ValueError, match="numerical-only output table"):
        sp.ReduceEstimators().transform_ensemble(table)


@withCUDA
def test_reduce_estimators_reduces_members_in_order(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(torch.tensor([[0.0, 2.0]], device=device))
    second = TableTensor.from_tensor(torch.tensor([[3.0, 1.0]], device=device))
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 1),
    )

    output = sp.ReduceEstimators().transform_ensemble(table)

    assert output.num_members == 1
    torch.testing.assert_close(
        output.table(0).numerical,
        (first.numerical + 2 * second.numerical) / 3,
    )


@withCUDA
def test_reduce_estimators_reduces_across_storage_groups(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[0.0, 2.0]], device=device),
        columns=("a", "b"),
    )
    second = TableTensor.from_tensor(
        torch.tensor([[4.0, 6.0]], device=device),
        columns=("b", "a"),
    )
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 1),
    )

    output = sp.ReduceEstimators().transform_ensemble(table)

    assert output.table(0).columns == first.columns
    torch.testing.assert_close(
        output.table(0).numerical,
        torch.tensor([[4.0, 10.0 / 3.0]], device=device),
    )


def test_reduce_estimators_rejects_empty_ensemble_table() -> None:
    table = TableTensor.from_tensor(torch.ones(4, 2))

    with pytest.raises(ValueError, match="at least one ensemble member"):
        sp.ReduceEstimators().transform_ensemble(
            EnsembleTable.from_table(table, num_members=0)
        )


@withCUDA
def test_reduce_estimators_composes_with_following_processor(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(torch.tensor([[0.0, 2.0]], device=device))
    second = TableTensor.from_tensor(torch.tensor([[2.0, 0.0]], device=device))
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    output = sp.Sequential(
        sp.ReduceEstimators(), sp.Softmax()
    ).transform_ensemble(table)

    assert output.num_members == 1
    torch.testing.assert_close(
        output.table(0).numerical,
        torch.full((1, 2), 0.5, device=device),
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "expanded"])
def test_reduce_estimators_preserves_strided_batched_inputs(
    device: torch.device,
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
    layout: Literal["contiguous", "strided", "expanded"],
) -> None:
    values = torch.randn(4, *batch_shape, 6, 3, device=device, dtype=dtype)
    if layout == "strided":
        values = values.transpose(-1, -2)
    elif layout == "expanded":
        values = values[:1].expand_as(values)
    before = values.clone()
    ensemble = EnsembleTable(
        groups=(
            TableTensor(numerical=values[:2]),
            TableTensor(numerical=values[2:]),
        ),
        locations=((0, 1), (1, 0), (0, 1), (0, 0), (1, 1)),
    )
    expected = torch.stack(
        [ensemble.table(i).numerical for i in range(ensemble.num_members)]
    ).mean(dim=0)
    processor = sp.ReduceEstimators()

    actual = processor.transform_ensemble(ensemble)
    repeated = processor.transform_ensemble(ensemble)

    torch.testing.assert_close(actual.table(0).numerical, expected)
    torch.testing.assert_close(
        repeated.table(0).numerical, actual.table(0).numerical
    )
    torch.testing.assert_close(values, before)


@withCUDA
@pytest.mark.parametrize("strided", [False, True])
def test_reduce_estimators_aligns_first_stored_group_to_first_member(
    device: torch.device,
    strided: bool,
) -> None:
    first = torch.randn(2, 3, 10, 3, device=device)
    second = torch.randn(2, 3, 10, 3, device=device)
    if strided:
        first, second = first[..., ::2, :], second[..., ::2, :]
    ensemble = EnsembleTable(
        groups=(
            TableTensor.from_tensor(first, columns=("c", "a", "b")),
            TableTensor.from_tensor(second, columns=("a", "b", "c")),
        ),
        locations=((1, 0), (0, 1), (0, 0), (1, 0)),
    )
    expected = second[0] * 2 + first[1][..., [1, 2, 0]]
    expected = (expected + first[0][..., [1, 2, 0]]) / 4

    actual = sp.ReduceEstimators().transform_ensemble(ensemble).table(0)

    assert actual.columns == ensemble.table(0).columns
    torch.testing.assert_close(actual.numerical, expected)


@withCUDA
@pytest.mark.parametrize("reordered", [False, True])
def test_reduce_estimators_preserves_mixed_group_dtype_arithmetic(
    device: torch.device,
    reordered: bool,
) -> None:
    first = torch.tensor([[[1.0, 2.0]]], device=device)
    second = torch.tensor(
        [[[-1.0 + 1e-9, -2.0 + 1e-9]]],
        dtype=torch.float64,
        device=device,
    )
    if reordered:
        second = second.flip(-1)
    ensemble = EnsembleTable(
        groups=(
            TableTensor.from_tensor(first, columns=("a", "b")),
            TableTensor.from_tensor(
                second, columns=("b", "a") if reordered else ("a", "b")
            ),
        ),
        locations=((0, 0), (1, 0)),
    )
    expected = first[0].clone()
    expected.add_(second[0].flip(-1) if reordered else second[0]).div_(2)

    actual = sp.ReduceEstimators().transform_ensemble(ensemble).table(0)

    assert actual.numerical.dtype == torch.float32
    torch.testing.assert_close(actual.numerical, expected, rtol=1e-5, atol=0)


@withCUDA
@pytest.mark.parametrize(
    ("dtype", "large"), [(torch.float16, 2048), (torch.bfloat16, 256)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_reduce_estimators_preserves_low_precision_group_accumulation(
    device: torch.device,
    dtype: torch.dtype,
    large: int,
    strided: bool,
) -> None:
    first = torch.full((1, 2, 2), -large, device=device, dtype=dtype)
    second = (
        torch.tensor([1.0, float(large)], device=device, dtype=dtype)
        .view(2, 1, 1)
        .expand(2, 2, 2)
    )
    if not strided:
        second = second.contiguous()
    ensemble = EnsembleTable(
        groups=(TableTensor(numerical=first), TableTensor(numerical=second)),
        locations=((0, 0), (1, 0), (1, 1)),
    )
    expected = torch.tensordot(first.new_ones(1), first, dims=([0], [0]))
    expected.add_(
        torch.tensordot(second.new_ones(2), second, dims=([0], [0]))
    ).div_(3)

    actual = sp.ReduceEstimators().transform_ensemble(ensemble).table(0)

    torch.testing.assert_close(actual.numerical, expected, rtol=0, atol=0)


@withCUDA
@pytest.mark.parametrize("strided", [False, True])
def test_reduce_estimators_preserves_unused_nonfinite_members(
    device: torch.device,
    strided: bool,
) -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[float("nan"), float("inf"), -float("inf"), 5.0]],
        ],
        device=device,
    ).expand(2, 3, 4)
    if not strided:
        values = values.contiguous()
    ensemble = EnsembleTable(
        groups=(TableTensor(numerical=values),), locations=((0, 0),)
    )
    # Zero-weight nonfinite values propagate through the existing contraction.
    expected = torch.tensordot(
        values.new_tensor([1, 0]), values, dims=([0], [0])
    )

    actual = sp.ReduceEstimators().transform_ensemble(ensemble).table(0)

    torch.testing.assert_close(actual.numerical, expected, equal_nan=True)


@withCUDA
def test_reduce_estimators_preserves_empty_rows(device: torch.device) -> None:
    ensemble = EnsembleTable(
        groups=(
            TableTensor(numerical=torch.empty(2, 0, 3, device=device)),
            TableTensor(numerical=torch.empty(1, 0, 3, device=device)),
        ),
        locations=((0, 0), (1, 0), (0, 1)),
    )

    actual = sp.ReduceEstimators().transform_ensemble(ensemble).table(0)

    assert actual.numerical.shape == (0, 3)
    assert actual.columns == ensemble.table(0).columns


@withCUDA
def test_reduce_estimators_preserves_autocast_reduction(
    device: torch.device,
) -> None:
    first = torch.randn(2, 3, 8, device=device).transpose(-1, -2)
    second = torch.randn(1, 8, 3, device=device)
    ensemble = EnsembleTable(
        groups=(TableTensor(numerical=first), TableTensor(numerical=second)),
        locations=((0, 0), (1, 0), (0, 1), (0, 1)),
    )
    with torch.autocast(device.type, dtype=torch.bfloat16):
        expected = torch.tensordot(
            first.new_tensor([1, 2]), first, dims=([0], [0])
        )
        expected.add_(
            torch.tensordot(second.new_ones(1), second, dims=([0], [0]))
        ).div_(4)
        actual = sp.ReduceEstimators().transform_ensemble(ensemble).table(0)

    torch.testing.assert_close(actual.numerical, expected)
