from typing import cast

import pytest
import torch

from sdm import CategoricalTensor, EnsembleTable, TableTensor
from sdm.processing import RandomProjection
from sdm.testing import withCUDA


def _table() -> TableTensor:
    return TableTensor(
        numerical=torch.randn(6, 4),
        categorical=CategoricalTensor(
            torch.randint(0, 2, (6, 1)), categories=(torch.arange(2),)
        ),
    )


def test_random_projection() -> None:
    table = _table()

    inp = EnsembleTable.from_table(table, num_members=8)
    out = RandomProjection(8).fit_transform_ensemble(inp)

    assert out.num_groups == 1
    assert out.num_members == 8
    group = out._groups[0]
    assert group.size() == (8, 6, 9)
    assert group.numerical.size() == (8, 6, 8)
    assert group.numerical.stride() == (6 * 8, 8, 1)
    assert group.categorical.size() == (8, 6, 1)
    assert group.categorical.stride() == (0, 1, 1)


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("positions", [(1, 0), (1, 0, 1, 1)])
def test_reordered_members_match_expanded_projection(
    device: torch.device,
    batch_shape: tuple[int, ...],
    dtype: torch.dtype,
    positions: tuple[int, ...],
) -> None:
    values = torch.randn(2, *batch_shape, 18, 5, device=device, dtype=dtype)
    codes = torch.randint(0, 3, (2, *batch_shape, 9, 1), device=device)
    group = TableTensor(
        numerical=values[..., ::2, :],
        categorical=CategoricalTensor(
            code=codes,
            categories=(torch.arange(3, device=device),),
        ),
    )
    before = cast(TableTensor, group.clone())
    inp = EnsembleTable(
        groups=(group,),
        locations=tuple((0, position) for position in positions),
    )
    expanded = EnsembleTable(
        groups=(inp.expanded_group(0),),
        locations=tuple((0, i) for i in range(len(positions))),
    )
    processor = RandomProjection(3).fit_ensemble(inp)

    actual = processor.transform_ensemble(inp)
    expected = processor.transform_ensemble(expanded)
    repeated = processor.transform_ensemble(inp)

    assert actual.num_groups == expected.num_groups == 1
    assert actual.num_members == expected.num_members == len(positions)
    for member in range(actual.num_members):
        output = actual.table(member)
        torch.testing.assert_close(
            output.numerical, expected.table(member).numerical
        )
        assert output.categorical.equal(inp.table(member).categorical)
        assert output.columns == expected.table(member).columns
        assert output.equal(repeated.table(member))
    assert group.equal(before)


@withCUDA
def test_reordered_members_preserve_autocast_dtype(
    device: torch.device,
) -> None:
    inp = EnsembleTable(
        groups=(TableTensor(numerical=torch.randn(2, 9, 5, device=device)),),
        locations=((0, 1), (0, 0), (0, 1)),
    )
    processor = RandomProjection(3).fit_ensemble(inp)
    expanded = EnsembleTable(
        groups=(inp.expanded_group(0),),
        locations=((0, 0), (0, 1), (0, 2)),
    )

    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = processor.transform_ensemble(inp)
        expected = processor.transform_ensemble(expanded)

    for member in range(actual.num_members):
        torch.testing.assert_close(
            actual.table(member).numerical, expected.table(member).numerical
        )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_batched_members_preserve_projection_and_metadata(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    values = torch.randn(2, 3, 12, 5, device=device, dtype=dtype)[..., ::2, :]
    table = TableTensor(
        numerical=values,
        categorical=CategoricalTensor(
            code=torch.randint(0, 3, (2, 3, 6, 1), device=device),
            categories=(torch.arange(3, device=device),),
        ),
    )
    before = cast(TableTensor, table.clone())
    inp = EnsembleTable.from_table(table, num_members=4)
    expanded = EnsembleTable(
        groups=(cast(TableTensor, inp.expanded_group(0).clone()),),
        locations=tuple((0, i) for i in range(4)),
    )
    processor = RandomProjection(3).fit_ensemble(inp)

    actual = processor.transform_ensemble(inp)
    expected = processor.transform_ensemble(expanded)

    assert actual.num_groups == expected.num_groups == 1
    assert actual.num_members == expected.num_members == 4
    for member in range(actual.num_members):
        torch.testing.assert_close(
            actual.table(member).numerical, expected.table(member).numerical
        )
        assert actual.table(member).categorical.equal(table.categorical)
    assert table.equal(before)
