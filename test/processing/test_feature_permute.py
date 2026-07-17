from typing import Literal

import pytest
import torch
from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import FeaturePermute


def _table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        columns=("x0", "x1", "x2"),
    )


def _mixed_table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("x0", "x1", "x2"),
            "categorical": ("kind", "segment"),
        },
        numerical=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["a", "b"]),
                StringTensor.from_list(["small", "large"]),
            ),
        ),
    )


def test_feature_permute_shift_rotates_numerical_block() -> None:
    table = _table()
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three columns

    output = FeaturePermute(method="shift").fit_transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("x1", "x2", "x0")
    assert torch.equal(
        output.numerical,
        table.numerical.index_select(-1, torch.tensor([1, 2, 0])),
    )


@pytest.mark.parametrize("method", ["shift", "random"])
def test_feature_permute_is_reproducible_with_generator(
    method: Literal["shift", "random"],
) -> None:
    table = _table()

    first = FeaturePermute(method=method)
    first_output = first.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    second = FeaturePermute(method=method)
    second_output = second.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(first.permutation, second.permutation)
    assert torch.equal(first_output.numerical, second_output.numerical)


def test_feature_permute_inverse_uses_cached_permutation() -> None:
    table = _table()
    processor = FeaturePermute(method="random").fit(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    transformed = processor.transform(table)
    restored = processor.inverse_transform(transformed)

    assert torch.equal(
        processor.inverse_permutation,
        processor.permutation.argsort(),
    )
    assert "inverse_permutation" not in processor.state_dict()
    assert restored.columns[Stype.numerical] == table.columns[Stype.numerical]
    assert torch.equal(restored.numerical, table.numerical)


def test_feature_permute_rebuilds_inverse_cache_lazily() -> None:
    table = _table()
    processor = FeaturePermute(method="shift").fit(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    transformed = processor.transform(table)
    processor.inverse_permutation = torch.empty(0, dtype=torch.long)
    processor._inverse_permutation_indices = ()

    restored = processor.inverse_transform(transformed)

    assert torch.equal(
        processor.inverse_permutation,
        processor.permutation.argsort(),
    )
    assert restored.columns[Stype.numerical] == table.columns[Stype.numerical]
    assert torch.equal(restored.numerical, table.numerical)
