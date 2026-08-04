from typing import Literal

import pytest
import torch

from sdm import (
    CategoricalTensor,
    EnsembleTable,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import ShuffleColumns


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
            code=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["a", "b"]),
                StringTensor.from_list(["small", "large"]),
            ),
        ),
    )


def test_shuffle_columns_shift_rotates_numerical_block() -> None:
    table = _table()
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three columns

    output = ShuffleColumns(method="shift").fit_transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("x1", "x2", "x0")
    assert torch.equal(
        output.numerical,
        table.numerical.index_select(-1, torch.tensor([1, 2, 0])),
    )


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_columns_is_reproducible_with_generator(
    method: Literal["shift", "random"],
) -> None:
    table = _table()

    first = ShuffleColumns(method=method)
    first_output = first.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    second = ShuffleColumns(method=method)
    second_output = second.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(first.permutation, second.permutation)
    assert torch.equal(first_output.numerical, second_output.numerical)


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_columns_ensemble_matches_independent_processors(
    method: Literal["shift", "random"],
) -> None:
    context = _table()
    query = context.replace_blocks(numerical=context.numerical + 10)
    ensemble = ShuffleColumns(method=method)
    ensemble_generator = torch.Generator().manual_seed(7)

    context_output = ensemble.fit_transform_ensemble(
        EnsembleTable(context, num_members=8),
        generator=ensemble_generator,
    )
    query_output = ensemble.transform_ensemble(
        EnsembleTable(query, num_members=8)
    )
    restored = ensemble.inverse_transform_ensemble(context_output)

    reference_generator = torch.Generator().manual_seed(7)
    references = [ShuffleColumns(method=method) for _ in range(8)]
    for member_id, processor in enumerate(references):
        expected_context = processor.fit_transform(
            context,
            generator=reference_generator,
        )
        expected_query = processor.transform(query)
        assert context_output.representation(member_id).equal(expected_context)
        assert query_output.representation(member_id).equal(expected_query)
        assert restored.representation(member_id).equal(context)


def test_shuffle_columns_reuses_equal_member_permutations() -> None:
    processor = ShuffleColumns(method="shift")
    output = processor.fit_transform_ensemble(
        EnsembleTable(_table(), num_members=8),
        generator=torch.Generator().manual_seed(9),
    )

    unique_columns = {
        output.representation(member_id).columns[Stype.numerical]
        for member_id in range(output.num_members)
    }
    assert sum(
        packed.size(0) for packed in output.iter_packed_representations()
    ) == len(unique_columns)

    with pytest.raises(RuntimeError, match="fitted for an ensemble"):
        processor.transform(_table())
    with pytest.raises(RuntimeError, match="inverse transform"):
        processor.inverse_transform_ensemble(
            EnsembleTable(_table(), num_members=7)
        )


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_columns_fit_ensemble_matches_fit_transform(
    method: Literal["shift", "random"],
) -> None:
    table = EnsembleTable(_table(), num_members=8)
    fitted = ShuffleColumns(method=method)
    combined = ShuffleColumns(method=method)

    fitted.fit_ensemble(
        table,
        generator=torch.Generator().manual_seed(7),
    )
    transformed = fitted.transform_ensemble(table)
    expected = combined.fit_transform_ensemble(
        table,
        generator=torch.Generator().manual_seed(7),
    )

    for member_id in range(table.num_members):
        assert transformed.table(member_id).equal(expected.table(member_id))
