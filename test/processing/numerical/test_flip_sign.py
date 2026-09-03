import pytest
import torch

from sdm import TableTensor
from sdm.processing import FlipSign
from sdm.tensor import EnsembleTable


def _table(offset: float = 0.0) -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        + offset
    )


def test_flip_sign_is_column_consistent_and_reused() -> None:
    context = _table()
    query = _table(offset=10.0)
    processor = FlipSign()

    transformed = processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=8),
        generator=torch.Generator().manual_seed(0),
    )
    query_transformed = processor.transform_ensemble(
        EnsembleTable(query, num_members=8)
    )
    restored = processor.inverse_transform_ensemble(transformed)

    member_signs = []
    for member_id in range(transformed.num_members):
        output = transformed.table(member_id).numerical
        signs = output / context.numerical
        torch.testing.assert_close(signs, signs[:1].expand_as(signs))
        assert torch.equal(signs.abs(), torch.ones_like(signs))
        torch.testing.assert_close(
            query_transformed.table(member_id).numerical,
            query.numerical * signs[:1],
        )
        assert restored.table(member_id).equal(context)
        member_signs.append(signs[0])

    signs = torch.stack(member_signs)
    assert signs.unique(dim=0).size(0) > 1
    assert bool((signs == -1).any())
    assert bool((signs == 1).any())


def test_flip_sign_is_reproducible_with_generator() -> None:
    ensemble_table = EnsembleTable(_table(), num_members=8)

    first = FlipSign().fit_transform_ensemble(
        ensemble_table,
        generator=torch.Generator().manual_seed(4),
    )
    second = FlipSign().fit_transform_ensemble(
        ensemble_table,
        generator=torch.Generator().manual_seed(4),
    )

    for member_id in range(ensemble_table.num_members):
        assert first.table(member_id).equal(second.table(member_id))


def test_flip_sign_reuses_signs_for_logical_members() -> None:
    table = TableTensor.from_tensor(torch.arange(1.0, 49.0).view(3, 16))
    tables = (table[:2], table)
    context = EnsembleTable.from_tables(
        tables=tables,
        member_table_ids=(0, 1),
    )
    processor = FlipSign()
    context_output = processor.fit_transform_ensemble(
        context,
        generator=torch.Generator().manual_seed(0),
    )
    reordered = EnsembleTable.from_tables(
        tables=tables[::-1],
        member_table_ids=(1, 0),
    )
    reordered_fit_output = FlipSign().fit_transform_ensemble(
        reordered,
        generator=torch.Generator().manual_seed(0),
    )
    queries = (
        reordered,
        EnsembleTable(table, num_members=2),
    )

    for member_id in range(context.num_members):
        assert reordered_fit_output.table(member_id).equal(
            context_output.table(member_id)
        )

    for query in queries:
        output = processor.transform_ensemble(query)
        for member_id in range(context.num_members):
            sign = (
                context_output.table(member_id).numerical[:1]
                / context.table(member_id).numerical[:1]
            )
            torch.testing.assert_close(
                output.table(member_id).numerical,
                query.table(member_id).numerical * sign,
            )


@pytest.mark.parametrize(
    ("probability", "factor"),
    [(0.0, 1.0), (1.0, -1.0)],
)
def test_flip_sign_probability_extremes(
    probability: float,
    factor: float,
) -> None:
    table = _table()
    processor = FlipSign(probability=probability)

    transformed = processor.fit_transform(table)
    restored = processor.inverse_transform(transformed)

    torch.testing.assert_close(
        transformed.numerical,
        table.numerical * factor,
    )
    assert restored.equal(table)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_flip_sign_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        FlipSign(probability=probability)
