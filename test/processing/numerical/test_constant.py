import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import DropConstantColumns
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


@withCUDA
def test_unique_filter(device: torch.device) -> None:
    data = torch.tensor(
        [
            [1.0, 0.0, 3.0, 5.0, 9.0],
            [1.0, 1.0, 3.0, 6.0, 9.0],
            [1.0, 2.0, 4.0, 5.0, 9.0],
            [1.0, 3.0, 4.0, 6.0, 9.0],
        ],
        device=device,
    )
    table = TableTensor.from_tensor(
        data,
        columns=(
            "constant",
            "variable",
            "two_values",
            "two_more_values",
            "also_constant",
        ),
    )

    output = DropConstantColumns().fit_transform(table)

    assert output.columns[Stype.numerical] == (
        "variable",
        "two_values",
        "two_more_values",
    )
    assert output.numerical.equal(data[:, [1, 2, 3]])
    assert output.device == device


@withCUDA
def test_unique_filter_with_higher_threshold(device: torch.device) -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [
                [1.0, -torch.inf, -torch.inf, torch.nan],
                [1.0, torch.inf, 0.0, torch.nan],
                [torch.nan, torch.nan, torch.inf, torch.nan],
                [1.0, -torch.inf, torch.nan, torch.nan],
            ],
            device=device,
        ),
        columns=("one", "two", "three", "missing"),
    )

    output = DropConstantColumns(threshold=2).fit_transform(table)

    assert output.columns[Stype.numerical] == ("three",)


@withCUDA
def test_unique_filter_ignores_missing_values_and_replays_selection(
    device: torch.device,
) -> None:
    context = TableTensor.from_tensor(
        torch.tensor(
            [
                [torch.nan, torch.nan, torch.nan],
                [torch.nan, 4.0, 1.0],
                [torch.nan, 4.0, 2.0],
            ],
            device=device,
        ),
        columns=("missing", "constant", "variable"),
    )
    query = TableTensor.from_tensor(
        torch.tensor([[8.0, 9.0, 10.0]], device=device),
        columns=context.columns[Stype.numerical],
    )
    processor = DropConstantColumns().fit(context)

    context_output = processor.transform(context)
    query_output = processor.transform(query)

    assert context_output.columns[Stype.numerical] == ("variable",)
    torch.testing.assert_close(
        context_output.numerical,
        context.numerical[:, 2:],
        equal_nan=True,
    )
    assert torch.equal(query_output.numerical, query.numerical[:, 2:])


@withCUDA
def test_unique_filter_one_row_drops_only_all_missing(
    device: torch.device,
) -> None:
    context = TableTensor.from_tensor(
        torch.tensor([[torch.nan, 4.0]], device=device),
        columns=("missing", "observed"),
    )
    output = DropConstantColumns().fit_transform(context)

    assert output.columns[Stype.numerical] == ("observed",)
    assert torch.equal(output.numerical, context.numerical[:, 1:])


def test_unique_filter_keeps_all_columns_with_too_few_rows() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0, 4.0], [1.0, 4.0]]))

    assert DropConstantColumns(threshold=2).fit_transform(table) is table


@withCUDA
def test_variance_filter(device: torch.device) -> None:
    data = torch.tensor(
        [
            [1.0, 1.0, 1.0, 8.0],
            [1.0, 1.0000005, 2.0, 8.0],
            [1.0, 0.9999995, 3.0, 8.0],
            [1.0, 1.0000002, 4.0, 8.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(
        data,
        columns=("constant", "near_constant", "variable", "also_constant"),
    )

    output = DropConstantColumns(method="variance").fit_transform(table)

    assert output.columns[Stype.numerical] == ("variable",)
    assert output.numerical.equal(data[:, [2]])
    assert output.device == device


def test_drop_constant_columns_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="tolerance must be None"):
        DropConstantColumns(tolerance=1e-6)
    with pytest.raises(ValueError, match="threshold must be None"):
        DropConstantColumns(method="variance", threshold=1)
    with pytest.raises(ValueError, match="threshold must be positive"):
        DropConstantColumns(threshold=0)
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        DropConstantColumns(method="variance", tolerance=-1.0)


@withCUDA
def test_drop_constant_columns_ensemble_matches_member_fits(
    device: torch.device,
) -> None:
    first_context = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [1.0, 3.0]], device=device),
        columns=("a", "b"),
    )
    second_context = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [3.0, 2.0]], device=device),
        columns=("a", "b"),
    )
    member_table_ids = (1, 0, 1, 0, 0, 1, 1, 0)
    context = EnsembleTable.from_tables(
        tables=(first_context, second_context),
        member_table_ids=member_table_ids,
    )
    query = TableTensor.from_tensor(
        torch.tensor([[4.0, 5.0], [6.0, 7.0]], device=device),
        columns=("a", "b"),
    )
    processor = DropConstantColumns()

    context_output = processor.fit_transform_ensemble(context)
    query_output = processor.transform_ensemble(
        EnsembleTable(query, num_members=len(member_table_ids))
    )
    separate_query_output = processor.transform_ensemble(
        EnsembleTable.from_tables(
            tables=(query,) * len(member_table_ids),
            member_table_ids=range(len(member_table_ids)),
        )
    )
    references = [
        DropConstantColumns().fit(first_context),
        DropConstantColumns().fit(second_context),
    ]
    context_tables = (first_context, second_context)
    for member_id, table_id in enumerate(member_table_ids):
        reference = references[table_id]
        assert context_output.table(member_id).equal(
            reference.transform(context_tables[table_id])
        )
        expected_query = reference.transform(query)
        assert query_output.table(member_id).equal(expected_query)
        assert separate_query_output.table(member_id).equal(expected_query)
        assert query_output.table(member_id).device == device


def test_drop_constant_columns_requires_fitted_member_count() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    processor = DropConstantColumns().fit_ensemble(
        EnsembleTable(table, num_members=2)
    )

    with pytest.raises(RuntimeError, match="same number"):
        processor.transform_ensemble(EnsembleTable(table, num_members=1))
