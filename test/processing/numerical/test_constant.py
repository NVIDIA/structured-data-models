import pytest
import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import DropConstantColumns
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
    table = TableTensor.from_tensor(data)

    output = DropConstantColumns().fit_transform(table)

    assert output.columns[Stype.numerical] == (
        "1",
        "2",
        "3",
    )
    assert output.numerical.equal(data[:, [1, 2, 3]])


@withCUDA
def test_unique_filter_with_higher_threshold(device: torch.device) -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 3.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ],
            device=device,
        ),
    )

    output = DropConstantColumns(threshold=2).fit_transform(table)

    assert output.columns[Stype.numerical] == ("2",)


def test_unique_filter_keeps_all_columns_with_too_few_rows() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0, 4.0], [1.0, 4.0]]))

    assert DropConstantColumns(threshold=2).fit_transform(table) is table


def test_drop_constant_columns_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        DropConstantColumns(threshold=0)


@withCUDA
def test_drop_constant_columns_ensemble_matches_member_fits(
    device: torch.device,
) -> None:
    first_context = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [1.0, 3.0]], device=device),
    )
    second_context = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [3.0, 2.0]], device=device),
    )
    member_table_ids = (1, 0, 1, 0, 0, 1, 1, 0)
    context = EnsembleTable.from_tables(
        tables=(first_context, second_context),
        member_table_ids=member_table_ids,
    )
    query = TableTensor.from_tensor(
        torch.tensor([[4.0, 5.0], [6.0, 7.0]], device=device),
    )
    processor = DropConstantColumns()

    context_output = processor.fit_transform_ensemble(context)
    query_output = processor.transform_ensemble(
        EnsembleTable.from_table(query, num_members=len(member_table_ids))
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


def test_drop_constant_columns_requires_fitted_member_count() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    processor = DropConstantColumns().fit_ensemble(
        EnsembleTable.from_table(table, num_members=2)
    )

    with pytest.raises(RuntimeError, match="same number"):
        processor.transform_ensemble(
            EnsembleTable.from_table(table, num_members=1)
        )


@withCUDA
def test_unique_filter_nan(device: torch.device) -> None:
    data = torch.tensor(
        [
            [float("nan"), 1.0, 1.0, 1.0, float("inf"), float("inf")],
            [float("nan"), 1.0, float("nan"), 2.0, float("inf"), float("nan")],
            [float("nan"), 1.0, 1.0, float("nan"), float("inf"), float("inf")],
            [float("nan"), 1.0, float("nan"), 2.0, float("inf"), float("nan")],
        ],
        device=device,
    )
    table = TableTensor.from_tensor(data)

    output = DropConstantColumns(threshold=2).fit_transform(table)

    assert output.columns[Stype.numerical] == ("3",)
    torch.testing.assert_close(output.numerical, data[:, [3]], equal_nan=True)
