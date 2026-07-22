import pytest
import torch
from sdm import ColumnarTensor, RelatedTables, RelationalData, TableTensor
from sdm.testing import withCUDA


def _related_tables(relational_data: RelationalData) -> RelatedTables:
    return RelatedTables(
        tables=relational_data.tables,
        relationships=relational_data.relationships,
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )


def _task_table(
    user_ids: list[int],
    device: torch.device,
) -> TableTensor:
    return TableTensor(
        columns={"id": ("user_id",)},
        id=ColumnarTensor((torch.tensor(user_ids, device=device),)),
    )


@withCUDA
def test_task_indices_repeated_task_ids(
    relational_data: RelationalData,
    device: torch.device,
) -> None:
    related_tables = _related_tables(relational_data)

    (task_index,) = related_tables.task_indices(
        _task_table([3, 0, 3, 1], device)
    )

    assert task_index.size() == (2, 4)
    assert task_index[1][task_index[0].argsort()].equal(
        torch.tensor([3, 0, 3, 1], device=device)
    )


@withCUDA
def test_task_indices_unmatched_task_row(
    relational_data: RelationalData,
    device: torch.device,
) -> None:
    related_tables = _related_tables(relational_data)

    with pytest.raises(ValueError, match="task rows without a match"):
        related_tables.task_indices(_task_table([0, 99, 3], device))


@withCUDA
def test_task_indices_duplicate_table_keys(device: torch.device) -> None:
    related_tables = RelatedTables(
        tables={"users": _task_table([0, 1, 1, 2], device)},
        relationships=[],
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )

    with pytest.raises(ValueError, match="duplicate keys"):
        related_tables.task_indices(_task_table([0, 1, 2], device))
