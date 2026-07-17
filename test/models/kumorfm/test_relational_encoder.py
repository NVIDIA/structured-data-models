from dataclasses import dataclass
from typing import cast

import torch
from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.models.kumorfm.relational_encoder import (
    RelationalEncoder,
)
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from torch import Tensor


@dataclass
class _Call:
    x: Tensor
    y: Tensor
    train_mask: Tensor
    max_keys: int | None
    generator: torch.Generator | None


class _RecordingRowEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 2)
        self.calls: list[_Call] = []

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        train_mask: Tensor,
        max_keys: int | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        self.calls.append(
            _Call(
                x=x.detach().clone(),
                y=y.detach().clone(),
                train_mask=train_mask.detach().clone(),
                max_keys=max_keys,
                generator=generator,
            )
        )
        return x.new_full((x.size(0), 2), float(len(self.calls)))


def _table(values: list[float], **ids: list[int]) -> TableTensor:
    return TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.id: tuple(ids),
        },
        numerical=torch.tensor(values).unsqueeze(-1),
        id=ColumnarTensor(
            tuple(torch.tensor(value) for value in ids.values())
        ),
    )


def _task(entity_id: list[int], values: list[float]) -> TableTensor:
    return _table(values, entity_id=entity_id)


def _related_tables(*, query: bool) -> RelatedTables:
    if query:
        users = _table(
            [4.0, 3.0],
            entity_id=[40, 30],
            group_id=[2, 2],
        )
        groups = _table([20.0], group_id=[2], topic_id=[6])
        topics = _table([60.0], topic_id=[6])
    else:
        users = _table(
            [2.0, 1.0, 9.0],
            entity_id=[20, 10, 99],
            group_id=[1, 1, 9],
        )
        groups = _table(
            [10.0, 90.0],
            group_id=[1, 9],
            topic_id=[5, 9],
        )
        topics = _table([50.0, 90.0], topic_id=[5, 9])

    return RelatedTables(
        tables={"users": users, "groups": groups, "topics": topics},
        relationships=[
            {
                "left_table": "users",
                "left_column": "group_id",
                "right_table": "groups",
                "right_column": "group_id",
            },
            {
                "left_table": "groups",
                "left_column": "topic_id",
                "right_table": "topics",
                "right_column": "topic_id",
            },
        ],
        task_links=[
            {
                "task_column": "entity_id",
                "table": "users",
                "table_column": "entity_id",
            }
        ],
    )


def _encoder() -> tuple[RelationalEncoder, _RecordingRowEmbedding]:
    row_embedding = _RecordingRowEmbedding()
    return (
        RelationalEncoder(cast(RowEmbedding, row_embedding)),
        row_embedding,
    )


def test_classification_combines_graph_and_encodes_each_table_once() -> None:
    encoder, row_embedding = _encoder()
    x_context = _task([10, 20], [100.0, 200.0])
    x_query = _task([30, 40], [300.0, 400.0])
    generator = torch.Generator().manual_seed(123)

    out = encoder(
        x_context=x_context,
        y_context=torch.tensor([1, 0]),
        x_query=x_query,
        related_context_tables=_related_tables(query=False),
        related_query_tables=_related_tables(query=True),
        num_hops=2,
        max_keys=17,
        generator=generator,
    )

    assert out.entity_table == "users"
    assert out.root_index.equal(torch.tensor([1, 0, 4, 3]))
    assert set(out.x_dict) == {"users", "groups", "topics"}
    assert len(row_embedding.calls) == 3
    assert all(call.max_keys == 17 for call in row_embedding.calls)
    assert all(call.generator is generator for call in row_embedding.calls)

    users, groups, topics = row_embedding.calls
    torch.testing.assert_close(
        users.x,
        torch.tensor(
            [
                [2.0, 200.0],
                [1.0, 100.0],
                [9.0, 0.0],
                [4.0, 400.0],
                [3.0, 300.0],
            ]
        ),
    )
    assert users.train_mask.equal(
        torch.tensor([True, True, False, False, False])
    )
    assert users.y.equal(torch.tensor([0, 1]))

    assert groups.train_mask.equal(torch.tensor([True, False, False]))
    assert groups.y.equal(torch.tensor([0]))
    assert topics.train_mask.equal(torch.tensor([True, False, False]))
    assert topics.y.equal(torch.tensor([0]))

    assert out.edge_index_dict[("users", "0", "groups")].equal(
        torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 2]])
    )
    assert out.edge_index_dict[("groups", "1", "topics")].equal(
        torch.tensor([[0, 1, 2], [0, 1, 2]])
    )


def test_regression_propagates_values_with_support_counts() -> None:
    encoder, row_embedding = _encoder()
    context_tables = _related_tables(query=False).select_tables(
        ("users", "groups")
    )
    query_tables = _related_tables(query=True).select_tables(
        ("users", "groups")
    )

    encoder(
        x_context=_task([10, 20], [100.0, 200.0]),
        y_context=torch.tensor([2.0, 6.0]),
        x_query=_task([30, 40], [300.0, 400.0]),
        related_context_tables=context_tables,
        related_query_tables=query_tables,
        num_hops=1,
    )

    assert len(row_embedding.calls) == 2
    users, groups = row_embedding.calls
    torch.testing.assert_close(users.y, torch.tensor([6.0, 2.0]))
    torch.testing.assert_close(groups.y, torch.tensor([4.0]))
    assert groups.train_mask.equal(torch.tensor([True, False, False]))


def test_num_hops_is_supplied_by_the_caller() -> None:
    encoder, row_embedding = _encoder()

    encoder(
        x_context=_task([10, 20], [100.0, 200.0]),
        y_context=torch.tensor([1, 0]),
        x_query=_task([30, 40], [300.0, 400.0]),
        related_context_tables=_related_tables(query=False),
        related_query_tables=_related_tables(query=True),
        num_hops=1,
    )

    topics = row_embedding.calls[2]
    assert not topics.train_mask.any()
    assert topics.y.numel() == 0
