from dataclasses import dataclass
from typing import cast

import pytest
import torch
from sdm import (
    ColumnarTensor,
    RelatedTables,
    RelationalData,
    Stype,
    TableTensor,
)
from sdm.models.kumorfm.table_hop_encoder import (
    TableHopEncoder,
    _ordered_roots,
)
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.relational.sampler import EXAMPLE_ID
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


def _table(
    *,
    example: list[int],
    ids: dict[str, list[int]],
    value: list[float],
) -> TableTensor:
    return TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.id: (EXAMPLE_ID, *ids),
        },
        numerical=torch.tensor(value).unsqueeze(-1),
        id=ColumnarTensor(
            (
                torch.tensor(example),
                *(torch.tensor(ids[name]) for name in ids),
            )
        ),
    )


def _task(
    *, example: list[int], entity_id: list[int], value: list[float]
) -> TableTensor:
    return TableTensor(
        columns={
            Stype.numerical: ("task_value",),
            Stype.id: (EXAMPLE_ID, "entity_id"),
        },
        numerical=torch.tensor(value).unsqueeze(-1),
        id=ColumnarTensor((torch.tensor(example), torch.tensor(entity_id))),
    )


def _related(
    *,
    entity: TableTensor,
    orders: TableTensor,
) -> RelatedTables:
    return RelatedTables(
        tables={"entity": entity, "orders": orders},
        relationships=(
            {
                "left_table": "orders",
                "left_columns": (EXAMPLE_ID, "owner_id"),
                "right_table": "entity",
                "right_columns": (EXAMPLE_ID, "entity_id"),
            },
            {
                "left_table": "orders",
                "left_columns": (EXAMPLE_ID, "referred_id"),
                "right_table": "entity",
                "right_columns": (EXAMPLE_ID, "entity_id"),
            },
        ),
        task_links=(
            {
                "task_columns": (EXAMPLE_ID, "entity_id"),
                "table": "entity",
                "table_columns": (EXAMPLE_ID, "entity_id"),
            },
        ),
    )


def _inputs() -> tuple[
    TableTensor,
    TableTensor,
    RelatedTables,
    RelatedTables,
]:
    x_context = _task(example=[0, 1], entity_id=[10, 11], value=[100.0, 200.0])
    x_query = _task(example=[0], entity_id=[20], value=[300.0])
    related_context = _related(
        entity=_table(
            example=[1, 0],
            ids={"entity_id": [11, 10]},
            value=[11.0, 10.0],
        ),
        orders=_table(
            example=[0, 1],
            ids={
                "order_id": [100, 101],
                "owner_id": [10, 11],
                "referred_id": [99, 99],
            },
            value=[1.0, 2.0],
        ),
    )
    related_query = _related(
        entity=_table(
            example=[0, 0],
            ids={"entity_id": [20, 21]},
            value=[20.0, 21.0],
        ),
        orders=_table(
            example=[0],
            ids={
                "order_id": [200],
                "owner_id": [20],
                "referred_id": [21],
            },
            value=[3.0],
        ),
    )
    return x_context, x_query, related_context, related_query


def test_combines_context_query_and_encodes_table_hops() -> None:
    x_context, x_query, related_context, related_query = _inputs()
    row_embedding = _RecordingRowEmbedding()
    generator = torch.Generator().manual_seed(7)

    result = TableHopEncoder(cast(RowEmbedding, row_embedding))(
        x_context=x_context,
        y_context=torch.tensor([4, 8]),
        x_query=x_query,
        related_context_tables=related_context,
        related_query_tables=related_query,
        max_keys=11,
        generator=generator,
    )

    assert result.root_index.tolist() == [1, 0, 2]
    assert result.num_hops == 2
    assert list(result.edge_index_dict) == [
        ("orders", "0", "entity"),
        ("orders", "1", "entity"),
    ]
    assert result.edge_index_dict[("orders", "0", "entity")].tolist() == [
        [0, 1, 2],
        [1, 0, 2],
    ]
    assert result.edge_index_dict[("orders", "1", "entity")].tolist() == [
        [2],
        [3],
    ]

    assert len(row_embedding.calls) == 3
    entity_root, entity_hop_two, orders = row_embedding.calls
    assert entity_root.x.tolist() == [
        [11.0, 200.0],
        [10.0, 100.0],
        [20.0, 300.0],
    ]
    assert entity_root.y.tolist() == [8, 4]
    assert entity_root.train_mask.tolist() == [True, True, False]
    assert entity_hop_two.x.tolist() == [[21.0]]
    assert entity_hop_two.y.numel() == 0
    assert entity_hop_two.train_mask.tolist() == [True]
    assert orders.x.tolist() == [[1.0], [2.0], [3.0]]
    assert orders.y.tolist() == [4, 8]
    assert orders.train_mask.tolist() == [True, True, False]
    assert all(call.max_keys == 11 for call in row_embedding.calls)
    assert all(call.generator is generator for call in row_embedding.calls)

    assert result.x_dict["entity"].tolist() == [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    assert result.x_dict["orders"].tolist() == [
        [3.0, 3.0],
        [3.0, 3.0],
        [3.0, 3.0],
    ]


def test_context_only_table_is_retained() -> None:
    x_context, x_query, related_context, related_query = _inputs()
    query_entity = related_query.tables["entity"]
    entity_only_query = RelatedTables(
        tables={"entity": query_entity[:1]},
        relationships=(),
        task_links=related_query.task_links,
    )

    result = TableHopEncoder(cast(RowEmbedding, _RecordingRowEmbedding()))(
        x_context=x_context,
        y_context=torch.tensor([4, 8]),
        x_query=x_query,
        related_context_tables=related_context,
        related_query_tables=entity_only_query,
    )

    assert result.num_hops == 1
    assert result.x_dict["orders"].size(0) == 2


def test_task_root_contract() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _ordered_roots(
            torch.tensor([[0, 0], [1, 2]]),
            num_rows=2,
        )

    x_context, x_query, related_context, related_query = _inputs()
    mismatched_query = RelatedTables(
        tables=related_query.tables,
        relationships=related_query.relationships,
        task_links=(),
    )
    with pytest.raises(ValueError, match="same task links"):
        TableHopEncoder(cast(RowEmbedding, _RecordingRowEmbedding()))(
            x_context=x_context,
            y_context=torch.tensor([4, 8]),
            x_query=x_query,
            related_context_tables=related_context,
            related_query_tables=mismatched_query,
        )


def test_sampler_output_preserves_task_root_order() -> None:
    pytest.importorskip("pyg_lib")
    data = RelationalData(
        tables={
            "entity": TableTensor(
                columns={
                    Stype.numerical: ("value",),
                    Stype.id: ("entity_id",),
                },
                numerical=torch.tensor([[1.0], [2.0], [3.0]]),
                id=ColumnarTensor((torch.tensor([10, 20, 30]),)),
            )
        },
        relationships=(),
    )
    sampler = data.sampler()

    def sample(entity_ids: list[int]) -> tuple[TableTensor, RelatedTables]:
        task_table = TableTensor(
            columns={Stype.id: ("entity_id",)},
            id=ColumnarTensor((torch.tensor(entity_ids),)),
        )
        return sampler(
            task_table=task_table,
            task_link={
                "task_column": "entity_id",
                "table": "entity",
                "table_column": "entity_id",
            },
            num_neighbors=(),
        )

    x_context, related_context = sample([30, 10])
    x_query, related_query = sample([20])
    result = TableHopEncoder(cast(RowEmbedding, _RecordingRowEmbedding()))(
        x_context=x_context,
        y_context=torch.tensor([0, 1]),
        x_query=x_query,
        related_context_tables=related_context,
        related_query_tables=related_query,
    )

    entity_ids = torch.cat(
        (
            related_context.tables["entity"].id[..., 0],
            related_query.tables["entity"].id[..., 0],
        )
    )
    assert entity_ids[result.root_index].tolist() == [30, 10, 20]
