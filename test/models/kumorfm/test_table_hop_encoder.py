from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

import pytest
import torch
from sdm import CategoricalTensor, ColumnarTensor, RelatedTables, TableTensor
from sdm.models.kumorfm.table_hop_encoder import (
    TableHopEncoder,
    _encode_datetime_features,
    _fit_kumo_categorical_align,
    _preprocess_features,
)
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.relational.sampler import EXAMPLE_ID
from test.models.kumorfm.sample_utils import with_full_sample
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
        value = float(len(self.calls))
        return x.new_full((x.size(0), 2), value)


def _table(
    *,
    example: list[int],
    ids: dict[str, list[int]],
    value: list[float],
) -> TableTensor:
    id_columns = (EXAMPLE_ID, *ids)
    return TableTensor(
        columns={
            "numerical": ("value",),
            "id": id_columns,
        },
        numerical=torch.tensor(value).unsqueeze(-1),
        id=ColumnarTensor(
            (
                torch.tensor(example),
                *(torch.tensor(ids[column]) for column in ids),
            )
        ),
    )


def _related_tables() -> RelatedTables:
    related_tables = RelatedTables(
        tables={
            # Roots occupy rows 0, 1, and 3. Row 2 is a second-hop entity.
            "entity": _table(
                example=[2, 0, 2, 1],
                ids={"entity_id": [2, 0, 3, 1]},
                value=[30.0, 10.0, 40.0, 20.0],
            ),
            "orders": _table(
                example=[0, 2, 1],
                ids={
                    "order_id": [10, 12, 11],
                    "owner_id": [0, 2, 1],
                    "referred_id": [99, 3, 99],
                },
                value=[1.0, 3.0, 2.0],
            ),
            "inactive": _table(
                example=[2],
                ids={"order_id": [12]},
                value=[5.0],
            ),
        },
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
            {
                "left_table": "inactive",
                "left_columns": (EXAMPLE_ID, "order_id"),
                "right_table": "orders",
                "right_columns": (EXAMPLE_ID, "order_id"),
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
    return with_full_sample(
        related_tables,
        num_task_rows=3,
        root_index=torch.tensor([1, 3, 0]),
    )


def test_multiple_hops_unlabeled_hop_pruning_and_row_alignment() -> None:
    related_tables = _related_tables()
    row_embedding = _RecordingRowEmbedding()
    encoder = TableHopEncoder(cast(RowEmbedding, row_embedding))
    generator = torch.Generator().manual_seed(7)

    result = encoder(
        x=torch.tensor([[100.0], [200.0], [300.0]]),
        y=torch.tensor([4, 8]),
        related_tables=related_tables,
        max_keys=11,
        generator=generator,
    )
    out = result.x_dict

    assert list(out) == ["entity", "orders"]
    assert list(result.edge_index_dict) == [
        ("orders", "0", "entity"),
        ("orders", "1", "entity"),
    ]
    assert result.num_hops == 2
    assert len(row_embedding.calls) == 3

    entity_root, entity_hop_two, orders = row_embedding.calls
    torch.testing.assert_close(
        entity_root.x,
        torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert entity_root.y.tolist() == [4, 8]
    assert entity_root.train_mask.tolist() == [False, True, True]

    assert torch.equal(entity_hop_two.x, torch.zeros(1, 1))
    assert entity_hop_two.y.numel() == 0
    assert entity_hop_two.train_mask.tolist() == [True]

    assert orders.y.tolist() == [4, 8]
    assert orders.train_mask.tolist() == [True, False, True]
    assert all(call.max_keys == 11 for call in row_embedding.calls)
    assert all(call.generator is generator for call in row_embedding.calls)

    # Hop-wise calls are scattered back to table order. In particular, the
    # second-hop entity remains at local row 2, as addressed by the edge.
    assert torch.equal(
        out["entity"],
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [1.0, 1.0]]),
    )
    assert torch.equal(out["orders"], torch.full((3, 2), 3.0))
    assert result.edge_index_dict[("orders", "1", "entity")].equal(
        torch.tensor([[1], [2]])
    )
    assert result.root_index.tolist() == [1, 3, 0]
    assert result.root_index.max() < result.x_dict["entity"].size(0)


def test_categorical_preprocessing_matches_kumo_table_mapping() -> None:
    table = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [1], [2]]),
            categories=(torch.tensor([20, 10, 30]),),
        ),
    )
    align = _fit_kumo_categorical_align(table[:3])

    actual = align.transform(table).categorical.as_tensor().flatten()

    assert actual.tolist() == [1, 0, 0, -1]


def test_numerical_preprocessing_matches_kumo_train_statistics() -> None:
    values = torch.tensor(
        [
            [10.0, -40.0, torch.nan, 0.0],
            [20.0, -20.0, torch.nan, 1.0],
            [30.0, 20.0, torch.nan, 2.0],
            [40.0, 40.0, torch.nan, 3.0],
            [-100.0, -100.0, 5.0, 100.0],
            [1_000.0, 1_000.0, torch.nan, 1_000.0],
        ]
    )
    fit_mask = torch.tensor([True, True, True, True, False, False])
    actual = _preprocess_features(
        TableTensor.from_tensor(values[:, :3]),
        train_mask=fit_mask,
        task_x=values[:, 3:],
        batch=torch.arange(values.size(0)),
        seed_time=None,
        device=torch.device("cpu"),
        dtype=torch.float,
    )

    low, high = torch.nanquantile(
        values[fit_mask], torch.tensor([0.02, 0.98]), dim=0
    )
    low = torch.nan_to_num(low, nan=float("-inf")).clamp(max=0.0)
    high = torch.nan_to_num(high, nan=float("inf"))
    expected = values.clamp(min=low, max=high)
    expected = expected.where(expected.isfinite(), 0.0)
    keep = (expected[fit_mask] != expected[fit_mask][:1]).any(dim=0)
    expected = expected[:, keep]
    train = expected[fit_mask]
    expected = (expected - train.mean(dim=0)) / (
        train.std(dim=0, correction=0) + 1e-6
    )

    torch.testing.assert_close(actual, expected.clamp(-15.0, 15.0))


def test_relational_input_without_exact_sample_is_rejected() -> None:
    sampled = _related_tables()
    related_tables = RelatedTables(
        tables=sampled.tables,
        relationships=sampled.relationships,
        task_links=sampled.task_links,
    )

    row_embedding = _RecordingRowEmbedding()
    with pytest.raises(ValueError, match="require exact sample metadata"):
        TableHopEncoder(cast(RowEmbedding, row_embedding))(
            x=torch.tensor([[100.0], [200.0], [300.0]]),
            y=torch.tensor([4, 8]),
            related_tables=related_tables,
        )


def test_datetime_features_match_kumo_and_use_anchor_time() -> None:
    timestamp = datetime(
        2024,
        2,
        29,
        13,
        45,
        0,
        900_000,
        tzinfo=timezone.utc,
    )
    first_anchor = datetime(
        2024, 3, 1, microsecond=100_000, tzinfo=timezone.utc
    )
    second_anchor = datetime(
        2024, 3, 2, microsecond=100_000, tzinfo=timezone.utc
    )

    def to_microseconds(value: datetime) -> int:
        return int(value.timestamp() * 1_000_000)

    missing = torch.iinfo(torch.int64).min
    values = torch.tensor(
        [[to_microseconds(timestamp), missing]],
        dtype=torch.long,
    )
    first = _encode_datetime_features(
        timestamp=values,
        anchor_time=torch.tensor([to_microseconds(first_anchor)]),
        dtype=torch.float,
    )
    second = _encode_datetime_features(
        timestamp=values,
        anchor_time=torch.tensor([to_microseconds(second_anchor)]),
        dtype=torch.float,
    )

    expected = torch.tensor(
        [
            45 / 60,
            13 / 24,
            3 / 7,
            28 / 29,
            59 / 366,
            (10 * 60 + 15) / (24 * 60),
        ]
    )
    torch.testing.assert_close(first[0, :6], expected)
    assert torch.equal(first[0, 6:], torch.zeros(6))
    torch.testing.assert_close(second[0, :5], first[0, :5])
    torch.testing.assert_close(second[0, 5], first[0, 5] + 1)

    table = TableTensor(
        columns={"datetime": ("known", "missing")},
        datetime=values,
    )
    with pytest.raises(ValueError, match="anchor times"):
        _preprocess_features(
            table,
            train_mask=torch.tensor([True]),
            task_x=None,
            batch=torch.tensor([0]),
            seed_time=None,
            device=torch.device("cpu"),
            dtype=torch.float,
        )


def test_real_row_embedding_produces_finite_aligned_rows() -> None:
    row_embedding = RowEmbedding(
        num_classes=10,
        channels=4,
        num_layers=1,
        num_heads=1,
        group_size=1,
        num_inducing_points=2,
        num_readout_tokens=1,
        norm_bias=True,
    )
    for module in row_embedding.modules():
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.02)

    result = TableHopEncoder(row_embedding)(
        x=torch.tensor([[100.0], [200.0], [300.0]]),
        y=torch.tensor([4, 8]),
        related_tables=_related_tables(),
        generator=torch.Generator().manual_seed(7),
    )

    assert result.x_dict["entity"].size() == (4, 4)
    assert result.x_dict["orders"].size() == (3, 4)
    assert all(torch.isfinite(value).all() for value in result.x_dict.values())
