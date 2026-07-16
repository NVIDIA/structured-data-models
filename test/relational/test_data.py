from textwrap import dedent
from typing import Any, cast

import pandas as pd
import pytest
import torch
from sdm import RelationalData, StringTensor, Stype, TableTensor, infer_stypes

USERS = {
    "user_id": [0, 1, 2, 3],
    "age": [25, 30, 35, 40],
    "city": ["NYC", "LA", "Chicago", "NYC"],
}
ORDERS = {
    "user_id": [0, 0, 1, 3, 3, 3],
    "item_id": ["A", "B", "C", "A", "B", "A"],
    "amount": [29.99, 49.99, 19.99, 99.99, 199.99, 39.99],
}
ITEMS = {
    "item_id": ["A", "B", "C"],
    "category": ["A", "B", "C"],
}
RELATIONSHIPS = [
    {
        "left_table": "orders",
        "left_column": "user_id",
        "right_table": "users",
        "right_column": "user_id",
    },
    {
        "left_table": "orders",
        "left_column": "item_id",
        "right_table": "items",
        "right_column": "item_id",
    },
]


def test_repr(data: RelationalData) -> None:
    assert repr(data) == dedent("""\
        RelationalData(
          tables={
            users: TableTensor(
              size=(4, 3),
              blocks={
                numerical (1): [age],
                categorical (1): [city],
                id (1): [user_id],
              },
            ),
            orders: TableTensor(
              size=(6, 3),
              blocks={
                numerical (1): [amount],
                id (2): [user_id, item_id],
              },
            ),
            items: TableTensor(
              size=(3, 2),
              blocks={
                categorical (1): [category],
                id (1): [item_id],
              },
            ),
          },
          relationships=[
            orders.user_id <> users.user_id,
            orders.item_id <> items.item_id,
          ],
        )""")


@pytest.fixture
def cuda_data() -> RelationalData:
    _import_cudf()
    frames = {
        "users": pd.DataFrame(USERS),
        "orders": pd.DataFrame(ORDERS).assign(
            unused_id=range(len(ORDERS["user_id"]))
        ),
        "items": pd.DataFrame(ITEMS),
    }
    stypes = {
        "users": {"user_id": Stype.id},
        "orders": {
            "user_id": Stype.id,
            "item_id": Stype.id,
            "unused_id": Stype.id,
        },
        "items": {"item_id": Stype.id},
    }
    return RelationalData(
        tables={
            name: cast(
                TableTensor,
                TableTensor.from_pandas(
                    df=frame,
                    stypes=stypes[name],
                ).cuda(),
            )
            for name, frame in frames.items()
        },
        relationships=RELATIONSHIPS,
    )


def _import_cudf() -> Any:
    if not torch.cuda.is_available():
        cast(Any, pytest.skip)("CUDA is not available")
    return pytest.importorskip("cudf")


def _composite_data(*, cuda: bool) -> RelationalData:
    left = {
        "account_id": [1, 1, 1, 2, 9],
        "region_id": ["é", "é", "東京", "", "🙂"],
    }
    right = {
        "owner_id": [1, 1, 1, 2],
        "area_id": ["é", "é", "東京", ""],
    }
    if cuda:
        cudf = _import_cudf()
        left_df = cudf.DataFrame(left)
        right_df = cudf.DataFrame(right)
        left_table = TableTensor.from_cudf(
            df=left_df,
            stypes={
                "account_id": Stype.id,
                "region_id": Stype.id,
            },
        )
        right_table = TableTensor.from_cudf(
            df=right_df,
            # Deliberately opposite the relationship key order.
            stypes={"area_id": Stype.id, "owner_id": Stype.id},
        )
    else:
        left_df = pd.DataFrame(left)
        right_df = pd.DataFrame(right)
        left_table = TableTensor.from_pandas(
            df=left_df,
            stypes={
                "account_id": Stype.id,
                "region_id": Stype.id,
            },
        )
        right_table = TableTensor.from_pandas(
            df=right_df,
            # Deliberately opposite the relationship key order.
            stypes={"area_id": Stype.id, "owner_id": Stype.id},
        )

    return RelationalData(
        tables={"left": left_table, "right": right_table},
        relationships=[
            {
                "left_table": "left",
                "left_columns": ("account_id", "region_id"),
                "right_table": "right",
                "right_columns": ("owner_id", "area_id"),
            }
        ],
    )


def _edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    return sorted(tuple(pair) for pair in edge_index.cpu().t().tolist())


def test_edge_indices_cpu(data: RelationalData) -> None:
    edge_indices = data.edge_indices()

    assert len(edge_indices) == 2
    assert edge_indices[0].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 3, 3, 3]])
    )
    assert edge_indices[1].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 0]])
    )


def test_edge_indices_cuda(cuda_data: RelationalData) -> None:
    edge_indices = cuda_data.edge_indices()

    assert all(edge_index.device.type == "cuda" for edge_index in edge_indices)
    assert _edge_pairs(edge_indices[0]) == [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 3),
        (4, 3),
        (5, 3),
    ]
    assert _edge_pairs(edge_indices[1]) == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 0),
        (4, 1),
        (5, 0),
    ]


@pytest.mark.parametrize(
    "noncontiguous",
    [False, True],
    ids=["contiguous", "noncontiguous"],
)
def test_edge_indices_cuda_does_not_export_to_host(
    cuda_data: RelationalData,
    monkeypatch: pytest.MonkeyPatch,
    noncontiguous: bool,
) -> None:
    cudf = _import_cudf()

    relational_data = cuda_data
    if noncontiguous:
        relational_data = RelationalData(
            tables={
                name: cuda_data.tables[name][::2]
                for name in ("orders", "users")
            },
            # Exercise non-contiguous numeric conversion without the known
            # dynamic-size synchronization of compacting string storage.
            relationships=RELATIONSHIPS[:1],
        )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("GPU edge materialization exported data to host")

    with monkeypatch.context() as host_guard:
        host_guard.setattr(TableTensor, "to_arrow", fail)
        host_guard.setattr(cudf.DataFrame, "to_arrow", fail)
        host_guard.setattr(cudf.DataFrame, "to_pandas", fail)
        host_guard.setattr(cudf.Series, "to_dlpack", fail)
        host_guard.setattr(cudf.Series, "to_numpy", fail)

        previous_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            edge_indices = relational_data.edge_indices()
            torch.cuda.synchronize()
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)

    assert all(edge_index.device.type == "cuda" for edge_index in edge_indices)


@pytest.mark.parametrize("cuda", [False, True], ids=["cpu", "cuda"])
def test_edge_indices_composite_keys_and_duplicates(cuda: bool) -> None:
    edge_index = _composite_data(cuda=cuda).edge_indices()[0]

    assert _edge_pairs(edge_index) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]


@pytest.mark.parametrize("cuda", [False, True], ids=["cpu", "cuda"])
def test_edge_indices_empty(cuda: bool) -> None:
    relational_data = _composite_data(cuda=cuda)
    right = relational_data.tables["right"][:0]
    relational_data = RelationalData(
        tables={**relational_data.tables, "right": right},
        relationships=relational_data.relationships,
    )

    edge_index = relational_data.edge_indices()[0]

    assert edge_index.size() == (2, 0)
    assert edge_index.device.type == ("cuda" if cuda else "cpu")


def test_edge_indices_cuda_sliced_string_keys() -> None:
    relational_data = _composite_data(cuda=True)
    left = relational_data.tables["left"][1:]
    right = relational_data.tables["right"][1:]
    region_id = left[..., ["region_id"]].id.unbind(-1)[0]
    assert isinstance(region_id, StringTensor)
    assert region_id.storage_offset() > 0
    relational_data = RelationalData(
        tables={"left": left, "right": right},
        relationships=relational_data.relationships,
    )

    edge_index = relational_data.edge_indices()[0]

    assert _edge_pairs(edge_index) == [(0, 0), (1, 1), (2, 2)]


def test_edge_indices_cuda_noncontiguous_string_keys() -> None:
    relational_data = _composite_data(cuda=True)
    left = relational_data.tables["left"][::2]
    right = relational_data.tables["right"][::2]
    region_id = left[..., ["region_id"]].id.unbind(-1)[0]
    area_id = right[..., ["area_id"]].id.unbind(-1)[0]
    assert isinstance(region_id, StringTensor)
    assert isinstance(area_id, StringTensor)
    assert not region_id.is_contiguous()
    assert not area_id.is_contiguous()
    relational_data = RelationalData(
        tables={"left": left, "right": right},
        relationships=relational_data.relationships,
    )

    edge_index = relational_data.edge_indices()[0]

    assert _edge_pairs(edge_index) == [(0, 0), (1, 1)]


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires at least two CUDA devices",
)
def test_edge_indices_non_default_cuda_device() -> None:
    _import_cudf()
    relational_data = _composite_data(cuda=False)
    relational_data = RelationalData(
        tables={
            name: cast(TableTensor, table.cuda(1))
            for name, table in relational_data.tables.items()
        },
        relationships=relational_data.relationships,
    )

    edge_index = relational_data.edge_indices()[0]

    assert edge_index.device == torch.device("cuda:1")


def test_edge_indices_rejects_mixed_table_devices(
    data: RelationalData,
) -> None:
    cudf = _import_cudf()
    users_df = cudf.DataFrame(USERS)
    users = TableTensor.from_cudf(
        df=users_df,
        stypes=infer_stypes(users_df),
    )
    relational_data = RelationalData(
        tables={**data.tables, "users": users},
        relationships=RELATIONSHIPS,
    )

    with pytest.raises(RuntimeError, match="same device"):
        relational_data.edge_indices()


def test_edge_indices_respects_output_options(
    data: RelationalData,
    cuda_data: RelationalData,
) -> None:
    cpu_to_cuda = data.edge_indices(dtype=torch.int32, device="cuda")
    cuda_to_cpu = cuda_data.edge_indices(dtype=torch.int32, device="cpu")

    assert all(
        edge_index.device.type == "cuda" and edge_index.dtype == torch.int32
        for edge_index in cpu_to_cuda
    )
    assert all(
        edge_index.device.type == "cpu" and edge_index.dtype == torch.int32
        for edge_index in cuda_to_cpu
    )
