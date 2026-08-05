import pyarrow as pa
import pytest

from sdm import RelatedTables, TableTensor


def test_to_nim_config() -> None:
    task_table = TableTensor.from_columns(
        data={"user_id": [1, 2], "__example__": [0, 1]},
        stypes={"user_id": "id", "__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={
            "users": TableTensor.from_columns(
                data={
                    "age": [30, 40],
                    "user_id": [1, 2],
                    "__example__": [0, 1],
                },
                stypes={
                    "age": "numerical",
                    "user_id": "id",
                    "__example__": "id",
                },
            ),
            "orders": TableTensor.from_columns(
                data={
                    "amount": [10.0, 20.0, 30.0],
                    "user_id": [1, 1, 2],
                    "order_id": [10, 11, 12],
                    "__example__": [0, 0, 1],
                },
                stypes={
                    "amount": "numerical",
                    "user_id": "id",
                    "order_id": "id",
                    "__example__": "id",
                },
            ),
        },
        relationships=[
            {
                "left_table": "orders",
                "left_columns": ("user_id", "__example__"),
                "right_table": "users",
                "right_columns": ("user_id", "__example__"),
            }
        ],
        task_links=[
            {
                "task_columns": ("user_id", "__example__"),
                "table": "users",
                "table_columns": ("user_id", "__example__"),
            }
        ],
    )

    config = related_tables.to_nim_config(task_table)

    assert config["schema"]["instance_table"]["primary_key"] == "instance_id"
    assert config["schema"]["related_tables"]["users"]["primary_key"] == [
        "instance_id",
        "user_id",
    ]
    assert config["schema"]["related_tables"]["orders"]["primary_key"] == [
        "instance_id",
        "__node_id",
    ]
    assert config["schema"]["relationships"] == [
        {
            "source_columns": ["user_id"],
            "target_table": "users",
            "target_columns": ["user_id"],
        },
        {
            "source_table": "orders",
            "source_columns": ["user_id"],
            "target_table": "users",
            "target_columns": ["user_id"],
        },
    ]
    instance_data = config["data"]["instance_table"]
    assert isinstance(instance_data, pa.Table)
    assert instance_data.to_pydict() == {
        "user_id": [1, 2],
        "instance_id": [0, 1],
    }
    orders_data = config["data"]["related_tables"]["orders"]
    assert orders_data.column_names == [
        "amount",
        "user_id",
        "order_id",
        "instance_id",
        "__node_id",
    ]
    assert orders_data.to_pydict() == {
        "amount": [10.0, 20.0, 30.0],
        "user_id": [1, 1, 2],
        "order_id": [10, 11, 12],
        "instance_id": [0, 0, 1],
        "__node_id": [0, 1, 2],
    }

    tables = [
        instance_data,
        *config["data"]["related_tables"].values(),
    ]
    for table in tables:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        assert pa.ipc.open_stream(sink.getvalue()).read_all().equals(table)


def test_to_nim_config_requires_sample_scope() -> None:
    task_table = TableTensor.from_columns(
        data={"user_id": [1]},
        stypes={"user_id": "id"},
    )
    related_tables = RelatedTables(
        tables={},
        relationships=[],
        task_links=[],
    )

    with pytest.raises(ValueError, match="sampler column '__example__'"):
        related_tables.to_nim_config(task_table)


def test_to_nim_config_validates_relationship_scope() -> None:
    task_table = TableTensor.from_columns(
        data={"user_id": [1], "__example__": [0]},
        stypes={"user_id": "id", "__example__": "id"},
    )
    users = TableTensor.from_columns(
        data={"user_id": [1], "__example__": [0]},
        stypes={"user_id": "id", "__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={"users": users},
        relationships=[],
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )

    with pytest.raises(ValueError, match="aligned '__example__' scope"):
        related_tables.to_nim_config(task_table)


def test_to_nim_config_requires_task_root_match() -> None:
    task_table = TableTensor.from_columns(
        data={"user_id": [2], "__example__": [0]},
        stypes={"user_id": "id", "__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={
            "users": TableTensor.from_columns(
                data={"user_id": [1], "__example__": [0]},
                stypes={"user_id": "id", "__example__": "id"},
            )
        },
        relationships=[],
        task_links=[
            {
                "task_columns": ("user_id", "__example__"),
                "table": "users",
                "table_columns": ("user_id", "__example__"),
            }
        ],
    )

    with pytest.raises(ValueError, match="match exactly one same-instance"):
        related_tables.to_nim_config(task_table)


def test_to_nim_config_requires_matching_join_dtypes() -> None:
    task_table = TableTensor.from_columns(
        data={"user_id": [1], "__example__": [0]},
        stypes={"user_id": "id", "__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={
            "users": TableTensor.from_columns(
                data={"user_id": ["1"], "__example__": [0]},
                stypes={"user_id": "id", "__example__": "id"},
            )
        },
        relationships=[],
        task_links=[
            {
                "task_columns": ("user_id", "__example__"),
                "table": "users",
                "table_columns": ("user_id", "__example__"),
            }
        ],
    )

    with pytest.raises(ValueError, match="same NIM dtype"):
        related_tables.to_nim_config(task_table)


def test_to_nim_config_validates_task_links_per_table() -> None:
    task_table = TableTensor.from_columns(
        data={
            "user_id": [1],
            "other_user_id": [1],
            "__example__": [0],
        },
        stypes={
            "user_id": "id",
            "other_user_id": "id",
            "__example__": "id",
        },
    )
    users = TableTensor.from_columns(
        data={"user_id": [1], "__example__": [0]},
        stypes={"user_id": "id", "__example__": "id"},
    )
    link = {
        "task_columns": ("user_id", "__example__"),
        "table": "users",
        "table_columns": ("user_id", "__example__"),
    }
    related_tables = RelatedTables(
        tables={"users": users},
        relationships=[],
        task_links=[link, link],
    )

    config = related_tables.to_nim_config(task_table)
    assert len(config["schema"]["relationships"]) == 1

    related_tables = RelatedTables(
        tables={"users": users},
        relationships=[],
        task_links=[
            link,
            {
                "task_columns": ("other_user_id", "__example__"),
                "table": "users",
                "table_columns": ("user_id", "__example__"),
            },
        ],
    )
    with pytest.raises(ValueError, match="at most one distinct task link"):
        related_tables.to_nim_config(task_table)


def test_to_nim_config_allows_non_unique_relationship_targets() -> None:
    task_table = TableTensor.from_columns(
        data={"__example__": [0]},
        stypes={"__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={
            "orders": TableTensor.from_columns(
                data={"item_id": [10], "__example__": [0]},
                stypes={"item_id": "id", "__example__": "id"},
            ),
            "items": TableTensor.from_columns(
                data={"item_id": [10, 10], "__example__": [0, 0]},
                stypes={"item_id": "id", "__example__": "id"},
            ),
        },
        relationships=[
            {
                "left_table": "orders",
                "left_columns": ("item_id", "__example__"),
                "right_table": "items",
                "right_columns": ("item_id", "__example__"),
            }
        ],
        task_links=[],
    )

    config = related_tables.to_nim_config(task_table)

    assert config["schema"]["related_tables"]["items"]["primary_key"] == [
        "instance_id",
        "__node_id",
    ]


def test_to_nim_config_rejects_unknown_instance() -> None:
    task_table = TableTensor.from_columns(
        data={"__example__": [0]},
        stypes={"__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={
            "events": TableTensor.from_columns(
                data={"event_id": [10], "__example__": [1]},
                stypes={"event_id": "id", "__example__": "id"},
            )
        },
        relationships=[],
        task_links=[],
    )

    with pytest.raises(ValueError, match="to reference the task table"):
        related_tables.to_nim_config(task_table)


def test_to_nim_config_uses_collision_safe_synthetic_key() -> None:
    task_table = TableTensor.from_columns(
        data={"__example__": [0]},
        stypes={"__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={
            "events": TableTensor.from_columns(
                data={
                    "__node_id": [100],
                    "__example__": [0],
                },
                stypes={"__node_id": "id", "__example__": "id"},
            )
        },
        relationships=[],
        task_links=[],
    )

    config = related_tables.to_nim_config(task_table)

    assert config["schema"]["related_tables"]["events"]["primary_key"] == [
        "instance_id",
        "__node_id_1",
    ]


def test_to_nim_config_leaves_name_limits_to_client() -> None:
    name = "x" * 1_025
    task_table = TableTensor.from_columns(
        data={"__example__": [0]},
        stypes={"__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={
            name: TableTensor.from_columns(
                data={"__example__": [0]},
                stypes={"__example__": "id"},
            )
        },
        relationships=[],
        task_links=[],
    )

    config = related_tables.to_nim_config(task_table)

    assert name in config["data"]["related_tables"]


@pytest.mark.parametrize(
    ("table_name", "column_name", "match"),
    [
        ("instance_table", "__example__", "reserved by NIM"),
        ("events", "instance_id", "column 'instance_id' is reserved"),
    ],
)
def test_to_nim_config_rejects_reserved_names(
    table_name: str,
    column_name: str,
    match: str,
) -> None:
    task_table = TableTensor.from_columns(
        data={"__example__": [0]},
        stypes={"__example__": "id"},
    )
    related_tables = RelatedTables(
        tables={
            table_name: TableTensor.from_arrow(
                pa.table({column_name: [0], "__example__": [0]}),
                stypes={column_name: "id", "__example__": "id"},
            )
        },
        relationships=[],
        task_links=[],
    )

    with pytest.raises(ValueError, match=match):
        related_tables.to_nim_config(task_table)
