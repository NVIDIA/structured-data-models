from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from sdm.stype import Stype
from sdm.tensor.io.nim import to_nim_config as table_to_nim_config
from sdm.tensor.table import TableTensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sdm.relational.task import RelatedTables

_EXAMPLE_ID = "__example__"
_INSTANCE_ID = "instance_id"
_INSTANCE_TABLE = "instance_table"
_SYNTHETIC_NODE_ID = "__node_id"


def to_nim_config(
    related_tables: RelatedTables,
    task_table: TableTensor,
) -> dict[str, Any]:
    r"""Convert sampled task and related tables to NIM fragments."""
    _validate_table_names(related_tables)

    instance_arrow, instance_stypes = _scoped_table(
        task_table,
        path="task table",
    )
    prepared_tables = {
        name: _scoped_table(table, path=f"related table {name!r}")
        for name, table in related_tables.tables.items()
    }

    task_relationships: list[dict[str, Any]] = []
    task_relationship_by_table: dict[str, dict[str, Any]] = {}
    logical_primary_keys: dict[str, tuple[str, ...]] = {}
    for link in related_tables.task_links:
        target = _related_table(related_tables, link.table)
        source_columns, target_columns = _logical_join_columns(
            task_table,
            link.task_columns,
            source_path="task table",
            target=target,
            target_columns=link.table_columns,
            target_path=f"related table {link.table!r}",
        )
        previous = logical_primary_keys.setdefault(
            link.table,
            target_columns,
        )
        if previous != target_columns:
            raise ValueError(
                f"Cannot infer one NIM primary key for related table "
                f"{link.table!r} from task links targeting {previous!r} "
                f"and {target_columns!r}"
            )
        relationship = {
            "source_columns": list(source_columns),
            "target_table": link.table,
            "target_columns": list(target_columns),
        }
        previous_relationship = task_relationship_by_table.get(link.table)
        if previous_relationship is None:
            task_relationship_by_table[link.table] = relationship
            task_relationships.append(relationship)
        elif previous_relationship != relationship:
            raise ValueError(
                f"Expected at most one distinct task link targeting related "
                f"table {link.table!r}"
            )

    relationships: list[dict[str, Any]] = []
    for relationship in related_tables.relationships:
        source = _related_table(related_tables, relationship.left_table)
        target = _related_table(related_tables, relationship.right_table)
        source_columns, target_columns = _logical_join_columns(
            source,
            relationship.left_columns,
            source_path=f"related table {relationship.left_table!r}",
            target=target,
            target_columns=relationship.right_columns,
            target_path=f"related table {relationship.right_table!r}",
        )
        relationships.append(
            {
                "source_table": relationship.left_table,
                "source_columns": list(source_columns),
                "target_table": relationship.right_table,
                "target_columns": list(target_columns),
            }
        )

    instance_config = table_to_nim_config(
        instance_arrow,
        instance_stypes,
        primary_key=_INSTANCE_ID,
    )
    table_configs: dict[str, dict[str, Any]] = {}
    for name, (table, stypes) in prepared_tables.items():
        primary_key = logical_primary_keys.get(name)
        if primary_key is None:
            table, stypes, synthetic_key = _add_synthetic_key(table, stypes)
            primary_key = (synthetic_key,)
        table_configs[name] = table_to_nim_config(
            table,
            stypes,
            primary_key=(_INSTANCE_ID, *primary_key),
        )

    _validate_instance_references(instance_config, table_configs)
    _validate_join_dtypes(
        instance_config,
        table_configs,
        [*task_relationships, *relationships],
    )
    _validate_task_roots(
        instance_config,
        table_configs,
        task_relationships,
    )

    return {
        "schema": {
            "instance_table": instance_config["schema"],
            "related_tables": {
                name: config["schema"]
                for name, config in table_configs.items()
            },
            "relationships": [*task_relationships, *relationships],
        },
        "data": {
            "instance_table": instance_config["data"],
            "related_tables": {
                name: config["data"] for name, config in table_configs.items()
            },
        },
    }


def _validate_table_names(related_tables: RelatedTables) -> None:
    for name in related_tables.tables:
        if not isinstance(name, str):
            raise TypeError(
                f"Expected related table names to be strings (got {name!r})"
            )
        if name == _INSTANCE_TABLE:
            raise ValueError(
                f"Related table name {_INSTANCE_TABLE!r} is reserved by NIM"
            )


def _scoped_table(
    table: TableTensor,
    *,
    path: str,
) -> tuple[pa.Table, dict[str, Stype]]:
    if table.dim() != 2:
        raise ValueError(
            f"Expected {path} to be two-dimensional (got {table.dim()}D)"
        )
    if _INSTANCE_ID in table.column_names:
        raise ValueError(
            f"Cannot convert {path} because column {_INSTANCE_ID!r} is "
            "reserved by NIM"
        )
    if _EXAMPLE_ID not in table.column_names:
        raise ValueError(
            f"Expected request-scoped {path} to contain sampler column "
            f"{_EXAMPLE_ID!r}"
        )
    if table.stype(_EXAMPLE_ID) != Stype.id:
        raise ValueError(
            f"Expected column {_EXAMPLE_ID!r} in {path} to have semantic "
            f"type {Stype.id.value!r}"
        )

    arrow = table.to_arrow()
    columns = [
        _INSTANCE_ID if column == _EXAMPLE_ID else column
        for column in arrow.column_names
    ]
    stypes = {
        (_INSTANCE_ID if column == _EXAMPLE_ID else column): stype
        for column, stype in table.stypes.items()
    }
    return arrow.rename_columns(columns), stypes


def _related_table(
    related_tables: RelatedTables,
    name: str,
) -> TableTensor:
    try:
        return related_tables.tables[name]
    except KeyError:
        raise ValueError(
            f"Expected {name!r} to be registered as a related table"
        ) from None


def _logical_join_columns(
    source: TableTensor,
    source_columns: Sequence[str],
    *,
    source_path: str,
    target: TableTensor,
    target_columns: Sequence[str],
    target_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(source_columns) != len(target_columns) or not source_columns:
        raise ValueError(
            "Expected relationship source and target columns to have the "
            "same non-zero length"
        )

    _validate_id_columns(source, source_columns, path=source_path)
    _validate_id_columns(target, target_columns, path=target_path)

    pairs = tuple(zip(source_columns, target_columns))
    if pairs.count((_EXAMPLE_ID, _EXAMPLE_ID)) != 1 or any(
        _EXAMPLE_ID in pair and pair != (_EXAMPLE_ID, _EXAMPLE_ID)
        for pair in pairs
    ):
        raise ValueError(
            f"Expected relationship between {source_path} and {target_path} "
            f"to contain exactly one aligned {_EXAMPLE_ID!r} scope"
        )

    logical_pairs = tuple(
        pair for pair in pairs if pair != (_EXAMPLE_ID, _EXAMPLE_ID)
    )
    if not logical_pairs:
        raise ValueError(
            f"Expected relationship between {source_path} and {target_path} "
            "to contain at least one logical join column"
        )
    return (
        tuple(source for source, _ in logical_pairs),
        tuple(target for _, target in logical_pairs),
    )


def _validate_id_columns(
    table: TableTensor,
    columns: Sequence[str],
    *,
    path: str,
) -> None:
    for column in columns:
        if column not in table.column_names:
            raise ValueError(f"Expected column {column!r} to exist in {path}")
        stype = table.stype(column)
        if stype != Stype.id:
            raise ValueError(
                f"Expected column {column!r} in {path} to have semantic "
                f"type {Stype.id.value!r} (got {stype.value!r})"
            )


def _add_synthetic_key(
    table: pa.Table,
    stypes: dict[str, Stype],
) -> tuple[pa.Table, dict[str, Stype], str]:
    name = _SYNTHETIC_NODE_ID
    suffix = 1
    while name in table.column_names:
        name = f"{_SYNTHETIC_NODE_ID}_{suffix}"
        suffix += 1

    table = table.append_column(
        name,
        pa.array(range(table.num_rows), type=pa.int64()),
    )
    return table, {**stypes, name: Stype.id}, name


def _validate_instance_references(
    instance_config: dict[str, Any],
    table_configs: dict[str, dict[str, Any]],
) -> None:
    instance_dtype = instance_config["schema"]["columns"][_INSTANCE_ID][
        "dtype"
    ]
    known_ids = set(_column_values(instance_config["data"], _INSTANCE_ID))

    for name, config in table_configs.items():
        related_dtype = config["schema"]["columns"][_INSTANCE_ID]["dtype"]
        if related_dtype != instance_dtype:
            raise ValueError(
                f"Expected {_INSTANCE_ID!r} in related table {name!r} to "
                f"have dtype {instance_dtype!r} (got {related_dtype!r})"
            )
        for row, value in enumerate(
            _column_values(config["data"], _INSTANCE_ID)
        ):
            if value not in known_ids:
                raise ValueError(
                    f"Expected {_INSTANCE_ID!r} value {value!r} at related "
                    f"table {name!r} row {row} to reference the task table"
                )


def _validate_join_dtypes(
    instance_config: dict[str, Any],
    table_configs: dict[str, dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> None:
    for relationship in relationships:
        source_name = relationship.get("source_table")
        source_config = (
            instance_config
            if source_name is None
            else table_configs[source_name]
        )
        target_name = relationship["target_table"]
        target_config = table_configs[target_name]
        for source_column, target_column in zip(
            relationship["source_columns"],
            relationship["target_columns"],
        ):
            source_dtype = source_config["schema"]["columns"][source_column][
                "dtype"
            ]
            target_dtype = target_config["schema"]["columns"][target_column][
                "dtype"
            ]
            if source_dtype != target_dtype:
                source_table = source_name or _INSTANCE_TABLE
                raise ValueError(
                    f"Expected relationship columns "
                    f"{source_table!r}.{source_column} and "
                    f"{target_name!r}.{target_column} to have the same NIM "
                    f"dtype (got {source_dtype!r} and {target_dtype!r})"
                )


def _validate_task_roots(
    instance_config: dict[str, Any],
    table_configs: dict[str, dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> None:
    instance_data = instance_config["data"]
    for relationship in relationships:
        target_name = relationship["target_table"]
        target_data = table_configs[target_name]["data"]
        target_keys = _row_keys(
            target_data,
            (_INSTANCE_ID, *relationship["target_columns"]),
        )
        target_counts: dict[tuple[Any, ...], int] = {}
        for key in target_keys:
            target_counts[key] = target_counts.get(key, 0) + 1

        source_keys = _row_keys(
            instance_data,
            (_INSTANCE_ID, *relationship["source_columns"]),
        )
        for row, key in enumerate(source_keys):
            count = target_counts.get(key, 0)
            if count != 1:
                raise ValueError(
                    f"Expected task table row {row} to match exactly one "
                    f"same-instance row in related table {target_name!r} "
                    f"(found {count})"
                )


def _row_keys(
    data: pa.Table,
    columns: Sequence[str],
) -> list[tuple[Any, ...]]:
    return list(zip(*(_column_values(data, column) for column in columns)))


def _column_values(data: pa.Table, column: str) -> list[Any]:
    return data[column].to_pylist()
