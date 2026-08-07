from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any, Self, cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.relational.join import LEFT_ROW_ID, RIGHT_ROW_ID, join_index
from sdm.tensor.mixin import DeviceMixin

if TYPE_CHECKING:
    import graphviz

    from sdm.relational import RelationalSampler, TemporalSamplingConfig


@dataclass(frozen=True, repr=False)
class Relationship:
    r"""Join relationship between two tables.

    Args:
        left_table: Name of the left table.
        left_columns: Column names from the left table.
        right_table: Name of the right table.
        right_columns: Column names from the right table.
    """

    left_table: str
    left_columns: tuple[str, ...]
    right_table: str
    right_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.left_columns) != len(self.right_columns):
            raise ValueError(
                f"Expected 'left_columns' and 'right_columns' to have the "
                f"same length (got {len(self.left_columns)} and "
                f"{len(self.right_columns)})"
            )

        if len(self.left_columns) == 0:
            raise ValueError(
                "Expected 'left_columns' and 'right_columns' to be non-empty"
            )

        for column in (*self.left_columns, *self.right_columns):
            for reserved in (LEFT_ROW_ID, RIGHT_ROW_ID):
                if column == reserved:
                    raise ValueError(
                        f"Column name {column!r} is reserved for internal "
                        f"row indexing"
                    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str | Sequence[str]]) -> Self:
        r"""Create a :class:`Relationship` from a mapping.

        Args:
            mapping: The mapping with ``"left_table"``, ``"left_columns"``,
                ``"right_table"``, ``"right_columns"`` entries.
        """
        left_table = mapping["left_table"]
        assert isinstance(left_table, str)

        if "left_column" in mapping:
            left_columns = mapping["left_column"]
        else:
            left_columns = mapping["left_columns"]
        if isinstance(left_columns, str):
            left_columns = (left_columns,)

        right_table = mapping["right_table"]
        assert isinstance(right_table, str)

        if "right_column" in mapping:
            right_columns = mapping["right_column"]
        else:
            right_columns = mapping["right_columns"]
        if isinstance(right_columns, str):
            right_columns = (right_columns,)

        return cls(
            left_table=left_table,
            left_columns=tuple(left_columns),
            right_table=right_table,
            right_columns=tuple(right_columns),
        )

    def _left_columns_repr(self) -> str:
        if len(self.left_columns) == 1:
            return f"{self.left_table}.{self.left_columns[0]}"
        return f"{self.left_table}.[{','.join(self.left_columns)}]"

    def _right_columns_repr(self) -> str:
        if len(self.right_columns) == 1:
            return f"{self.right_table}.{self.right_columns[0]}"
        return f"{self.right_table}.[{','.join(self.right_columns)}]"

    def __repr__(self) -> str:
        return f"{self._left_columns_repr()} <> {self._right_columns_repr()}"


@dataclass(frozen=True, init=False, repr=False)
class RelationalData(DeviceMixin):
    r"""Collection of named tables and join relationships.

    .. testcode::

        from sdm import RelationalData, TableTensor

        data = RelationalData(
            tables={
                "users": TableTensor.from_columns(
                    {"user_id": [0, 1]},
                    stypes={"user_id": "id"},
                ),
                "orders": TableTensor.from_columns(
                    {
                        "user_id": [0, 1],
                        "item_id": [10, 11],
                    },
                    stypes={
                        "user_id": "id",
                        "item_id": "id",
                    },
                ),
                "items": TableTensor.from_columns(
                    {"item_id": [10, 11]},
                    stypes={"item_id": "id"},
                ),
            },
            relationships=[
                # Foreign key from orders to users:
                dict(left_table="orders", left_column="user_id", right_table="users", right_column="user_id"),
                # Foreign key from orders to items:
                dict(left_table="orders", left_column="item_id", right_table="items", right_column="item_id"),
            ],
        )

    Args:
        tables: Tables keyed by table name.
        relationships: Join relationships among ``tables``.
    """  # noqa: E501

    tables: Mapping[str, TableTensor]
    relationships: tuple[Relationship, ...]

    def __init__(
        self,
        tables: Mapping[str, TableTensor],
        relationships: Collection[
            Relationship | Mapping[str, str | Sequence[str]]
        ],
    ) -> None:

        relationships = tuple(
            relationship
            if isinstance(relationship, Relationship)
            else Relationship.from_mapping(relationship)
            for relationship in relationships
        )

        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "relationships", relationships)
        self.__post_init__()

    def __post_init__(self) -> None:
        for table in self.tables.values():
            if table.dim() != 2:
                raise ValueError("Tables need to be two-dimensional")

        for relationship in self.relationships:
            for table, columns in (
                (relationship.left_table, relationship.left_columns),
                (relationship.right_table, relationship.right_columns),
            ):
                if table not in self.tables:
                    raise ValueError(
                        f"Expected {table!r} to be registered as a table"
                    )

                for column in columns:
                    stype = self.tables[table].stype(column)
                    if stype != Stype.id:
                        raise ValueError(
                            f"Expected column {column!r} in table {table!r} "
                            f"to have semantic type {Stype.id!r} "
                            f"(got {stype!r})"
                        )

    def to(self, device: torch.device | str | None) -> Self:
        r""":meta private:"""  # noqa: D415
        return self.__class__(
            tables={
                table_name: cast(TableTensor, table.to(device))
                for table_name, table in self.tables.items()
            },
            relationships=self.relationships,
        )

    @property
    def device(self) -> torch.device:
        r""":meta private:"""  # noqa: D415
        devices = {table.device for table in self.tables.values()}
        if len(devices) == 0:
            raise RuntimeError(
                f"Could not determine 'device' of empty "
                f"{self.__class__.__name__!r}"
            )
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected tables in {self.__class__.__name__!r} to be on "
                f"the same device (got {list(devices)})"
            )
        return next(iter(devices))

    def edge_indices(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, ...]:
        r"""Materialize heterogeneous graph edges for table relationships.

        Args:
            dtype: The dtype.
            device: The device.

        Returns:
            The edge indices for each relationship in order.
            Each edge index has shape ``[2, num_edges]`` and stores left table
            indices in the first row and right table indices in the second row.
        """
        edge_indices: list[Tensor] = []
        for rel in self.relationships:
            src, dst = join_index(
                left_table=self.tables[rel.left_table],
                right_table=self.tables[rel.right_table],
                left_keys=rel.left_columns,
                right_keys=rel.right_columns,
                how="inner",
                dtype=dtype,
                device=device,
            )
            edge_indices.append(torch.stack([src, dst], dim=0))

        return tuple(edge_indices)

    def sampler(
        self,
        temporal: TemporalSamplingConfig | dict[str, Any] | None = None,
    ) -> RelationalSampler:
        r"""Create a device-appropriate sampler over this relational data.

        .. testcode::

            from sdm import (
                RelationalData,
                TableTensor,
                TemporalSamplingConfig,
            )

            data = RelationalData(
                tables={
                    "users": TableTensor.from_columns(
                        {"user_id": [0, 1]},
                        stypes={"user_id": "id"},
                    ),
                    "orders": TableTensor.from_columns(
                        {
                            "user_id": [0, 1],
                            "item_id": [10, 11],
                            "order_date": ["2026-01-01", "2026-01-02"],
                        },
                        stypes={
                            "user_id": "id",
                            "item_id": "id",
                            "order_date": "datetime",
                        },
                    ),
                    "items": TableTensor.from_columns(
                        {"item_id": [10, 11]},
                        stypes={"item_id": "id"},
                    ),
                },
                relationships=[
                    # Foreign key from orders to users:
                    dict(left_table="orders", left_column="user_id", right_table="users", right_column="user_id"),
                    # Foreign key from orders to items:
                    dict(left_table="orders", left_column="item_id", right_table="items", right_column="item_id"),
                ],
            )

            sampler = data.sampler(
                temporal=TemporalSamplingConfig(
                    time_columns={"orders": "order_date"},
                    strategy="last",
                ),
            )

        Args:
            temporal: Temporal sampling configuration or a dictionary of its
                constructor arguments. A row in a time-aware table can only
                be sampled if its timestamp does not exceed the query
                timestamp.
        """  # noqa: E501
        from sdm.relational.sampler import (  # noqa: PLC0415
            TemporalSamplingConfig,
        )

        if temporal is not None and not isinstance(
            temporal, TemporalSamplingConfig
        ):
            temporal = TemporalSamplingConfig(**temporal)

        if self.device.type == "cuda":
            from sdm.relational.cugraph_sampler import (  # noqa: PLC0415
                CuGraphRelationalSampler,
            )

            return CuGraphRelationalSampler(
                data=self,
                temporal=temporal,
            )

        # Avoid a circular import through `sdm.relational`.
        from sdm.relational import RelationalSampler  # noqa: PLC0415

        return RelationalSampler(
            data=self,
            temporal=temporal,
        )

    def to_graphviz(
        self,
        *,
        hide_columns: bool = False,
        **kwargs: Any,
    ) -> graphviz.Graph:
        r"""Return a graph visualization of the relational schema.

        Args:
            hide_columns: Whether to hide column name descriptions.
            **kwargs: Additional keyword arguments passed to
                :class:`graphviz.Graph`.
        """
        import graphviz

        def left_align(keys: list[str]) -> str:
            if len(keys) == 0:
                return ""
            return "\\l".join(keys) + "\\l"

        graph = graphviz.Graph(**kwargs)

        for table_name, table in self.tables.items():
            if hide_columns:
                label = f"{{{table_name}}}"
            else:
                columns = [
                    f"{column}: {stype}"
                    for stype, columns in table._columns.items()
                    for column in columns
                ]
                label = f"{{{table_name}|{left_align(columns)}}}"
            graph.node(table_name, shape="record", label=label)

        for rel in self.relationships:
            label = "\\n".join(
                f" {left_column} <> {right_column} "
                for left_column, right_column in zip(
                    rel.left_columns, rel.right_columns
                )
            )
            graph.edge(
                rel.left_table,
                rel.right_table,
                label=label,
                fontsize="11pt",
            )

        return graph

    def __repr__(self) -> str:
        out = f"{self.__class__.__name__}(\n"
        if len(self.tables) > 0:
            out += "  tables={\n"
            out += "".join(
                f"    {name}: {table.__repr__(indent=4)[4:]},\n"
                for name, table in self.tables.items()
            )
            out += "  },\n"
        else:
            out += "  tables={},\n"
        if len(self.relationships) > 0:
            out += "  relationships=[\n"
            out += "".join(f"    {rel},\n" for rel in self.relationships)
            out += "  ],\n"
        else:
            out += "  relationships=[],\n"
        out += ")"
        return out

    def _repr_html_(self) -> str:
        import pandas as pd

        rows = [
            [
                name,
                table.size(-2),
                table.size(-1),
                ", ".join(
                    str(stype)
                    for stype, tensor in table.items()
                    if tensor.size(-1) > 0
                ),
            ]
            for name, table in self.tables.items()
        ]
        df = pd.DataFrame(
            rows,
            columns=pd.Index(["Table", "Rows", "Columns", "Stypes"]),
        )

        ul = "".join(
            f"<li>"
            f"<code>{escape(rel._left_columns_repr())}</code>"
            f" ↔️ "
            f"<code>{escape(rel._right_columns_repr())}</code>"
            f"</li>"
            for rel in self.relationships
        )

        return df.to_html(index=False, escape=True) + f"<ul>{ul}</ul>"
