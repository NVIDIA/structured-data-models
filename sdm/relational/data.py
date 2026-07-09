from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import Self

from sdm import StringTensor, Stype, TableTensor
from sdm.tensor.mixin import DeviceMixin

PREFIX = "sdm_internal"
ROW_ID = f"__{PREFIX}_row_id__"
LEFT_ROW_ID = f"__{PREFIX}_left_row_id__"
RIGHT_ROW_ID = f"__{PREFIX}_right_row_id__"

if TYPE_CHECKING:
    import cudf

    from sdm.relational import RelationalSampler


def _to_cudf_series(column: Tensor) -> cudf.Series:
    try:
        import cudf
        import pylibcudf as plc
    except ImportError as exc:
        raise ImportError(
            "CUDA-resident relational joins require cuDF"
        ) from exc

    if isinstance(column, StringTensor):
        if not column.is_contiguous():
            column = column.contiguous()
            assert isinstance(column, StringTensor)

        offset_column = plc.Column.from_array(  # ty: ignore[missing-argument]
            obj=column._offset
        )
        plc_column = plc.Column(
            data_type=plc.DataType(plc.TypeId.STRING),
            size=column.numel(),
            data=plc.gpumemoryview(column._data),
            mask=None,
            null_count=0,
            offset=int(column.storage_offset()),
            children=[offset_column],
        )
    else:
        column = column.detach().contiguous().view(-1)
        plc_column = plc.Column.from_array(  # ty: ignore[missing-argument]
            obj=column
        )

    return cudf.Series.from_pylibcudf(plc_column)


def _to_cudf(
    table: TableTensor,
    columns: Sequence[str],
) -> cudf.DataFrame:
    try:
        import cudf
    except ImportError as exc:
        raise ImportError(
            "CUDA-resident relational joins require cuDF"
        ) from exc

    selected = table[..., columns]
    return cudf.DataFrame(
        {
            name: _to_cudf_series(column)
            for name, column in zip(
                selected.columns[Stype.id],
                selected.id.unbind(-1),
            )
        }
    )


@dataclass(frozen=True)
class Relationship:
    r"""Join relationship between two tables.

    Args:
        left_table: Name of the left table.
        left_columns: Column names from the left table.
        right_table: Name of the right table.
        right_columns: Column names from the right table.
    """

    left_table: str
    left_columns: Sequence[str]
    right_table: str
    right_columns: Sequence[str]

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
            for reserved in (ROW_ID, LEFT_ROW_ID, RIGHT_ROW_ID):
                if column == reserved:
                    raise ValueError(
                        f"Column name '{column}' is reserved for internal "
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
            left_columns=left_columns,
            right_table=right_table,
            right_columns=right_columns,
        )


@dataclass(frozen=True, init=False)
class RelationalData(DeviceMixin):
    r"""Collection of named tables and join relationships.

    .. code-block:: python

        from sdm import RelationalData, TableTensor

        data = RelationalData(
            tables={
                "users": TableTensor.from_pandas(...),
                "orders": TableTensor.from_pandas(...),
                "items": TableTensor.from_pandas(...),
            },
            relationships=[
                # Foreign key from orders to users:
                dict(left_table="orders", left_column="user_id",
                     right_table="users", right_column="user_id"),
                # Foreign key from orders to items:
                dict(left_table="orders", left_column="item_id",
                     right_table="items", right_column="item_id"),
            ],
        )

    Args:
        tables: Tables keyed by table name.
        relationships: Join relationships among ``tables``.
    """

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
                        f"Expected '{table}' to be registered as a table"
                    )

                for column in columns:
                    stype = self.tables[table].stype(column)
                    if stype != Stype.id:
                        raise ValueError(
                            f"Expected column '{column}' in table '{table}' "
                            f"to have semantic type '{Stype.id.value}' "
                            f"(got '{stype.value}')"
                        )

    def to(self, device: torch.device | str | None) -> Self:  # noqa: D102
        return self.__class__(
            tables={
                table_name: cast(TableTensor, table.to(device))
                for table_name, table in self.tables.items()
            },
            relationships=self.relationships,
        )

    @property
    def device(self) -> torch.device:  # noqa: D102
        devices = {table.device for table in self.tables.values()}
        if len(devices) == 0:
            raise RuntimeError(
                f"Could not determine 'device' of empty "
                f"'{self.__class__.__name__}'"
            )
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected tables in '{self.__class__.__name__}' to be on "
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
            dtype: The edge index dtype.
            device: The output device. If ``None``, edges stay on the device
                of the participating tables.

        Returns:
            The edge indices for each relationship in order.
            Each edge index has shape ``[2, num_edges]`` and stores left table
            indices in the first row and right table indices in the second row.

        Raises:
            ValueError: If participating tables are not on the same device.
            ImportError: If CUDA tables are used without cuDF installed.
        """
        device = self.device if device is None else device

        columns: dict[str, list[str]] = defaultdict(list)
        for rel in self.relationships:
            for table, rel_columns in (
                (rel.left_table, rel.left_columns),
                (rel.right_table, rel.right_columns),
            ):
                for column in rel_columns:
                    if column not in columns[table]:
                        columns[table].append(column)

        devices = {self.tables[name].device for name in columns}
        if len(devices) > 1:
            devices_repr = ", ".join(
                str(device) for device in sorted(devices, key=str)
            )
            raise ValueError(
                "Expected all tables participating in relationships to be "
                f"on the same device (got {devices_repr})"
            )
        if len(devices) == 0:
            return ()

        execution_device = next(iter(devices))
        if execution_device.type == "cuda":
            with torch.cuda.device(execution_device):
                return self._edge_indices_cudf(
                    columns=columns,
                    dtype=dtype,
                    device=device,
                )

        tables = {
            name: table[..., columns[name]].to_arrow()
            for name, table in self.tables.items()
            if name in columns
        }

        tables = {
            name: table.append_column(
                ROW_ID,
                pa.array(torch.arange(table.num_rows, dtype=dtype).numpy()),
            )
            for name, table in tables.items()
        }

        edge_indices: list[Tensor] = []
        for rel in self.relationships:
            left = tables[rel.left_table]
            left = left.select((*rel.left_columns, ROW_ID))
            left = left.rename_columns({ROW_ID: LEFT_ROW_ID})
            right = tables[rel.right_table]
            right = right.select((*rel.right_columns, ROW_ID))
            right = right.rename_columns({ROW_ID: RIGHT_ROW_ID})

            joined = left.join(
                right,
                keys=list(rel.left_columns),
                right_keys=list(rel.right_columns),
                join_type="inner",
            )

            src = torch.from_numpy(joined[LEFT_ROW_ID].to_numpy()).to(device)
            dst = torch.from_numpy(joined[RIGHT_ROW_ID].to_numpy()).to(device)
            edge_indices.append(torch.stack([src, dst], dim=0))

        return tuple(edge_indices)

    def _edge_indices_cudf(
        self,
        columns: Mapping[str, Sequence[str]],
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> tuple[Tensor, ...]:
        tables = {
            name: _to_cudf(table=table, columns=columns[name])
            for name, table in self.tables.items()
            if name in columns
        }
        for name, table in tables.items():
            row_id = torch.arange(
                len(table),
                dtype=dtype,
                device=self.tables[name].device,
            )
            table[ROW_ID] = _to_cudf_series(row_id)

        edge_indices: list[Tensor] = []
        for rel in self.relationships:
            left = tables[rel.left_table][[*rel.left_columns, ROW_ID]].rename(
                columns={ROW_ID: LEFT_ROW_ID}
            )
            right = tables[rel.right_table][
                [*rel.right_columns, ROW_ID]
            ].rename(columns={ROW_ID: RIGHT_ROW_ID})

            joined = left.merge(
                right,
                left_on=list(rel.left_columns),
                right_on=list(rel.right_columns),
                how="inner",
            )

            src = torch.as_tensor(joined[LEFT_ROW_ID])
            dst = torch.as_tensor(joined[RIGHT_ROW_ID])
            edge_indices.append(
                torch.stack([src, dst], dim=0).to(device=device)
            )

        return tuple(edge_indices)

    def sampler(
        self,
        time_columns: Mapping[str, str] | None = None,
    ) -> RelationalSampler:
        r"""Create a subgraph sampler over this relational data.

        .. code-block:: python

            from sdm import RelationalData, TableTensor

            data = RelationalData(
                tables={
                    "users": TableTensor.from_pandas(...),
                    "orders": TableTensor.from_pandas(...),
                    "items": TableTensor.from_pandas(...),
                },
                relationships=[
                    # Foreign key from orders to users:
                    dict(left_table="orders", left_column="user_id",
                         right_table="users", right_column="user_id"),
                    # Foreign key from orders to items:
                    dict(left_table="orders", left_column="item_id",
                         right_table="items", right_column="item_id"),
                ],
            )

            sampler = data.sampler(
                time_columns={"orders": "order_date"},
            )

        Args:
            time_columns: Mapping from table name to the datetime column used
                for temporal sampling. A row in a time-aware table can only be
                sampled if its timestamp does not exceed the query timestamp.
        """
        from sdm.relational import RelationalSampler

        return RelationalSampler(
            data=self,
            time_columns=time_columns,
        )
