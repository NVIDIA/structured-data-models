# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import torch

from sdm import Stype, StypeLike, TableTensor
from sdm.relational import (
    RelatedTables,
    RelationalData,
    RelationalSamplerOutput,
    Relationship,
    TaskLink,
)

_ROW = "__sdm_disk_row__"
_EXAMPLE = "__example__"


class ParquetRelationalSampler:
    """Sample relational Parquet tables while keeping feature columns on disk.

    The relationship keys and time columns are held in CPU memory and sampled
    by the same PyG backend as :class:`RelationalSampler`. Polars reads only
    sampled feature rows from Parquet. The task table must be two-dimensional.
    Source row order must match the order used to create any in-memory
    :class:`RelationalData` being compared.

    Args:
        tables: Parquet file path for each table.
        stypes: Semantic types to load from each table. Include every key used
            by ``relationships`` and the task link as ``"id"``.
        relationships: Joins between the source tables.
        time_columns: Datetime column used to constrain sampling by time for
            each time-aware table.
    """

    def __init__(
        self,
        tables: Mapping[str, str | Path],
        stypes: Mapping[str, Mapping[str, StypeLike]],
        relationships: Collection[
            Relationship | Mapping[str, str | Sequence[str]]
        ],
        time_columns: Mapping[str, str] | None = None,
    ) -> None:
        try:
            import polars as pl  # noqa: PLC0415
        except ImportError as error:
            raise ImportError(
                "ParquetRelationalSampler requires 'polars'"
            ) from error

        self._pl = pl
        self.paths = {name: Path(path) for name, path in tables.items()}
        self.stypes = {
            name: {column: Stype(stype) for column, stype in schema.items()}
            for name, schema in stypes.items()
        }
        self.time_columns = dict(time_columns or {})
        self._categories: dict[str, dict[str, list[object]]] = {}

        graph_tables = {}
        for name, path in self.paths.items():
            schema = self.stypes[name]
            self._categories[name] = {}
            for column, stype in schema.items():
                if stype != Stype.categorical:
                    continue
                categories = (
                    pl.scan_parquet(path)
                    .select(column)
                    .unique(maintain_order=True)
                    .filter(pl.col(column).is_not_null())
                    .collect(engine="streaming")
                )
                self._categories[name][column] = categories[column].to_list()
            columns = [
                column
                for column, stype in schema.items()
                if stype == Stype.id or column == self.time_columns.get(name)
            ]
            if (
                _ROW in schema
                or _ROW in pl.scan_parquet(path).collect_schema()
            ):
                raise ValueError(f"Column {_ROW!r} is reserved")
            frame = pl.scan_parquet(path, row_index_name=_ROW).select(
                pl.col(_ROW).cast(pl.Int64), *columns
            )
            graph_tables[name] = TableTensor.from_pandas(
                df=frame.collect(engine="streaming").to_pandas(),
                stypes={
                    _ROW: Stype.id,
                    **{column: schema[column] for column in columns},
                },
            )

        self.data = RelationalData(
            tables=graph_tables,
            relationships=relationships,
        )
        self._sampler = self.data.sampler(time_columns=self.time_columns)

    def __call__(
        self,
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        num_neighbors: Sequence[int],
        task_time_column: str | None = None,
        temporal_strategy: Literal["last", "uniform"] = "last",
    ) -> RelationalSamplerOutput:
        """Alias of :meth:`sample`."""
        return self.sample(
            task_table=task_table,
            task_link=task_link,
            num_neighbors=num_neighbors,
            task_time_column=task_time_column,
            temporal_strategy=temporal_strategy,
        )

    def sample(
        self,
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        num_neighbors: Sequence[int],
        task_time_column: str | None = None,
        temporal_strategy: Literal["last", "uniform"] = "last",
    ) -> RelationalSamplerOutput:
        """Return sampled task and related tables for model input.

        Args:
            task_table: Task rows with entity IDs and optional timestamps.
            task_link: Link from task rows to a source table.
            num_neighbors: Number of neighbors per relationship at each hop.
            task_time_column: Datetime column in ``task_table``.
            temporal_strategy: ``"last"`` or ``"uniform"`` sampling.

        Returns:
            The same output type as :meth:`RelationalSampler.sample`.
        """
        if task_table.dim() != 2:
            raise ValueError("Task table needs to be two-dimensional")

        sampled = self._sampler.sample(
            task_table=task_table,
            task_link=task_link,
            num_neighbors=num_neighbors,
            task_time_column=task_time_column,
            temporal_strategy=temporal_strategy,
        )
        import pandas as pd

        tables: dict[str, TableTensor] = {}
        for name, graph_table in sampled.related_tables.tables.items():
            assert isinstance(graph_table, TableTensor)
            index = graph_table[_ROW].id[..., 0].numpy()
            columns = list(self.stypes[name])
            frame = (
                self._pl.scan_parquet(self.paths[name], row_index_name=_ROW)
                .filter(self._pl.col(_ROW).is_in(index.tolist()))
                .select(_ROW, *columns)
                .collect(engine="streaming")
                .to_pandas()
                .set_index(_ROW)
                .loc[index]
                .reset_index(drop=True)
            )
            for column, categories in self._categories[name].items():
                frame[column] = pd.Categorical(
                    frame[column], categories=categories
                )
            hydrated = TableTensor.from_pandas(
                df=frame,
                stypes=self.stypes[name],
            )
            tables[name] = cast(
                TableTensor,
                torch.cat([hydrated, graph_table[_EXAMPLE]], dim=-1),
            )

        return RelationalSamplerOutput(
            task_table=sampled.task_table,
            related_tables=cast(
                RelatedTables[TableTensor],
                sampled.related_tables.replace_tables(tables),
            ),
        )
