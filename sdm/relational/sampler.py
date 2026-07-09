from collections.abc import Mapping, Sequence

from torch import Tensor

from sdm import Stype, TableTensor
from sdm.relational import RelatedTables, RelationalData, TaskLink


class RelationalSampler:
    r"""Subgraph sampler over relational data.

    Args:
        data: The collection of named tables and their relationships.
        time_columns: Mapping from table name to the datetime column used for
            temporal sampling. A row in a time-aware table can only be sampled
            if its timestamp does not exceed the query timestamp.
    """

    def __init__(
        self,
        data: RelationalData,
        time_columns: Mapping[str, str] | None = None,
    ) -> None:
        self.data = data
        self.time_columns = time_columns or {}

        for table_name, column_name in self.time_columns.items():
            stype = self.data.tables[table_name].stype(column_name)
            if stype != Stype.datetime:
                raise ValueError(
                    f"Expected '{column_name}' in table '{table_name}' to "
                    f"have semantic type '{Stype.datetime.value}' "
                    f"(got '{stype.value})"
                )

        self._colptr_dict: dict[str, Tensor] = {}
        self._row_dict: dict[str, Tensor] = {}
        self._time_dict: dict[str, Tensor] = {}

        for table_name, time_column in self.time_columns.items():
            time = self.data.tables[table_name][time_column].squeeze(-1)
            self._time_dict[table_name] = time.contiguous().cpu()

        for relationship, edge_index in zip(
            self.data.relationships,
            self.data.edge_indices(),
        ):
            # Sort by time if available
            # index2ptr
            # Do the reverse connection as well.

    def __call__(
        self,
        num_neighbors: Sequence[int],
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        task_time_column: str | None = None,
    ) -> RelatedTables:
        r"""Alias of :meth:`sample`."""
        return self.sample(
            num_neighbors=num_neighbors,
            task_table=task_table,
            task_link=task_link,
            task_time_column=task_time_column,
        )

    def sample(
        self,
        num_neighbors: Sequence[int],
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        task_time_column: str | None = None,
    ) -> RelatedTables:
        r"""Sample :class:`RelatedTables` for task rows.

        Args:
            num_neighbors: Number of neighbors to sample per hop.
            task_table: Task table whose rows define the sampling queries.
            task_link: Link from ``task_table`` rows to a table in the
                relational data.
            task_time_column: Datetime column in ``task_table`` used as the
                query timestamp for temporal sampling.

        """
        if not isinstance(task_link, TaskLink):
            task_link = TaskLink.from_mapping(task_link)

        if task_time_column is not None:
            stype = task_table.stype(task_time_column)
            if stype != Stype.datetime:
                raise ValueError(
                    f"Expected task time column to have semantic type "
                    f"'{Stype.datetime.value}' (got '{stype.value})"
                )

        for table, columns in (
            (task_table, task_link.task_columns),
            (self.data.tables[task_link.table], task_link.table_columns),
        ):
            for column in columns:
                stype = table.stype(column)
                if stype != Stype.id:
                    raise ValueError(
                        f"Expected column '{column}' to have semantic type "
                        f"'{Stype.id.value}' (got '{stype.value}')"
                    )

        # TODO Implement sampling.

        return RelatedTables(
            tables=self.data.tables,
            relationships=self.data.relationships,
            task_links=(task_link,),
        )
