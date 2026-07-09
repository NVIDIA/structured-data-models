from collections.abc import Mapping, Sequence

from sdm import TableTensor
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
        self.time_columns = time_columns  # TODO Check for existence.

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

        # TODO Check for existence of time_column
        # TODO Implement sampling:

        return RelatedTables(
            tables=self.data.tables,
            relationships=self.data.relationships,
            task_links=(task_link,),
        )
