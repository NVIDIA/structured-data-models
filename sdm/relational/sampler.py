from collections.abc import Mapping

from sdm.relational import RelationalData


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
        pass
