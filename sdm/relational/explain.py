from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import Tensor

from sdm.relational.data import Relationship
from sdm.relational.task import RelatedTables, TaskLink
from sdm.tensor.table import TableTensor


@dataclass(frozen=True)
class RelationalFeatureRef:
    r"""Attribution destination for one encoded feature channel.

    Args:
        table: Source related-table name, or ``None`` for the task input.
        column: Source column name.
        row_index: Optional source-to-encoded row mapping with shape
            ``[R_source]``. For source row ``i``, ``row_index[i]`` is its
            encoded row. When omitted, encoded and source rows have equal
            size and positional alignment.
    """

    table: str | None
    column: str
    row_index: Tensor | None = None


@dataclass(frozen=True)
class RelationalExplanationTopology:
    r"""Input-row-aligned topology for one context or query execution.

    All row-index tensors refer to the rows of the corresponding source
    tables. ``relationships`` and ``edges`` are positionally aligned. Each
    edge pair indexes the left and right tables of its relationship and
    contains equally sized row-index tensors.
    ``task_row_by_table`` and ``hop_by_table`` contain one entry per source
    table row and use ``-1`` for rows outside the sampled task subgraphs.

    Args:
        table_names: Related tables represented by this topology.
        relationships: Relationships used by the model.
        edges: Left and right source-row index tensors with shape ``[E_i]``
            for each relationship.
        task_link: Task link used to identify the readout rows.
        readout_table: Table containing the task entities.
        readout_index: Valid readout-table row for each task row, with shape
            ``[R_task]``.
        task_row_by_table: Owning task row for every related-table row, with
            shape ``[R_table]``, or ``-1`` when that row is outside a sampled
            task subgraph.
        hop_by_table: Hop from the task entity for every related-table row,
            with shape ``[R_table]``, or ``-1`` when that row is outside a
            sampled task subgraph.
    """

    table_names: tuple[str, ...]
    relationships: tuple[Relationship, ...]
    edges: tuple[tuple[Tensor, Tensor], ...]
    task_link: TaskLink
    readout_table: str
    readout_index: Tensor
    task_row_by_table: Mapping[str, Tensor]
    hop_by_table: Mapping[str, Tensor]


class _Recorder:
    def __init__(
        self,
        *,
        x_context: TableTensor | None = None,
        related_context_tables: RelatedTables | None = None,
    ) -> None:
        self.x_context = x_context
        self.related_context_tables = related_context_tables
        self.context_topology: RelationalExplanationTopology | None = None
        self.query_topology: RelationalExplanationTopology | None = None
        self.features: list[Tensor] = []
        self.refs: list[tuple[RelationalFeatureRef, ...]] = []

    def record_context(
        self,
        topology: RelationalExplanationTopology,
    ) -> None:
        self.context_topology = topology

    def record_query(
        self,
        topology: RelationalExplanationTopology,
    ) -> None:
        self.query_topology = topology

    def record_query_features(
        self,
        *,
        x: Tensor,
        refs: Sequence[RelationalFeatureRef],
    ) -> Tensor:
        assert x.size(-1) == len(refs)

        x = x.detach().requires_grad_(True)
        self.features.append(x)
        self.refs.append(tuple(refs))
        return x
