"""Relational input encoding for KumoRFM."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.models.kumorfm.graph import (
    _HomogeneousGraph,
    _make_homogeneous_graph,
)
from sdm.models.tabiclv2.row_embedding import RowEmbedding


@dataclass(frozen=True)
class _RelationalEncoding:
    """Encoded rows and their combined relational layout."""

    x_dict: dict[str, Tensor]
    edge_index_dict: dict[tuple[str, str, str], Tensor]
    root_index: Tensor
    entity_table: str


class RelationalEncoder(torch.nn.Module):
    r"""Build and encode the relational input to KumoRFM.

    Context and query relationships are materialized independently, then query
    rows and edges are offset into a joint layout. Context targets are diffused
    from task-linked entity roots to all reachable context rows before each
    table is encoded exactly once.

    Args:
        row_embedding: Generic TabICLv2 row embedding shared by all tables.
    """

    def __init__(self, row_embedding: RowEmbedding) -> None:
        super().__init__()
        self.row_embedding = row_embedding

    def forward(
        self,
        x_context: TableTensor,
        y_context: Tensor,
        x_query: TableTensor,
        related_context_tables: RelatedTables,
        related_query_tables: RelatedTables,
        *,
        num_hops: int,
        max_keys: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _RelationalEncoding:
        r"""Return encoded rows and their induced heterogeneous graph."""
        if num_hops < 0:
            raise ValueError("'num_hops' must be non-negative")

        parameter = self.row_embedding.lin.weight
        device, dtype = parameter.device, parameter.dtype
        y_context = y_context.to(device=device)
        entity_table = related_context_tables.task_links[0].table

        context_edges, context_task_edges = (
            related_context_tables.edge_indices(
                x_context, dtype=torch.long, device=device
            )
        )
        query_edges, query_task_edges = related_query_tables.edge_indices(
            x_query, dtype=torch.long, device=device
        )

        context_rows = {
            name: table.size(0)
            for name, table in related_context_tables.tables.items()
        }
        feature_dict: dict[str, Tensor] = {}
        for name, context_table in related_context_tables.tables.items():
            features = [context_table.numerical.to(device=device, dtype=dtype)]
            if name in related_query_tables.tables:
                features.append(
                    related_query_tables.tables[name].numerical.to(
                        device=device, dtype=dtype
                    )
                )
            feature_dict[name] = torch.cat(features)

        context_root = _ordered_roots(
            context_task_edges[0], num_rows=x_context.size(0)
        )
        query_root = (
            _ordered_roots(query_task_edges[0], num_rows=x_query.size(0))
            + context_rows[entity_table]
        )
        root_index = torch.cat((context_root, query_root))

        task_features = torch.cat(
            (
                x_context.numerical.to(device=device, dtype=dtype),
                x_query.numerical.to(device=device, dtype=dtype),
            )
        )
        scattered = task_features.new_zeros(
            (feature_dict[entity_table].size(0), task_features.size(-1))
        ).index_copy(0, root_index, task_features)
        feature_dict[entity_table] = torch.cat(
            (feature_dict[entity_table], scattered), dim=-1
        )

        query_edges_by_relationship = dict(
            zip(related_query_tables.relationships, query_edges)
        )

        edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
        for index, (relationship, context_edge) in enumerate(
            zip(related_context_tables.relationships, context_edges)
        ):
            src, dst = relationship.left_table, relationship.right_table
            edges = [context_edge]
            if relationship in query_edges_by_relationship:
                query_edge = query_edges_by_relationship[relationship]
                query_edge = query_edge + query_edge.new_tensor(
                    [[context_rows[src]], [context_rows[dst]]]
                )
                edges.append(query_edge)
            edge_index_dict[(src, str(index), dst)] = torch.cat(edges, dim=1)

        graph = _make_homogeneous_graph(feature_dict, edge_index_dict)
        entity_offset = graph.offset_dict[entity_table][0]
        global_root_index = root_index + entity_offset
        num_nodes = graph.colptr.numel() - 1
        labels, reachable = _propagate_targets(
            y_context,
            global_root_index[: x_context.size(0)],
            graph=graph,
            num_hops=num_hops,
            num_nodes=num_nodes,
            dtype=dtype,
        )

        context_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        for name, num_rows in context_rows.items():
            start = graph.offset_dict[name][0]
            context_mask[start : start + num_rows] = True
        train_mask = context_mask & reachable

        x_dict: dict[str, Tensor] = {}
        for name, features in feature_dict.items():
            start, end = graph.offset_dict[name]
            table_train_mask = train_mask[start:end]
            table_labels = labels[start:end][table_train_mask]
            if features.size(-1) == 0:
                features = features.new_zeros((features.size(0), 1))
            x_dict[name] = self.row_embedding(
                features,
                table_labels,
                train_mask=table_train_mask,
                max_keys=max_keys,
                generator=generator,
            )

        return _RelationalEncoding(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            root_index=root_index,
            entity_table=entity_table,
        )


def _ordered_roots(edge_index: Tensor, *, num_rows: int) -> Tensor:
    task_row, root = edge_index
    permutation = task_row.argsort(stable=True)
    expected = torch.arange(num_rows, device=task_row.device)
    if not torch.equal(task_row[permutation], expected):
        raise ValueError("Each task row must match exactly one related row")
    return root[permutation]


def _propagate_targets(
    y: Tensor,
    root_index: Tensor,
    *,
    graph: _HomogeneousGraph,
    num_hops: int,
    num_nodes: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if y.is_floating_point():
        source = torch.stack(
            (y.to(dtype), torch.ones_like(y, dtype=dtype)), -1
        )
        state = source.new_zeros((num_nodes, 2))
        state.index_add_(0, root_index, source)
        state = _propagate(state, graph=graph, num_hops=num_hops)
        support = state[:, 1]
        return state[:, 0] / support.clamp(min=1), support > 0

    num_classes = int(y.max()) + 1 if y.numel() > 0 else 1
    source = F.one_hot(y.to(torch.long), num_classes).to(dtype)
    state = source.new_zeros((num_nodes, num_classes))
    state.index_add_(0, root_index, source)
    state = _propagate(state, graph=graph, num_hops=num_hops)
    support = state.sum(dim=-1)
    return state.argmax(dim=-1), support > 0


def _propagate(
    state: Tensor,
    *,
    graph: _HomogeneousGraph,
    num_hops: int,
) -> Tensor:
    row = graph.edge_index[0]
    for _ in range(num_hops):
        state = state + torch.segment_reduce(
            state[row],
            offsets=graph.colptr,
            reduce="sum",
            unsafe=True,
            initial=0,
        )
    return state
