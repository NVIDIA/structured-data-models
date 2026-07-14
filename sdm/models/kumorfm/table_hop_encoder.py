"""Table-hop row encoding for KumoRFM."""

from dataclasses import dataclass

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.relational.sampler import EXAMPLE_ID


@dataclass(frozen=True)
class _TableHopEncoding:
    """Encoded graph layout for one forward call."""

    x_dict: dict[str, Tensor]
    edge_index_dict: dict[tuple[str, str, str], Tensor]
    root_index: Tensor
    num_hops: int


class TableHopEncoder(torch.nn.Module):
    r"""Encode each related-table hop with a shared row embedding.

    Context and query samples are joined independently before their rows are
    combined. :meth:`RelatedTables.edge_indices` materializes the induced
    graph over retained rows, so its edges and shortest-path discovery hops
    may differ from the exact restricted-fanout edges and hops returned by
    PyG sampling.

    Args:
        row_embedding: Generic row embedding shared across tables and hops.
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
        max_keys: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _TableHopEncoding:
        r"""Return encoded rows and their induced heterogeneous graph."""
        parameter = self.row_embedding.lin.weight
        device = parameter.device
        dtype = parameter.dtype
        num_context = x_context.size(0)
        y_context = y_context.to(device=device)
        if (
            related_context_tables.task_links
            != related_query_tables.task_links
        ):
            raise ValueError(
                "Related context and query tables must use the same task links"
            )
        entity_table = related_context_tables.task_links[0].table

        context_edges, context_task_edges = (
            related_context_tables.edge_indices(
                x_context, dtype=torch.long, device=device
            )
        )
        query_edges, query_task_edges = related_query_tables.edge_indices(
            x_query, dtype=torch.long, device=device
        )

        table_names = list(related_context_tables.tables)
        context_rows = {
            name: table.size(0)
            for name, table in related_context_tables.tables.items()
        }

        batch_dict: dict[str, Tensor] = {}
        feature_dict: dict[str, Tensor] = {}
        for name in table_names:
            context_table = related_context_tables.tables[name]
            batches = [_example_ids(context_table, device=device)]
            features = [context_table.numerical.to(device=device, dtype=dtype)]
            if name in related_query_tables.tables:
                query_table = related_query_tables.tables[name]
                batches.append(
                    _example_ids(query_table, device=device) + num_context
                )
                features.append(
                    query_table.numerical.to(device=device, dtype=dtype)
                )
            batch_dict[name] = torch.cat(batches)
            feature_dict[name] = torch.cat(features)

        context_root = _ordered_roots(
            context_task_edges[0], num_rows=x_context.size(0)
        )
        query_root = (
            _ordered_roots(query_task_edges[0], num_rows=x_query.size(0))
            + context_rows[entity_table]
        )
        root_index = torch.cat((context_root, query_root))

        table_offsets: dict[str, int] = {}
        num_nodes = 0
        for name in table_names:
            table_offsets[name] = num_nodes
            num_nodes += batch_dict[name].numel()

        edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
        global_edges: list[Tensor] = []
        query_edges_by_relationship = dict(
            zip(related_query_tables.relationships, query_edges)
        )
        for index, (relationship, context_edge) in enumerate(
            zip(related_context_tables.relationships, context_edges)
        ):
            src_table = relationship.left_table
            dst_table = relationship.right_table
            edges = [context_edge]
            if relationship in query_edges_by_relationship:
                query_edge = query_edges_by_relationship[relationship]
                query_edge = query_edge + query_edge.new_tensor(
                    [[context_rows[src_table]], [context_rows[dst_table]]]
                )
                edges.append(query_edge)
            edge_index = torch.cat(edges, dim=1)
            edge_index_dict[(src_table, str(index), dst_table)] = edge_index
            global_offset = edge_index.new_tensor(
                [
                    [table_offsets[src_table]],
                    [table_offsets[dst_table]],
                ]
            )
            global_edge = edge_index + global_offset
            global_edges.extend((global_edge, global_edge.flip(0)))
        global_edge_index = (
            torch.cat(global_edges, dim=1)
            if global_edges
            else root_index.new_empty((2, 0))
        )

        hop = _shortest_hops(
            global_edge_index,
            root_index + table_offsets[entity_table],
            num_nodes=num_nodes,
        )
        num_hops = int(hop.max()) if hop.numel() > 0 else 0
        hop_dict = {
            name: hop.narrow(0, table_offsets[name], batch_dict[name].numel())
            for name in table_names
        }

        task_features = torch.cat(
            (
                x_context.numerical.to(device=device, dtype=dtype),
                x_query.numerical.to(device=device, dtype=dtype),
            )
        )
        x_dict: dict[str, Tensor] = {}
        for name in table_names:
            row_indices: list[Tensor] = []
            embeddings: list[Tensor] = []
            for current_hop in range(num_hops + 1):
                row_index = (hop_dict[name] == current_hop).nonzero().flatten()
                if row_index.numel() == 0:
                    continue

                node_batch = batch_dict[name].index_select(0, row_index)
                features = feature_dict[name].index_select(0, row_index)
                if name == entity_table and current_hop == 0:
                    features = torch.cat(
                        (
                            features,
                            task_features.index_select(0, node_batch),
                        ),
                        dim=-1,
                    )
                if features.size(-1) == 0:
                    features = features.new_zeros((features.size(0), 1))

                train_mask = node_batch < num_context
                targets = y_context.index_select(0, node_batch[train_mask])
                context_mask = train_mask
                if not bool(context_mask.any()):
                    context_mask = torch.ones_like(context_mask)

                embeddings.append(
                    self.row_embedding(
                        features,
                        targets,
                        train_mask=context_mask,
                        max_keys=max_keys,
                        generator=generator,
                    )
                )
                row_indices.append(row_index)

            row_index = torch.cat(row_indices)
            encoded = torch.cat(embeddings)
            x_dict[name] = encoded.new_zeros(
                (batch_dict[name].numel(), encoded.size(-1))
            ).index_copy(0, row_index, encoded)

        return _TableHopEncoding(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            root_index=root_index,
            num_hops=num_hops,
        )


def _example_ids(table: TableTensor, *, device: torch.device) -> Tensor:
    (example_id,) = table[EXAMPLE_ID].id.unbind(-1)
    return example_id.to(device=device, dtype=torch.long)


def _ordered_roots(edge_index: Tensor, *, num_rows: int) -> Tensor:
    task_row, root = edge_index
    permutation = task_row.argsort(stable=True)
    expected = torch.arange(num_rows, device=task_row.device)
    if not torch.equal(task_row[permutation], expected):
        raise ValueError("Each task row must match exactly one related row")
    return root[permutation]


def _shortest_hops(
    edge_index: Tensor,
    root_index: Tensor,
    *,
    num_nodes: int,
) -> Tensor:
    hop = torch.full(
        (num_nodes,), -1, dtype=torch.long, device=edge_index.device
    )
    hop[root_index] = 0
    frontier = torch.zeros(
        num_nodes, dtype=torch.bool, device=edge_index.device
    )
    frontier[root_index] = True
    current_hop = 0
    while bool(frontier.any()):
        src, dst = edge_index
        discovered = dst[frontier[src]]
        discovered = discovered[hop[discovered] < 0]
        if discovered.numel() == 0:
            break
        frontier.fill_(False)
        frontier[discovered] = True
        current_hop += 1
        hop[frontier] = current_hop
    return hop
