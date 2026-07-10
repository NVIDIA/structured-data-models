import torch
from sdm import RelatedTables, RelationalData, RelationalSample
from sdm.relational.sampler import EXAMPLE_ID
from torch import Tensor


def with_full_sample(
    related_tables: RelatedTables,
    *,
    num_task_rows: int,
    root_index: Tensor | None = None,
    seed_time: Tensor | None = None,
) -> RelatedTables:
    """Attach an explicit full-graph sample to a hand-built test context."""
    node_batch: dict[str, Tensor] = {}
    for table_name, table in related_tables.tables.items():
        (batch,) = table[EXAMPLE_ID].id.unbind(-1)
        node_batch[table_name] = batch.long()
    entity_table = related_tables.task_links[0].table
    entity_batch = node_batch[entity_table]
    if root_index is None:
        permutation = entity_batch.argsort(stable=True)
        sorted_batch = entity_batch[permutation]
        first = torch.ones_like(sorted_batch, dtype=torch.bool)
        first[1:] = sorted_batch[1:] != sorted_batch[:-1]
        root_index = permutation[first]

    expected = torch.arange(num_task_rows)
    if root_index.numel() != num_task_rows or not torch.equal(
        entity_batch[root_index], expected
    ):
        raise ValueError("Test roots must map one-to-one to task rows")

    edge_indices = RelationalData(
        tables=related_tables.tables,
        relationships=related_tables.relationships,
    ).edge_indices(dtype=torch.long)

    offsets: dict[str, int] = {}
    num_nodes = 0
    for table_name, table in related_tables.tables.items():
        offsets[table_name] = num_nodes
        num_nodes += table.size(0)

    global_edges: list[Tensor] = []
    for relationship, edge_index in zip(
        related_tables.relationships, edge_indices
    ):
        offset = edge_index.new_tensor(
            [
                [offsets[relationship.left_table]],
                [offsets[relationship.right_table]],
            ]
        )
        edge_index = edge_index + offset
        global_edges.extend((edge_index, edge_index.flip(0)))
    global_edge_index = (
        torch.cat(global_edges, dim=1)
        if global_edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    hop = torch.full((num_nodes,), -1, dtype=torch.long)
    global_root = root_index + offsets[entity_table]
    hop[global_root] = 0
    frontier = torch.zeros(num_nodes, dtype=torch.bool)
    frontier[global_root] = True
    while bool(frontier.any()):
        src, dst = global_edge_index
        dst = dst[frontier[src]]
        dst = dst[hop[dst] < 0]
        if dst.numel() == 0:
            break
        frontier.fill_(False)
        frontier[dst] = True
        hop[frontier] = hop.max() + 1

    node_hops = {
        table_name: hop.narrow(
            0,
            offset,
            related_tables.tables[table_name].size(0),
        )
        for table_name, offset in offsets.items()
    }
    if any(bool((table_hop < 0).any()) for table_hop in node_hops.values()):
        raise ValueError("Every test row must be reachable from a task root")
    observed_hops = max(
        (
            int(table_hop.max())
            for table_hop in node_hops.values()
            if table_hop.numel()
        ),
        default=0,
    )
    num_hops = max(observed_hops, int(bool(related_tables.relationships)))

    edge_hops = tuple(
        torch.maximum(
            node_hops[relationship.left_table][edge_index[0]],
            node_hops[relationship.right_table][edge_index[1]],
        ).clamp_min(1)
        for relationship, edge_index in zip(
            related_tables.relationships, edge_indices
        )
    )
    sample = RelationalSample(
        node_batch=node_batch,
        node_hops=node_hops,
        edge_indices=edge_indices,
        edge_hops=edge_hops,
        task_edge_indices=(torch.stack((expected, root_index)),),
        num_hops=num_hops,
        num_neighbors=(10,) * num_hops,
        disjoint=True,
        temporal=seed_time is not None,
        temporal_strategy="last",
        seed_time=seed_time,
    )
    return RelatedTables(
        tables=related_tables.tables,
        relationships=related_tables.relationships,
        task_links=related_tables.task_links,
        sample=sample,
    )
