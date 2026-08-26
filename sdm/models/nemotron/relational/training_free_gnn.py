from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import RelatedTables, Relationship, Stype, TableTensor
from sdm.cache import Cache
from sdm.models.nemotron.relational.task import TaskGraph
from sdm.relational.join import join_index

_CHANNELS = 64
_RANDOM_SEED = 42


@dataclass(frozen=True)
class _RelationGraph:
    edge_type: int
    row: Tensor
    colptr: Tensor
    destination: slice


def _relationship_key(
    relationship: Relationship,
) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    return (
        relationship.left_table,
        relationship.left_columns,
        relationship.right_table,
        relationship.right_columns,
    )


def _canonical_related_tables(
    related_tables: RelatedTables,
    relationships: Sequence[Relationship] | None = None,
) -> RelatedTables:
    if relationships is None:
        relationships = related_tables.relationships
    return RelatedTables(
        tables={
            name: related_tables.tables[name]
            for name in sorted(related_tables.tables)
        },
        relationships=sorted(relationships, key=_relationship_key),
        task_links=related_tables.task_links,
    )


def _linear_state(
    *,
    in_channels: int,
    generator: torch.Generator,
    device: torch.device,
) -> Cache:
    bound = 1 / sqrt(in_channels)
    weight = torch.empty(
        (_CHANNELS, in_channels),
        dtype=torch.float32,
        device=device,
    ).uniform_(-bound, bound, generator=generator)
    bias = torch.empty(
        _CHANNELS,
        dtype=torch.float32,
        device=device,
    ).uniform_(-bound, bound, generator=generator)
    return Cache(weight=weight, bias=bias)


def _fit_projection(
    table: TableTensor,
    *,
    generator: torch.Generator,
) -> Cache:
    columns = tuple(sorted(table.columns[Stype.numerical]))
    in_channels = len(columns)
    if in_channels == 0:
        return Cache(
            columns=columns,
            weight=torch.empty(
                (_CHANNELS, 0),
                dtype=torch.float32,
                device=table.device,
            ),
            bias=torch.empty(0, dtype=torch.float32, device=table.device),
        )
    state = _linear_state(
        in_channels=in_channels,
        generator=generator,
        device=table.device,
    )
    state["columns"] = columns
    return state


def _project(table: TableTensor, state: Cache) -> Tensor:
    columns = cast(tuple[str, ...], state["columns"])
    if len(columns) == 0:
        return torch.ones(
            (table.size(-2), _CHANNELS),
            dtype=torch.float32,
            device=table.device,
        )

    indices = [
        table.columns[Stype.numerical].index(column) for column in columns
    ]
    x = table.numerical[..., indices].float()
    return F.linear(
        x,
        cast(Tensor, state["weight"]),
        cast(Tensor, state["bias"]),
    )


def _aggregate_stats(
    src_x: Tensor,
    colptr: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    sum_x = torch.segment_reduce(
        src_x,
        offsets=colptr,
        reduce="sum",
        unsafe=True,
        initial=0,
    )
    mean_x = sum_x / colptr.diff().clamp(min=1).unsqueeze(-1)
    var_x = (
        torch.segment_reduce(
            src_x.square(),
            offsets=colptr,
            reduce="mean",
            unsafe=True,
            initial=0,
        )
        - mean_x.square()
    )
    std_x = torch.where(
        var_x <= 1e-5,
        0.0,
        var_x.clamp(min=1e-5).sqrt(),
    )
    min_x = torch.segment_reduce(
        src_x,
        offsets=colptr,
        reduce="min",
        unsafe=True,
    )
    min_x = torch.where(min_x.isinf(), 0.0, min_x)
    max_x = torch.segment_reduce(
        src_x,
        offsets=colptr,
        reduce="max",
        unsafe=True,
    )
    max_x = torch.where(max_x.isinf(), 0.0, max_x)
    return sum_x, mean_x, min_x, max_x, std_x


def _gather_any(src_x: Tensor, row: Tensor, colptr: Tensor) -> Tensor:
    counts = colptr.diff()
    out = src_x.new_zeros((counts.numel(), src_x.size(-1)))
    mask = counts > 0
    out[mask] = src_x[row[(colptr[1:] - 1)[mask]]]
    return out


def _message(
    *,
    x: Tensor,
    edge_type: int,
    row: Tensor,
    colptr: Tensor,
    state: Cache,
) -> Tensor:
    if edge_type % 2 == 0:
        aggregated = torch.cat(
            _aggregate_stats(x[row], colptr),
            dim=-1,
        )
    else:
        aggregated = _gather_any(x, row, colptr)
    return F.linear(
        aggregated,
        cast(Tensor, state["weight"]),
        cast(Tensor, state["bias"]),
    )


def _fit_gnn_state(
    *,
    num_layers: int,
    num_edge_types: int,
    device: torch.device,
) -> tuple[tuple[Cache, ...], ...]:
    generator = torch.Generator(device=device).manual_seed(_RANDOM_SEED)
    layers = []
    for _ in range(num_layers):
        relations = []
        for edge_type in range(num_edge_types):
            in_channels = 5 * _CHANNELS if edge_type % 2 == 0 else _CHANNELS
            relations.append(
                _linear_state(
                    in_channels=in_channels,
                    generator=generator,
                    device=device,
                )
            )
        layers.append(tuple(relations))
    return tuple(layers)


def _relation_graphs(task_graph: TaskGraph) -> tuple[_RelationGraph, ...]:
    graph = task_graph.graph
    out = []
    for relation_index, relationship in enumerate(
        task_graph.related_tables.relationships
    ):
        if (
            relationship.left_table not in task_graph.related_tables.tables
            or relationship.right_table not in task_graph.related_tables.tables
        ):
            continue
        left, right = join_index(
            left_table=task_graph.related_tables.tables[
                relationship.left_table
            ],
            right_table=task_graph.related_tables.tables[
                relationship.right_table
            ],
            left_keys=relationship.left_columns,
            right_keys=relationship.right_columns,
            how="inner",
        )
        for direction, table_name in enumerate(
            (relationship.right_table, relationship.left_table)
        ):
            edge_type = 2 * relation_index + direction
            destination = graph.node_slice(table_name)
            if direction == 0:
                row = left + graph.start_node_offsets[relationship.left_table]
                col = right
            else:
                row = (
                    right + graph.start_node_offsets[relationship.right_table]
                )
                col = left
            col, perm = col.sort(stable=True)
            row = row[perm]
            counts = torch.bincount(
                col,
                minlength=destination.stop - destination.start,
            )
            colptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            out.append(
                _RelationGraph(
                    edge_type=edge_type,
                    row=row,
                    colptr=colptr,
                    destination=destination,
                )
            )
    return tuple(out)


def _gnn(
    *,
    x: Tensor,
    task_graph: TaskGraph,
    layers: tuple[tuple[Cache, ...], ...],
) -> Tensor:
    relation_graphs = _relation_graphs(task_graph)
    for layer in layers:
        out = x.clone()
        for relation in relation_graphs:
            destination = relation.destination
            out[destination] = out[destination] + _message(
                x=x,
                edge_type=relation.edge_type,
                row=relation.row,
                colptr=relation.colptr,
                state=layer[relation.edge_type],
            )
        x = F.layer_norm(out, (_CHANNELS,))
    return x


def _fit_state(
    *,
    x: TableTensor,
    task_graph: TaskGraph,
) -> Cache:
    projection_generator = torch.Generator(device=x.device).manual_seed(
        _RANDOM_SEED
    )
    table_projections = Cache()
    for name, table in task_graph.related_tables.tables.items():
        table_projections[name] = _fit_projection(
            table,
            generator=projection_generator,
        )
    task_projection = _fit_projection(
        x,
        generator=projection_generator,
    )
    return Cache(
        relationships=tuple(task_graph.related_tables.relationships),
        num_hops=task_graph.num_hops,
        table_projections=table_projections,
        task_projection=task_projection,
        layers=_fit_gnn_state(
            num_layers=task_graph.num_hops,
            num_edge_types=task_graph.graph.num_edge_types,
            device=x.device,
        ),
    )


def _embed(
    *,
    x: TableTensor,
    task_graph: TaskGraph,
    state: Cache,
) -> Tensor:
    table_projections = cast(Cache, state["table_projections"])
    table_embeddings = [
        _project(
            task_graph.related_tables.tables[name],
            cast(Cache, table_projections[name]),
        )
        for name in task_graph.related_tables.tables
    ]
    embedding = _gnn(
        x=torch.cat(table_embeddings, dim=-2),
        task_graph=task_graph,
        layers=cast(tuple[tuple[Cache, ...], ...], state["layers"]),
    )
    readout = embedding[task_graph.graph.node_slice(task_graph.readout_table)][
        task_graph.readout_index
    ]
    return readout + _project(
        x,
        cast(Cache, state["task_projection"]),
    )


def _fit_normalization(x: Tensor) -> tuple[Tensor, Tensor]:
    if x.size(-2) == 0:
        return x.new_zeros((1, x.size(-1))), x.new_full((1, x.size(-1)), 1e-6)
    mean = x.mean(dim=-2, keepdim=True)
    scale = x.var(dim=-2, correction=0, keepdim=True).sqrt() + 1e-6
    return mean, scale


def _to_table(x: Tensor) -> TableTensor:
    return TableTensor.from_tensor(
        x,
        columns=[f"__training_free_gnn_{i:02d}" for i in range(_CHANNELS)],
    )


def _fit_transform_training_free_gnn(
    *,
    x: TableTensor,
    related_tables: RelatedTables,
    num_hops: int | None,
) -> tuple[TableTensor, Cache]:
    related_tables = _canonical_related_tables(related_tables)
    task_graph = TaskGraph.from_input(
        x=x,
        related_tables=related_tables,
        num_hops=num_hops,
    )
    state = _fit_state(x=x, task_graph=task_graph)
    out = _embed(x=x, task_graph=task_graph, state=state)
    mean, scale = _fit_normalization(out)
    state["mean"] = mean
    state["scale"] = scale
    out = ((out - mean) / scale).clamp_(-15.0, 15.0)
    return _to_table(out), state.freeze()


def _transform_training_free_gnn(
    *,
    x: TableTensor,
    related_tables: RelatedTables,
    state: Cache,
) -> TableTensor:
    related_tables = _canonical_related_tables(
        related_tables,
        relationships=cast(tuple[Relationship, ...], state["relationships"]),
    )
    task_graph = TaskGraph.from_input(
        x=x,
        related_tables=related_tables,
        num_hops=cast(int, state["num_hops"]),
    )
    out = _embed(x=x, task_graph=task_graph, state=state)
    out = (
        (out - cast(Tensor, state["mean"])) / cast(Tensor, state["scale"])
    ).clamp_(-15.0, 15.0)
    return _to_table(out)
