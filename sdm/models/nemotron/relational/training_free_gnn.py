from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import (
    CategoricalTensor,
    NaT,
    RelatedTables,
    Relationship,
    Stype,
    TableTensor,
)
from sdm.cache import Cache
from sdm.models.nemotron.relational.task import TaskGraph
from sdm.processing import AlignCategories

_CHANNELS = 64
# Bound lookup memory deterministically. Categories sort by value so the cap
# never depends on Python hashing or input vocabulary-code order.
_MAX_CATEGORIES = 1024
_RANDOM_SEED = 42
_DAY_MICROSECONDS = 24 * 60 * 60 * 1_000_000


@dataclass(frozen=True)
class _RelationGraph:
    edge_type: int
    row: Tensor
    colptr: Tensor
    destination: slice


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
    return sum_x, mean_x, std_x, min_x, max_x


def _relationship_key(
    relationship: Relationship,
) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    return (
        relationship.left_table,
        relationship.left_columns,
        relationship.right_table,
        relationship.right_columns,
    )


def _canonical_related_tables(related_tables: RelatedTables) -> RelatedTables:
    return RelatedTables(
        tables={
            name: related_tables.tables[name]
            for name in sorted(related_tables.tables)
        },
        relationships=sorted(
            related_tables.relationships,
            key=_relationship_key,
        ),
        task_links=related_tables.task_links,
    )


def _linear_state(
    *,
    in_channels: int,
    out_channels: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    bound = 1 / sqrt(in_channels)
    weight = torch.empty(
        (out_channels, in_channels),
        device=device,
    ).uniform_(-bound, bound, generator=generator)
    bias = torch.empty(out_channels, device=device).uniform_(
        -bound,
        bound,
        generator=generator,
    )
    return weight, bias


def _continuous_values(
    table: TableTensor,
    *,
    numerical_columns: Sequence[str],
    datetime_columns: Sequence[str],
    datetime_origin: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    values = []
    valid = []
    if numerical_columns:
        indices = [
            table.columns[Stype.numerical].index(column)
            for column in numerical_columns
        ]
        numerical = table.numerical[..., indices].float()
        values.append(numerical)
        valid.append(numerical.isfinite())

    if datetime_columns:
        indices = [
            table.columns[Stype.datetime].index(column)
            for column in datetime_columns
        ]
        datetime = table.datetime[..., indices]
        datetime_valid = datetime != NaT
        if datetime_origin is None:
            maximum = torch.iinfo(datetime.dtype).max
            datetime_origin = torch.where(
                datetime_valid,
                datetime,
                maximum,
            ).amin(dim=-2)
            datetime_origin = torch.where(
                datetime_valid.any(dim=-2),
                datetime_origin,
                0,
            )
        relative = torch.where(
            datetime_valid,
            datetime - datetime_origin,
            0,
        )
        values.append(relative.float() / _DAY_MICROSECONDS)
        valid.append(datetime_valid)

    if values:
        return torch.cat(values, dim=-1), torch.cat(valid, dim=-1)

    empty = table.numerical.new_empty((table.size(-2), 0)).float()
    return empty, empty.bool()


def _categorical_table(
    table: TableTensor,
    columns: Sequence[str],
) -> TableTensor:
    indices = [
        table.columns[Stype.categorical].index(column) for column in columns
    ]
    categorical = table.categorical
    return TableTensor(
        columns={Stype.categorical: columns},
        categorical=CategoricalTensor(
            code=categorical.code[..., indices],
            categories=tuple(
                categorical.categories[index] for index in indices
            ),
        ),
    )


def _continuous_stats(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    count = valid.sum(dim=-2).clamp_min(1)
    finite_values = torch.where(valid, values, 0.0)
    mean = finite_values.sum(dim=-2) / count
    centered = torch.where(valid, values - mean, 0.0)
    scale = (centered.square().sum(dim=-2) / count).sqrt()
    scale = torch.where(scale > 1e-6, scale, 1.0)
    return mean, scale


def _fit_table_encoder(
    table: TableTensor,
    *,
    generator: torch.Generator,
) -> tuple[Cache, Tensor]:
    numerical_columns = tuple(sorted(table.columns[Stype.numerical]))
    datetime_columns = tuple(sorted(table.columns[Stype.datetime]))
    categorical_columns = tuple(sorted(table.columns[Stype.categorical]))

    datetime_origin = None
    if datetime_columns:
        indices = [
            table.columns[Stype.datetime].index(column)
            for column in datetime_columns
        ]
        datetime = table.datetime[..., indices]
        valid = datetime != NaT
        if datetime.size(-2) == 0:
            datetime_origin = datetime.new_zeros(datetime.size(-1))
        else:
            maximum = torch.iinfo(datetime.dtype).max
            datetime_origin = torch.where(
                valid,
                datetime,
                maximum,
            ).amin(dim=-2)
            datetime_origin = torch.where(
                valid.any(dim=-2),
                datetime_origin,
                0,
            )

    values, valid = _continuous_values(
        table,
        numerical_columns=numerical_columns,
        datetime_columns=datetime_columns,
        datetime_origin=datetime_origin,
    )
    mean, scale = _continuous_stats(values, valid)
    if values.size(-1) > 0:
        continuous_weight, continuous_bias = _linear_state(
            in_channels=values.size(-1),
            out_channels=_CHANNELS,
            generator=generator,
            device=table.device,
        )
    else:
        continuous_weight = values.new_empty((_CHANNELS, 0))
        continuous_bias = values.new_empty(0)

    categories: tuple[Tensor, ...] = ()
    category_weights: tuple[Tensor, ...] = ()
    categorical_code = table.categorical.code.new_empty((table.size(-2), 0))
    if categorical_columns:
        categorical = _categorical_table(table, categorical_columns)
        aligner = AlignCategories(sort_by="value")
        categorical = aligner.fit_transform(categorical)
        categories = tuple(
            values[:_MAX_CATEGORIES]
            for values in categorical.categorical.categories
        )
        categorical_code = categorical.categorical.code
        categorical_code = torch.stack(
            [
                torch.where(code < values.numel(), code, -1)
                for code, values in zip(
                    categorical_code.unbind(dim=-1),
                    categories,
                    strict=True,
                )
            ],
            dim=-1,
        )
        category_weights = tuple(
            torch.randn(
                (values.numel() + 1, _CHANNELS),
                device=table.device,
                generator=generator,
            )
            for values in categories
        )

    state = Cache(
        numerical_columns=numerical_columns,
        datetime_columns=datetime_columns,
        datetime_origin=datetime_origin,
        categorical_columns=categorical_columns,
        categories=categories,
        mean=mean,
        scale=scale,
        continuous_weight=continuous_weight,
        continuous_bias=continuous_bias,
        category_weights=category_weights,
    )
    return state, _encode_table(
        table=table,
        state=state,
        categorical_code=categorical_code,
    )


def _aligned_categorical_code(table: TableTensor, state: Cache) -> Tensor:
    categorical_columns = cast(tuple[str, ...], state["categorical_columns"])
    if not categorical_columns:
        return table.categorical.code.new_empty((table.size(-2), 0))

    categorical = _categorical_table(table, categorical_columns)
    aligner = AlignCategories()
    # Reuse AlignCategories' value-based unknown handling without refitting on
    # query values, which would grow the context vocabulary.
    return aligner._align_to_categories(
        categorical,
        (cast(tuple[Tensor, ...], state["categories"]),),
    )[0].categorical.code


def _encode_table(
    table: TableTensor,
    state: Cache,
    *,
    categorical_code: Tensor | None = None,
) -> Tensor:
    values, valid = _continuous_values(
        table,
        numerical_columns=cast(tuple[str, ...], state["numerical_columns"]),
        datetime_columns=cast(tuple[str, ...], state["datetime_columns"]),
        datetime_origin=cast(Tensor | None, state["datetime_origin"]),
    )
    mean = cast(Tensor, state["mean"])
    scale = cast(Tensor, state["scale"])
    values = torch.where(valid, (values - mean) / scale, 0.0)
    weight = cast(Tensor, state["continuous_weight"])
    bias = cast(Tensor, state["continuous_bias"])
    if values.size(-1) > 0:
        out = F.linear(values, weight, bias)
    else:
        out = values.new_zeros((values.size(-2), _CHANNELS))

    if categorical_code is None:
        categorical_code = _aligned_categorical_code(table, state)
    category_weights = cast(tuple[Tensor, ...], state["category_weights"])
    for code, category_weight in zip(
        categorical_code.unbind(dim=-1),
        category_weights,
        strict=True,
    ):
        out = out + category_weight[(code + 1).clamp_min(0).long()]

    if values.size(-1) + categorical_code.size(-1) == 0:
        return out.fill_(1)
    return out


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
            weight, bias = _linear_state(
                in_channels=in_channels,
                out_channels=_CHANNELS,
                generator=generator,
                device=device,
            )
            relations.append(Cache(weight=weight, bias=bias))
        layers.append(tuple(relations))
    return tuple(layers)


def _message(
    *,
    x: Tensor,
    edge_type: int,
    row: Tensor,
    colptr: Tensor,
    state: Cache,
) -> Tensor:
    weight = cast(Tensor, state["weight"])
    bias = cast(Tensor, state["bias"])

    if edge_type % 2 == 1:
        # This is a gather when the caller-declared right-to-left relation is
        # functional. Mean reduction keeps duplicate joins deterministic.
        gathered = torch.segment_reduce(
            x[row],
            offsets=colptr,
            reduce="mean",
            unsafe=True,
            initial=0,
        )
        return F.linear(gathered, weight, bias)

    sum_x, mean_x, std_x, min_x, max_x = _aggregate_stats(x[row], colptr)
    aggregated = torch.cat(
        (sum_x, mean_x, min_x, max_x, std_x),
        dim=-1,
    )
    return F.linear(aggregated, weight, bias)


def _relation_graphs(
    task_graph: TaskGraph,
    relationships: Sequence[Relationship],
) -> tuple[_RelationGraph, ...]:
    graph = task_graph.graph
    out = []
    for relation_index, relationship in enumerate(relationships):
        if (
            relationship.left_table not in task_graph.related_tables.tables
            or relationship.right_table not in task_graph.related_tables.tables
        ):
            continue
        for direction, table_name in enumerate(
            (relationship.right_table, relationship.left_table)
        ):
            edge_type = 2 * relation_index + direction
            destination = graph.node_slice(table_name)
            mask = graph.edge_type == edge_type
            row = graph.row[mask]
            col = graph.col[mask] - destination.start
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
    relationships: Sequence[Relationship],
    layers: tuple[tuple[Cache, ...], ...],
) -> Tensor:
    relation_graphs = _relation_graphs(task_graph, relationships)
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


def _fit_transform_training_free_gnn(
    *,
    x: TableTensor,
    related_tables: RelatedTables,
    num_hops: int | None,
) -> tuple[TableTensor, Cache]:
    # This is an SDM-native approximation of Kumo's training-free feature. It
    # uses the induced joins reconstructed from RelatedTables rather than the
    # exact sampler edge/count ordering retained by the Kumo runtime. The
    # public Relationship orientation defines left-to-right
    # [sum, mean, min, max, std] aggregation. Right-to-left uses a
    # deterministic mean, which equals Kumo's gather for functional joins.
    related_tables = _canonical_related_tables(related_tables)
    task_graph = TaskGraph.from_input(
        x=x,
        related_tables=related_tables,
        num_hops=num_hops,
    )
    generator = torch.Generator(device=x.device).manual_seed(_RANDOM_SEED)

    table_states = Cache()
    table_embeddings = []
    for name, table in related_tables.tables.items():
        table_state, embedding = _fit_table_encoder(
            table,
            generator=generator,
        )
        table_states[name] = table_state
        table_embeddings.append(embedding)
    task_state, task_embedding = _fit_table_encoder(x, generator=generator)

    layers = _fit_gnn_state(
        num_layers=task_graph.num_hops,
        num_edge_types=task_graph.graph.num_edge_types,
        device=x.device,
    )
    embedding = _gnn(
        x=torch.cat(table_embeddings, dim=-2),
        task_graph=task_graph,
        relationships=related_tables.relationships,
        layers=layers,
    )
    readout = embedding[task_graph.graph.node_slice(task_graph.readout_table)][
        task_graph.readout_index
    ]
    out = readout + task_embedding

    state = Cache(
        relationships=related_tables.relationships,
        num_hops=task_graph.num_hops,
        table_states=table_states,
        task_state=task_state,
        layers=layers,
    )
    return TableTensor.from_tensor(
        out,
        columns=[f"__training_free_gnn_{i:02d}" for i in range(_CHANNELS)],
    ), state


def _transform_training_free_gnn(
    *,
    x: TableTensor,
    related_tables: RelatedTables,
    state: Cache,
) -> TableTensor:
    relationships = cast(tuple[Relationship, ...], state["relationships"])
    canonical = _canonical_related_tables(
        RelatedTables(
            tables=related_tables.tables,
            relationships=relationships,
            task_links=related_tables.task_links,
        )
    )
    task_graph = TaskGraph.from_input(
        x=x,
        related_tables=canonical,
        num_hops=cast(int, state["num_hops"]),
    )
    table_states = cast(Cache, state["table_states"])
    table_embeddings = [
        _encode_table(canonical.tables[name], cast(Cache, table_states[name]))
        for name in canonical.tables
    ]
    embedding = _gnn(
        x=torch.cat(table_embeddings, dim=-2),
        task_graph=task_graph,
        relationships=relationships,
        layers=cast(tuple[tuple[Cache, ...], ...], state["layers"]),
    )
    readout = embedding[task_graph.graph.node_slice(task_graph.readout_table)][
        task_graph.readout_index
    ]
    task_embedding = _encode_table(x, cast(Cache, state["task_state"]))
    return TableTensor.from_tensor(
        readout + task_embedding,
        columns=[f"__training_free_gnn_{i:02d}" for i in range(_CHANNELS)],
    )
