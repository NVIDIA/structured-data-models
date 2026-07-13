"""Table-hop row encoding for KumoRFM."""

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.processing import (
    CategoricalAlign,
    Clip,
    ConstantFilter,
    Sequential,
    StandardScale,
    ToNumerical,
)
from sdm.relational.sampler import EXAMPLE_ID

_ZERO_FEATURES = "zero_features"
_MICROSECONDS_PER_MINUTE = 60 * 1_000_000
_MICROSECONDS_PER_HOUR = 60 * _MICROSECONDS_PER_MINUTE
_MICROSECONDS_PER_DAY = 24 * _MICROSECONDS_PER_HOUR
_FeatureProcessor: TypeAlias = Sequential | Literal["zero_features"]
_RelationshipSchema: TypeAlias = tuple[
    str,
    tuple[str, ...],
    str,
    tuple[str, ...],
]
_TableSchema: TypeAlias = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
_TaskLinkSchema: TypeAlias = tuple[tuple[str, ...], str, tuple[str, ...]]
_FEATURE_STYPES = frozenset(
    {Stype.numerical, Stype.categorical, Stype.datetime}
)
_SamplingPolicy: TypeAlias = tuple[int, bool, bool, str] | None


class _KumoNumericalClip(Clip):
    r"""Apply KumoRFM's finite-only clipping to selected columns."""

    def __init__(self, indices: tuple[int, ...]) -> None:
        super().__init__(q_low=0.02, q_high=0.98)
        self.indices = indices

    def _fit(self, inp: TableTensor) -> None:
        values = inp.numerical
        lower = values.new_full((values.size(-1),), float("-inf"))
        upper = values.new_full((values.size(-1),), float("inf"))
        if self.indices:
            index = values.new_tensor(self.indices, dtype=torch.long)
            selected = values.index_select(-1, index)
            selected = selected.where(selected.isfinite(), torch.nan)
            q_low, q_high = torch.nanquantile(
                selected,
                values.new_tensor([self.q_low, self.q_high]),
                dim=0,
            )
            q_low = torch.nan_to_num(q_low, nan=float("-inf")).clamp(max=0.0)
            q_high = torch.nan_to_num(q_high, nan=float("inf"))
            lower = lower.index_copy(0, index, q_low)
            upper = upper.index_copy(0, index, q_high)

        self.lower_bound = lower
        self.upper_bound = upper

    def _transform(self, inp: TableTensor) -> TableTensor:
        out = super()._transform(inp)
        values = out.numerical
        return out.replace_blocks(
            numerical=values.where(values.isfinite(), 0.0)
        )


@dataclass(frozen=True)
class _TableHopEncoding:
    """Ephemeral encoded graph layout for one forward call."""

    x_dict: dict[str, Tensor]
    edge_index_dict: dict[tuple[str, str, str], Tensor]
    root_index: Tensor
    num_hops: int


@dataclass(frozen=True)
class _MaterializedGraph:
    batch_dict: dict[str, Tensor]
    hop_dict: dict[str, Tensor]
    edge_index_dict: dict[tuple[str, str, str], Tensor]
    root_index: Tensor
    num_hops: int
    seed_time: Tensor | None


class TableHopEncoder(torch.nn.Module):
    r"""Encode each sampled table hop with a shared row embedding.

    Exact sampled edges, task roots, example assignments, discovery hops, and
    anchor times are consumed from :class:`RelatedTables`. Each non-empty hop
    is preprocessed and encoded independently. Encoded rows are restored to
    their original table order so table-local edge indices remain aligned.

    Args:
        row_embedding: Generic row embedding shared across tables and hops.
    """

    def __init__(self, row_embedding: RowEmbedding) -> None:
        super().__init__()
        self.row_embedding = row_embedding

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        related_tables: RelatedTables,
        *,
        cache: Cache | None = None,
        max_keys: int | None = None,
        generator: torch.Generator | None = None,
    ) -> _TableHopEncoding:
        r"""Return encoded rows and their ephemeral sampled-graph layout.

        ``x`` is the processed numerical task table. Identifier columns in
        related tables are not used as model features. Datetime columns use
        seasonal and anchor-relative encodings. Root indices are
        entity-table-local and ordered by task row.
        """
        if x.dim() != 2:
            raise ValueError("`x` must be a two-dimensional tensor")
        if y.dim() != 1:
            raise ValueError("`y` must be a one-dimensional tensor")
        if y.size(0) > x.size(0):
            raise ValueError("There cannot be more targets than task rows")
        task_link = related_tables.task_links[0]
        entity_table = task_link.table
        if entity_table not in related_tables.tables:
            raise ValueError(f"Unknown entity table {entity_table!r}")

        entity_columns = related_tables.tables[entity_table].stypes
        missing_columns = [
            column
            for column in task_link.table_columns
            if column not in entity_columns
        ]
        if missing_columns:
            raise ValueError(
                f"Unknown entity table columns {missing_columns!r}"
            )
        parameter = self.row_embedding.lin.weight
        x = x.to(device=parameter.device, dtype=parameter.dtype)
        y = y.to(device=parameter.device)

        relationship_matches: dict[int, int] = {}
        if cache is not None and cache.is_replaying:
            table_schema = cast(_TableSchema, cache["table_schema"])
            retained_tables = cast(tuple[str, ...], cache["retained_tables"])
            _validate_table_schema(
                related_tables,
                table_schema,
                retained_tables=retained_tables,
            )

            task_link_schema = _task_link_schema(related_tables)
            if task_link_schema != cache["task_link_schema"]:
                raise ValueError(
                    "Query task link is incompatible with the fitted schema"
                )

            sampling_policy = _sampling_policy(related_tables)
            if sampling_policy != cache["sampling_policy"]:
                raise ValueError(
                    "Query sampling policy is incompatible with the fitted "
                    "context"
                )

            relationship_schema = cast(
                tuple[_RelationshipSchema, ...],
                cache["relationship_schema"],
            )
            retained_relationships = cast(
                tuple[int, ...], cache["retained_relationships"]
            )
            fitted_nonempty_relationships = cast(
                tuple[int, ...], cache["nonempty_relationships"]
            )
            relationship_matches = _match_relationships(
                query=_relationship_schema(related_tables),
                fitted=relationship_schema,
                retained_indices=retained_relationships,
                retained_tables=retained_tables,
            )
            num_hops = cast(int, cache["num_hops"])
            tables_cache = cast(Cache, cache["tables"])
        else:
            table_schema = _table_schema(related_tables)
            relationship_schema = _relationship_schema(related_tables)
            retained_tables = ()
            retained_relationships = ()
            fitted_nonempty_relationships = ()
            num_hops = -1
            tables_cache = None

        graph = _materialize_graph(
            related_tables=related_tables,
            entity_table=entity_table,
            num_task_rows=x.size(0),
            device=parameter.device,
        )

        if (
            cache is not None
            and cache.is_recording
            and any(
                bool((batch >= y.size(0)).any())
                for batch in graph.batch_dict.values()
            )
        ):
            raise ValueError(
                "Fit-related tables must contain only training examples"
            )

        if cache is None or not cache.is_replaying:
            num_hops = graph.num_hops
            retained_tables = tuple(
                table_name
                for table_name in related_tables.tables
                if bool(
                    (
                        (graph.batch_dict[table_name] < y.size(0))
                        & (graph.hop_dict[table_name] >= 0)
                    ).any()
                )
            )
            retained_table_set = set(retained_tables)
            retained_relationships = tuple(
                index
                for index, relationship in enumerate(
                    related_tables.relationships
                )
                if relationship.left_table in retained_table_set
                and relationship.right_table in retained_table_set
            )
            fitted_nonempty_relationships = tuple(
                index
                for index in retained_relationships
                if graph.edge_index_dict[
                    (
                        relationship_schema[index][0],
                        str(index),
                        relationship_schema[index][2],
                    )
                ].numel()
                > 0
            )

            if cache is not None:
                tables_cache = Cache()
                cache["table_schema"] = table_schema
                cache["task_link_schema"] = _task_link_schema(related_tables)
                cache["sampling_policy"] = _sampling_policy(related_tables)
                cache["relationship_schema"] = relationship_schema
                cache["retained_tables"] = retained_tables
                cache["retained_relationships"] = retained_relationships
                cache["nonempty_relationships"] = fitted_nonempty_relationships
                cache["num_hops"] = num_hops
                cache["tables"] = tables_cache

        x_dict: dict[str, Tensor] = {}
        for table_name in retained_tables:
            table = related_tables.tables.get(table_name)
            if table is None:
                x_dict[table_name] = parameter.new_empty(
                    (0, _embedding_channels(self.row_embedding))
                )
                continue

            batch = graph.batch_dict[table_name]
            hop = graph.hop_dict[table_name]

            table_cache: Cache | None = None
            if tables_cache is not None:
                if tables_cache.is_replaying:
                    table_cache = cast(Cache, tables_cache[table_name])
                else:
                    table_cache = Cache()
                    tables_cache[table_name] = table_cache

            categorical_align: CategoricalAlign | None = None
            if table.categorical.size(-1) > 0:
                if table_cache is not None and table_cache.is_replaying:
                    categorical_align = cast(
                        CategoricalAlign,
                        table_cache["categorical_align"],
                    )
                else:
                    table_train_mask = (batch < y.size(0)) & (hop >= 0)
                    categorical_align = _fit_kumo_categorical_align(
                        table[table_train_mask.to(table.device)].select_stypes(
                            Stype.categorical
                        )
                    )
                    if table_cache is not None:
                        table_cache["categorical_align"] = categorical_align

            row_indices: list[Tensor] = []
            embeddings: list[Tensor] = []
            for current_hop in range(num_hops + 1):
                row_index = (hop == current_hop).nonzero().flatten()
                hop_batch = batch.index_select(0, row_index)
                train_mask = hop_batch < y.size(0)
                task_x = None
                if table_name == entity_table and current_hop == 0:
                    task_x = x.index_select(0, hop_batch)

                feature_table = _to_feature_table(
                    table[row_index.to(table.device)],
                    task_x=task_x,
                    batch=hop_batch,
                    seed_time=graph.seed_time,
                    device=parameter.device,
                    dtype=parameter.dtype,
                    categorical_align=categorical_align,
                )
                clip_indices = tuple(range(table.numerical.size(-1)))
                if task_x is not None:
                    clip_indices = (
                        *clip_indices,
                        *range(
                            feature_table.size(-1) - task_x.size(-1),
                            feature_table.size(-1),
                        ),
                    )
                has_context = bool(train_mask.any())
                hop_key = f"hop{current_hop}"
                hop_cache: Cache | None = None
                cached_context = False
                if table_cache is not None:
                    if table_cache.is_replaying:
                        hop_cache = cast(Cache, table_cache[hop_key])
                        expected_width = cast(
                            int, hop_cache["source_feature_width"]
                        )
                        if feature_table.size(-1) != expected_width:
                            raise ValueError(
                                f"Query source feature width for "
                                f"{table_name!r} hop {current_hop} is "
                                f"incompatible: expected {expected_width}, "
                                f"got {feature_table.size(-1)}"
                            )
                        cached_context = cast(bool, hop_cache["has_context"])
                    else:
                        hop_cache = Cache(
                            {
                                "has_context": has_context,
                                "source_feature_width": feature_table.size(-1),
                            }
                        )
                        table_cache[hop_key] = hop_cache

                if row_index.numel() == 0:
                    continue

                row_cache: Cache | None = None
                if cached_context:
                    assert hop_cache is not None
                    processor = cast(_FeatureProcessor, hop_cache["processor"])
                    features = _transform_features(
                        feature_table,
                        processor,
                    )
                    row_cache = cast(Cache, hop_cache["row_embedding"])
                else:
                    features, processor = _fit_transform_features(
                        feature_table,
                        fit_mask=train_mask,
                        clip_indices=clip_indices,
                    )
                    if hop_cache is not None and has_context:
                        row_cache = Cache()
                        hop_cache["processor"] = processor
                        hop_cache["row_embedding"] = row_cache

                targets = y.index_select(0, hop_batch[train_mask])

                # KumoRFM uses all rows as column-attention context when a hop
                # has no labeled examples, without injecting any targets.
                context_mask = train_mask
                if not has_context and not cached_context:
                    context_mask = torch.ones_like(context_mask)

                if row_cache is None:
                    embedding = self.row_embedding(
                        features,
                        targets,
                        train_mask=context_mask,
                        max_keys=max_keys,
                        generator=generator,
                    )
                else:
                    embedding = self.row_embedding(
                        features,
                        targets,
                        train_mask=context_mask,
                        max_keys=max_keys,
                        cache=row_cache,
                        generator=generator,
                    )
                embeddings.append(embedding)
                row_indices.append(row_index)

            if embeddings:
                row_index = torch.cat(row_indices)
                encoded = torch.cat(embeddings)
                x_dict[table_name] = encoded.new_zeros(
                    (table.size(0), encoded.size(-1))
                ).index_copy(
                    0,
                    row_index,
                    encoded,
                )
            else:
                x_dict[table_name] = parameter.new_empty(
                    (table.size(0), _embedding_channels(self.row_embedding))
                )

        empty_edge_index = graph.root_index.new_empty((2, 0))
        edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
        for index in retained_relationships:
            relationship = relationship_schema[index]
            edge_type = (relationship[0], str(index), relationship[2])
            if cache is not None and cache.is_replaying:
                query_index = relationship_matches.get(index)
                if query_index is None:
                    edge_index = empty_edge_index
                else:
                    query_edge_type = (
                        relationship[0],
                        str(query_index),
                        relationship[2],
                    )
                    edge_index = graph.edge_index_dict.get(
                        query_edge_type,
                        empty_edge_index,
                    )
                if (
                    index not in fitted_nonempty_relationships
                    and edge_index.numel() == 0
                ):
                    continue
            else:
                edge_index = graph.edge_index_dict.get(
                    edge_type,
                    empty_edge_index,
                )
                if edge_index.numel() == 0:
                    continue
            edge_index_dict[edge_type] = edge_index

        return _TableHopEncoding(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            root_index=graph.root_index,
            num_hops=num_hops,
        )


def _embedding_channels(row_embedding: RowEmbedding) -> int:
    channels = row_embedding.lin.out_features
    readout_token = getattr(row_embedding, "readout_token", None)
    if isinstance(readout_token, Tensor):
        channels *= readout_token.size(-2)
    return channels


def _table_schema(related_tables: RelatedTables) -> _TableSchema:
    return tuple(
        (
            table_name,
            tuple(
                (column, stype.value)
                for column, stype in table.stypes.items()
                if stype in _FEATURE_STYPES
            ),
        )
        for table_name, table in related_tables.tables.items()
    )


def _validate_table_schema(
    related_tables: RelatedTables,
    fitted: _TableSchema,
    *,
    retained_tables: tuple[str, ...],
) -> None:
    fitted_by_name = dict(fitted)
    for table_name in retained_tables:
        table = related_tables.tables.get(table_name)
        if table is None:
            continue
        actual = tuple(
            (column, stype.value)
            for column, stype in table.stypes.items()
            if stype in _FEATURE_STYPES
        )
        expected = fitted_by_name[table_name]
        if actual != expected:
            raise ValueError(
                f"Query table {table_name!r} is incompatible with the "
                "fitted schema"
            )


def _task_link_schema(related_tables: RelatedTables) -> _TaskLinkSchema:
    task_link = related_tables.task_links[0]
    return (
        tuple(task_link.task_columns),
        task_link.table,
        tuple(task_link.table_columns),
    )


def _sampling_policy(related_tables: RelatedTables) -> _SamplingPolicy:
    sample = related_tables.sample
    if sample is None:
        return None
    return (
        sample.num_hops,
        sample.disjoint,
        sample.temporal,
        sample.temporal_strategy,
    )


def _relationship_schema(
    related_tables: RelatedTables,
) -> tuple[_RelationshipSchema, ...]:
    return tuple(
        (
            relationship.left_table,
            tuple(relationship.left_columns),
            relationship.right_table,
            tuple(relationship.right_columns),
        )
        for relationship in related_tables.relationships
    )


def _match_relationships(
    *,
    query: tuple[_RelationshipSchema, ...],
    fitted: tuple[_RelationshipSchema, ...],
    retained_indices: tuple[int, ...],
    retained_tables: tuple[str, ...],
) -> dict[int, int]:
    unused = list(retained_indices)
    retained_table_set = set(retained_tables)
    matches: dict[int, int] = {}
    for query_index, relationship in enumerate(query):
        try:
            fitted_index = next(
                index for index in unused if fitted[index] == relationship
            )
        except StopIteration:
            if (
                relationship[0] in retained_table_set
                and relationship[2] in retained_table_set
            ):
                raise ValueError(
                    f"Query relationship at index {query_index} is "
                    "incompatible with the fitted schema"
                ) from None
            continue
        unused.remove(fitted_index)
        matches[fitted_index] = query_index
    return matches


def _materialize_graph(
    *,
    related_tables: RelatedTables,
    entity_table: str,
    num_task_rows: int,
    device: torch.device,
) -> _MaterializedGraph:
    if related_tables.sample is not None:
        return _materialize_exact_graph(
            related_tables=related_tables,
            entity_table=entity_table,
            num_task_rows=num_task_rows,
            device=device,
        )
    return _materialize_zero_hop_graph(
        related_tables=related_tables,
        entity_table=entity_table,
        num_task_rows=num_task_rows,
        device=device,
    )


def _materialize_exact_graph(
    *,
    related_tables: RelatedTables,
    entity_table: str,
    num_task_rows: int,
    device: torch.device,
) -> _MaterializedGraph:
    sample = related_tables.sample
    assert sample is not None
    if not sample.disjoint:
        raise ValueError("KumoRFM requires disjoint relational samples")

    batch_dict = {
        table_name: batch.to(device=device, dtype=torch.long)
        for table_name, batch in sample.node_batch.items()
    }
    hop_dict = {
        table_name: hop.to(device=device, dtype=torch.long)
        for table_name, hop in sample.node_hops.items()
    }
    _validate_example_ids(
        batch_dict=batch_dict,
        num_task_rows=num_task_rows,
    )

    edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
    for index, (relationship, edge_index) in enumerate(
        zip(related_tables.relationships, sample.edge_indices)
    ):
        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_index_dict[
            (relationship.left_table, str(index), relationship.right_table)
        ] = edge_index

    task_edge_index = sample.task_edge_indices[0].to(
        device=device,
        dtype=torch.long,
    )
    task_row, root = task_edge_index
    if task_row.numel() != num_task_rows:
        raise ValueError("Sampled task roots must map one-to-one to task rows")
    permutation = task_row.argsort(stable=True)
    expected = torch.arange(num_task_rows, device=device)
    if not torch.equal(task_row[permutation], expected):
        raise ValueError("Sampled task roots must map one-to-one to task rows")
    root_index = root[permutation]
    entity_batch = batch_dict[entity_table]
    entity_hop = hop_dict[entity_table]
    if not torch.equal(entity_batch[root_index], expected) or not bool(
        (entity_hop[root_index] == 0).all()
    ):
        raise ValueError(
            "Each sampled task root must belong to its task example at hop "
            "zero"
        )

    seed_time = sample.seed_time
    if seed_time is not None:
        if seed_time.numel() != num_task_rows:
            raise ValueError(
                "Sampled seed times must align one-to-one with task rows"
            )
        seed_time = seed_time.to(device=device, dtype=torch.long)

    return _MaterializedGraph(
        batch_dict=batch_dict,
        hop_dict=hop_dict,
        edge_index_dict=edge_index_dict,
        root_index=root_index,
        num_hops=sample.num_hops,
        seed_time=seed_time,
    )


def _materialize_zero_hop_graph(
    *,
    related_tables: RelatedTables,
    entity_table: str,
    num_task_rows: int,
    device: torch.device,
) -> _MaterializedGraph:
    r"""Handle only unambiguous, entity-only, manually supplied contexts."""
    if len(related_tables.relationships) > 0:
        raise ValueError(
            "Relational KumoRFM inputs require exact sample metadata; create "
            "them with RelationalSampler"
        )
    nonempty_other_tables = [
        table_name
        for table_name, table in related_tables.tables.items()
        if table_name != entity_table and table.size(0) > 0
    ]
    if nonempty_other_tables:
        raise ValueError(
            "KumoRFM inputs without sample metadata must be entity-only"
        )

    entity_links = (
        link
        for link in related_tables.task_links
        if link.table == entity_table
    )
    if not any(
        any(
            task_column == EXAMPLE_ID and table_column == EXAMPLE_ID
            for task_column, table_column in zip(
                link.task_columns, link.table_columns
            )
        )
        for link in entity_links
    ):
        raise ValueError(
            "Entity task link must contain the sampled example identifier"
        )

    batch_dict = {
        table_name: _example_ids(table, device=device)
        for table_name, table in related_tables.tables.items()
    }
    _validate_example_ids(
        batch_dict=batch_dict,
        num_task_rows=num_task_rows,
    )

    entity_batch = batch_dict[entity_table]
    permutation = entity_batch.argsort(stable=True)
    expected = torch.arange(num_task_rows, device=device)
    if entity_batch.numel() != num_task_rows or not torch.equal(
        entity_batch[permutation], expected
    ):
        raise ValueError(
            "Entity-only inputs require exactly one root row per task example"
        )

    return _MaterializedGraph(
        batch_dict=batch_dict,
        hop_dict={
            table_name: torch.zeros_like(batch)
            for table_name, batch in batch_dict.items()
        },
        edge_index_dict={},
        root_index=permutation,
        num_hops=0,
        seed_time=None,
    )


def _validate_example_ids(
    *,
    batch_dict: dict[str, Tensor],
    num_task_rows: int,
) -> None:
    for table_name, batch in batch_dict.items():
        if batch.numel() > 0 and (
            bool((batch < 0).any()) or bool((batch >= num_task_rows).any())
        ):
            raise ValueError(
                f"Sampled example identifiers in {table_name!r} must be "
                f"between 0 and {num_task_rows - 1}"
            )


def _example_ids(table: TableTensor, *, device: torch.device) -> Tensor:
    if EXAMPLE_ID not in table.stypes or table.stype(EXAMPLE_ID) != Stype.id:
        raise ValueError(
            f"Related tables must contain an ID column named {EXAMPLE_ID!r}"
        )

    (example_id,) = table[EXAMPLE_ID].id.unbind(-1)
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if example_id.dtype not in integer_dtypes:
        raise ValueError("Sampled example identifiers must be integers")
    return example_id.to(device=device, dtype=torch.long)


def _fit_kumo_categorical_align(table: TableTensor) -> CategoricalAlign:
    r"""Fit table-global categories in Kumo's frequency/value order."""
    orders: list[list[int]] = []
    for index, category in enumerate(table.categorical.categories):
        codes = table.categorical[..., index]
        observed = codes[codes >= 0].to(torch.long)
        counts = torch.bincount(observed, minlength=category.numel()).tolist()
        values = category.tolist()
        order = [i for i, count in enumerate(counts) if count > 0]
        order.sort(key=values.__getitem__)
        order.sort(key=counts.__getitem__, reverse=True)
        orders.append(order)

    data = table.categorical.new_full(
        (max(map(len, orders), default=0), len(orders)),
        -1,
    )
    for index, order in enumerate(orders):
        data[: len(order), index] = data.new_tensor(order)
    ordered = TableTensor(
        columns={Stype.categorical: table.columns[Stype.categorical]},
        categorical=CategoricalTensor(
            data=data,
            categories=table.categorical.categories,
        ),
    )
    return CategoricalAlign().fit(ordered)


@torch.inference_mode(False)
def _to_feature_table(
    table: TableTensor,
    *,
    task_x: Tensor | None,
    batch: Tensor,
    seed_time: Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    categorical_align: CategoricalAlign | None = None,
) -> TableTensor:
    feature_table = table.select_stypes((Stype.numerical, Stype.categorical))
    if categorical_align is not None:
        categorical = categorical_align.transform(
            feature_table.select_stypes(Stype.categorical)
        ).categorical
        feature_table = feature_table.replace_blocks(categorical=categorical)
    feature_table = ToNumerical().transform(feature_table)
    values = feature_table.numerical.to(device=device, dtype=dtype)
    if table.datetime.size(-1) > 0:
        if seed_time is None:
            raise ValueError("KumoRFM datetime features require anchor times")
        datetime_features = _encode_datetime_features(
            timestamp=table.datetime.to(device=device),
            anchor_time=seed_time.index_select(0, batch),
            dtype=dtype,
        )
        values = torch.cat((values, datetime_features), dim=-1)
    if task_x is not None:
        values = torch.cat((values, task_x), dim=-1)
    values = values.where(values.isfinite(), torch.nan)
    return TableTensor.from_tensor(values)


def _encode_datetime_features(
    *,
    timestamp: Tensor,
    anchor_time: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    r"""Encode microsecond timestamps as KumoRFM time features."""
    missing = timestamp == torch.iinfo(torch.int64).min
    safe_timestamp = timestamp.masked_fill(missing, 0)
    days = safe_timestamp.div(
        _MICROSECONDS_PER_DAY,
        rounding_mode="floor",
    )
    time_of_day = safe_timestamp.remainder(_MICROSECONDS_PER_DAY)
    minute = time_of_day.div(
        _MICROSECONDS_PER_MINUTE,
        rounding_mode="floor",
    ).remainder(60)
    hour = time_of_day.div(
        _MICROSECONDS_PER_HOUR,
        rounding_mode="floor",
    )

    year, month, day = _civil_from_days(days)
    leap = (year.remainder(4) == 0) & (
        (year.remainder(100) != 0) | (year.remainder(400) == 0)
    )
    month_lengths = timestamp.new_tensor(
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    )
    days_in_month = month_lengths[month - 1]
    days_in_month = days_in_month + (leap & (month == 2))
    month_starts = timestamp.new_tensor(
        (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    )
    day_of_year = month_starts[month - 1] + day - 1
    day_of_year = day_of_year + (leap & (month > 2))
    days_in_year = 365 + leap
    day_of_week = (days + 3).remainder(7)

    timestamp_seconds = safe_timestamp.div(1_000_000, rounding_mode="floor")
    anchor_seconds = anchor_time.div(1_000_000, rounding_mode="floor")
    relative_days = (anchor_seconds.unsqueeze(-1) - timestamp_seconds).to(
        dtype
    ) / (24 * 60 * 60)
    features = torch.stack(
        (
            minute.to(dtype) / 60,
            hour.to(dtype) / 24,
            day_of_week.to(dtype) / 7,
            (day - 1).to(dtype) / days_in_month.to(dtype),
            day_of_year.to(dtype) / days_in_year.to(dtype),
            relative_days,
        ),
        dim=-1,
    )
    features = features.masked_fill(missing.unsqueeze(-1), 0)
    return features.flatten(-2)


def _civil_from_days(days: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""Convert days since Unix epoch to Gregorian year, month, and day."""
    shifted = days + 719_468
    era = shifted.div(146_097, rounding_mode="floor")
    day_of_era = shifted - era * 146_097
    year_of_era = (
        day_of_era
        - day_of_era.div(1_460, rounding_mode="floor")
        + day_of_era.div(36_524, rounding_mode="floor")
        - day_of_era.div(146_096, rounding_mode="floor")
    ).div(365, rounding_mode="floor")
    year = year_of_era + era * 400
    day_of_year = day_of_era - (
        365 * year_of_era
        + year_of_era.div(4, rounding_mode="floor")
        - year_of_era.div(100, rounding_mode="floor")
    )
    month_prime = (5 * day_of_year + 2).div(153, rounding_mode="floor")
    day = (
        day_of_year
        - (153 * month_prime + 2).div(
            5,
            rounding_mode="floor",
        )
        + 1
    )
    month = month_prime + torch.where(month_prime < 10, 3, -9)
    year = year + (month <= 2)
    return year, month, day


@torch.inference_mode(False)
def _fit_transform_features(
    table: TableTensor,
    *,
    fit_mask: Tensor,
    clip_indices: tuple[int, ...],
) -> tuple[Tensor, _FeatureProcessor]:
    values = table.numerical
    if values.size(-1) == 0 or fit_mask.count_nonzero() <= 1:
        return values.new_zeros((values.size(0), 1)), _ZERO_FEATURES

    processor = Sequential(
        _KumoNumericalClip(clip_indices),
        ConstantFilter(method="unique", threshold=1),
        StandardScale(epsilon=1e-6),
    )
    processor.fit(table[fit_mask])
    values = processor.transform(table).numerical
    if values.size(-1) == 0:
        return values.new_zeros((values.size(0), 1)), _ZERO_FEATURES
    return values.clamp(-15.0, 15.0), processor


@torch.inference_mode(False)
def _transform_features(
    table: TableTensor,
    processor: _FeatureProcessor,
) -> Tensor:
    if processor == _ZERO_FEATURES:
        return table.numerical.new_zeros((table.size(0), 1))
    return processor.transform(table).numerical.clamp(-15.0, 15.0)
