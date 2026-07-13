"""Table-hop row encoding for KumoRFM."""

from dataclasses import dataclass

import torch
from torch import Tensor

from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
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

_MICROSECONDS_PER_MINUTE = 60 * 1_000_000
_MICROSECONDS_PER_HOUR = 60 * _MICROSECONDS_PER_MINUTE
_MICROSECONDS_PER_DAY = 24 * _MICROSECONDS_PER_HOUR


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

        graph = _materialize_graph(
            related_tables=related_tables,
            entity_table=entity_table,
            num_task_rows=x.size(0),
            device=parameter.device,
        )

        x_dict: dict[str, Tensor] = {}
        for table_name, table in related_tables.tables.items():
            batch = graph.batch_dict[table_name]
            hop = graph.hop_dict[table_name]
            if not ((batch < y.size(0)) & (hop >= 0)).any():
                continue

            categorical_align: CategoricalAlign | None = None
            if table.categorical.size(-1) > 0:
                table_train_mask = (batch < y.size(0)) & (hop >= 0)
                categorical_align = _fit_kumo_categorical_align(
                    table[table_train_mask.to(table.device)].select_stypes(
                        Stype.categorical
                    )
                )

            row_indices: list[Tensor] = []
            embeddings: list[Tensor] = []
            for current_hop in range(graph.num_hops + 1):
                row_index = (hop == current_hop).nonzero().flatten()
                if row_index.numel() == 0:
                    continue

                hop_batch = batch.index_select(0, row_index)
                train_mask = hop_batch < y.size(0)
                task_x = None
                if table_name == entity_table and current_hop == 0:
                    task_x = x.index_select(0, hop_batch)

                features = _preprocess_features(
                    table[row_index.to(table.device)],
                    train_mask=train_mask,
                    task_x=task_x,
                    batch=hop_batch,
                    seed_time=graph.seed_time,
                    device=parameter.device,
                    dtype=parameter.dtype,
                    categorical_align=categorical_align,
                )
                targets = y.index_select(0, hop_batch[train_mask])

                # KumoRFM uses all rows as column-attention context when a hop
                # has no labeled examples, without injecting any targets.
                context_mask = train_mask
                if not context_mask.any():
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

        retained_tables = set(x_dict)
        edge_index_dict = {
            edge_type: edge_index
            for edge_type, edge_index in graph.edge_index_dict.items()
            if edge_type[0] in retained_tables
            and edge_type[2] in retained_tables
            and edge_index.numel() > 0
        }
        return _TableHopEncoding(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            root_index=graph.root_index,
            num_hops=graph.num_hops,
        )


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


def _preprocess_features(
    table: TableTensor,
    *,
    train_mask: Tensor,
    task_x: Tensor | None,
    batch: Tensor,
    seed_time: Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    categorical_align: CategoricalAlign | None = None,
) -> Tensor:
    feature_table = table.select_stypes((Stype.numerical, Stype.categorical))
    if categorical_align is not None:
        categorical = categorical_align.transform(
            feature_table.select_stypes(Stype.categorical)
        ).categorical
        feature_table = feature_table.replace_blocks(categorical=categorical)
    values = (
        ToNumerical()
        .transform(feature_table)
        .numerical.to(
            device=device,
            dtype=dtype,
        )
    )
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

    if values.size(-1) == 0 or train_mask.count_nonzero() <= 1:
        return values.new_zeros((values.size(0), 1))

    table = TableTensor.from_tensor(values)
    clip_indices = tuple(range(feature_table.numerical.size(-1)))
    if task_x is not None:
        clip_indices = (
            *clip_indices,
            *range(values.size(-1) - task_x.size(-1), values.size(-1)),
        )
    processor = Sequential(
        _KumoNumericalClip(clip_indices),
        ConstantFilter(method="unique", threshold=1),
        StandardScale(epsilon=1e-6),
    )
    processor.fit(table[train_mask])
    values = processor.transform(table).numerical
    if values.size(-1) == 0:
        return values.new_zeros((values.size(0), 1))
    return values.clamp(-15.0, 15.0)


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
