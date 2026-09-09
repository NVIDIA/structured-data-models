"""V11 backward-horizon aggregates of supplied, censored raw event tables."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from relbench.base import Database


@dataclass(frozen=True)
class LagSpec:
    """Raw event source, horizon and aggregation for a released task."""

    source_table: str
    entity_column: str
    aggregation: Literal["count", "sum", "mean", "min"]
    days: int
    value_column: str | None = None
    filter_column: str | None = None
    filter_op: Literal["in", "ne"] | None = None
    filter_value: tuple[str, ...] | int | None = None
    value_join: tuple[str, str, str, str] | None = None


# Exactly the 14 applicable V11 formulas; the other seven tasks are no-ops.
LAG_SPECS = {
    ("rel-amazon", "item-churn"): LagSpec("review", "product_id", "count", 91),
    ("rel-event", "user-ignore"): LagSpec(
        source_table="event_attendees",
        entity_column="user_id",
        aggregation="count",
        days=7,
        filter_column="status",
        filter_op="in",
        filter_value=("invited",),
    ),
    ("rel-f1", "driver-dnf"): LagSpec(
        source_table="results",
        entity_column="driverId",
        aggregation="count",
        days=30,
        filter_column="statusId",
        filter_op="ne",
        filter_value=1,
    ),
    ("rel-f1", "driver-top3"): LagSpec(
        source_table="qualifying",
        entity_column="driverId",
        aggregation="min",
        days=30,
        value_column="position",
    ),
    ("rel-hm", "user-churn"): LagSpec(
        source_table="transactions",
        entity_column="customer_id",
        aggregation="count",
        days=7,
    ),
    ("rel-stack", "user-badge"): LagSpec("badges", "UserId", "count", 91),
    ("rel-amazon", "item-ltv"): LagSpec(
        source_table="review",
        entity_column="product_id",
        aggregation="sum",
        days=91,
        value_column="__product_price__",
        value_join=("product_id", "product", "product_id", "price"),
    ),
    ("rel-amazon", "user-ltv"): LagSpec(
        source_table="review",
        entity_column="customer_id",
        aggregation="sum",
        days=91,
        value_column="__product_price__",
        value_join=("product_id", "product", "product_id", "price"),
    ),
    ("rel-avito", "ad-ctr"): LagSpec(
        source_table="SearchStream",
        entity_column="AdID",
        aggregation="mean",
        days=4,
        value_column="IsClick",
    ),
    ("rel-event", "user-attendance"): LagSpec(
        source_table="event_attendees",
        entity_column="user_id",
        aggregation="count",
        days=7,
        filter_column="status",
        filter_op="in",
        filter_value=("yes", "maybe"),
    ),
    ("rel-f1", "driver-position"): LagSpec(
        source_table="results",
        entity_column="driverId",
        aggregation="mean",
        days=60,
        value_column="positionOrder",
    ),
    ("rel-hm", "item-sales"): LagSpec(
        source_table="transactions",
        entity_column="article_id",
        aggregation="sum",
        days=7,
        value_column="price",
    ),
    ("rel-stack", "post-votes"): LagSpec("votes", "PostId", "count", 91),
    ("rel-trial", "study-adverse"): LagSpec(
        source_table="reported_event_totals",
        entity_column="nct_id",
        aggregation="sum",
        days=365,
        value_column="subjects_affected",
        filter_column="event_type",
        filter_op="in",
        filter_value=("serious", "deaths"),
    ),
}


class RawEventLags:
    """Prepare V11 event features without reading task targets or loading data.

    Args:
        db: Already censored database supplied to the adapter. The caller must
            enforce availability of static values such as product prices.
        dataset: Dataset name used to select the published V11 formula.
        task: Task name; unsupported tasks produce no feature columns.
        entity_col: Entity identifier in context/query frames.
        time_col: Per-row prediction cutoff in context/query frames.

    Ten bins have width equal to the task horizon. Bin zero is (cutoff-width,
    cutoff], including events exactly at cutoff as V11 did. Future events are
    excluded separately for every context/query row. No training labels are
    used, and no database reload, download or feature-artifact cache occurs.
    """

    def __init__(
        self,
        db: Database,
        *,
        dataset: str,
        task: str,
        entity_col: str,
        time_col: str,
    ) -> None:
        self.spec = LAG_SPECS.get((dataset, task))
        self.entity_col = entity_col
        self.time_col = time_col
        self.columns = (
            tuple(f"__kumo_arl{lag}__" for lag in range(10))
            if self.spec
            else ()
        )
        if self.spec is None:
            return
        spec = self.spec
        table = db.table_dict[spec.source_table]
        if table.time_col is None:
            raise ValueError(
                f"Lag source {spec.source_table!r} is not temporal"
            )
        self.source_time_col = table.time_col
        columns = [spec.entity_column, table.time_col]
        if spec.value_join is not None:
            columns.append(spec.value_join[0])
        elif spec.value_column is not None:
            columns.append(spec.value_column)
        if spec.filter_column is not None:
            columns.append(spec.filter_column)
        source = table.df[list(dict.fromkeys(columns))].copy()
        if spec.value_join is not None:
            left, right_table, right, value = spec.value_join
            prices = db.table_dict[right_table].df[[right, value]]
            if not prices[right].is_unique:
                raise ValueError(f"Lag value join key is not unique: {right}")
            source[spec.value_column] = source[left].map(
                prices.set_index(right)[value]
            )
            if source[spec.value_column].isna().any():
                raise ValueError(
                    "Lag value join did not copy every product price"
                )
        if spec.filter_column is not None:
            values = source[spec.filter_column]
            if spec.filter_op == "in":
                source = source[values.isin(spec.filter_value)]
            else:
                assert spec.filter_op == "ne"
                source = source[values.notna() & values.ne(spec.filter_value)]
        source = source[
            source[spec.entity_column].notna()
            & source[self.source_time_col].notna()
        ]
        if spec.aggregation != "count":
            source = source[source[spec.value_column].notna()]
        self.source = source.sort_values(
            [spec.entity_column, self.source_time_col],
            kind="stable",
            ignore_index=True,
        )
        self.groups = self.source.groupby(
            spec.entity_column, sort=False
        ).indices

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        """Return FP32 lag columns in input order, preserving the input index.

        Empty count/sum bins are zero; empty mean/min bins are NaN. Missing
        entities or cutoffs have those same empty-bin values. Other columns,
        including any labels, are ignored.
        """
        spec = self.spec
        if spec is None:
            return pd.DataFrame(index=queries.index)
        fill = 0.0 if spec.aggregation in {"count", "sum"} else np.nan
        output = np.full((len(queries), 10), fill, dtype=np.float32)
        seeds = queries[[self.entity_col, self.time_col]].reset_index(
            drop=True
        )
        seeds = seeds.dropna(subset=[self.entity_col, self.time_col])
        width = pd.Timedelta(days=spec.days).value
        for entity, seed_rows in seeds.groupby(
            self.entity_col, sort=False
        ).groups.items():
            positions = self.groups.get(entity)
            if positions is None:
                continue
            group = self.source.iloc[positions]
            rows = np.asarray(seed_rows, dtype=np.int64)
            anchors = (
                pd.to_datetime(seeds.loc[rows, self.time_col])
                .to_numpy(
                    dtype="datetime64[ns]",
                )
                .astype(np.int64)
            )
            times = (
                pd.to_datetime(group[self.source_time_col])
                .to_numpy(
                    dtype="datetime64[ns]",
                )
                .astype(np.int64)
            )
            values = None
            prefix = None
            if spec.aggregation != "count":
                raw = group[spec.value_column]
                if pd.api.types.is_float_dtype(raw.dtype):
                    values = raw.to_numpy(dtype=np.float32)
                elif pd.api.types.is_bool_dtype(raw.dtype):
                    values = raw.to_numpy(dtype=np.bool_)
                elif pd.api.types.is_integer_dtype(raw.dtype):
                    values = raw.to_numpy(dtype=np.int64)
                else:
                    raise TypeError(
                        f"Unsupported lag value dtype: {raw.dtype}"
                    )
                if spec.aggregation in {"sum", "mean"}:
                    cumulative = np.cumsum(values)
                    prefix = np.concatenate(
                        [np.zeros(1, dtype=cumulative.dtype), cumulative]
                    )
            for lag in range(10):
                end = anchors - lag * width
                left = np.searchsorted(times, end - width, side="right")
                right = np.searchsorted(times, end, side="right")
                count = right - left
                if spec.aggregation == "count":
                    result = count
                elif spec.aggregation == "sum":
                    assert prefix is not None
                    result = prefix[right] - prefix[left]
                elif spec.aggregation == "mean":
                    assert prefix is not None
                    result = np.full(len(rows), np.nan, dtype=np.float64)
                    nonempty = count > 0
                    result[nonempty] = (
                        prefix[right[nonempty]] - prefix[left[nonempty]]
                    ) / count[nonempty]
                else:
                    assert values is not None
                    result = np.asarray(
                        [
                            values[lo:hi].min() if hi > lo else np.nan
                            for lo, hi in zip(left, right)
                        ]
                    )
                output[rows, lag] = result
        return pd.DataFrame(output, index=queries.index, columns=self.columns)
