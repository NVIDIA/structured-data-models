"""Benchmark NemotronRelational on RelArena's official outer test split.

This runs a fixed RelArena outer-test configuration with a seeded random
context of at most 10,000 train and validation rows, eight estimators, and
symmetric two-hop neighbor sampling. It adds lagged-target features defined by
the released Kumo RFM v2 benchmark where configured. Each invocation evaluates
one width.

Examples:
    python rel_arena.py --dataset rel-f1 --task driver-dnf
    python rel_arena.py --dataset rel-amazon --task item-ltv --num_neighbors 32
    for n in 1 2 4 8 16 32 48 64 96 128; do
        python rel_arena.py --dataset rel-f1 --task driver-dnf \
            --num_neighbors "$n"
    done
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import relarena
import relbench
import torch
import tqdm

import sdm

CONTEXT_SIZE = 10_000
BATCH_SIZE = 1_000
NUM_ESTIMATORS = 8
SEED = 0
LAG_TIMESTEPS = 10
NEIGHBOR_SWEEP = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)

TASKS = {
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "item-ltv"),
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "user-ltv"),
    ("rel-avito", "ad-ctr"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-top3"),
    ("rel-hm", "item-sales"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "post-votes"),
    ("rel-stack", "user-badge"),
    ("rel-stack", "user-engagement"),
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    ("rel-trial", "study-outcome"),
}


@dataclass(frozen=True)
class LagSpec:
    """Executable autoregressive target configuration for one Kumo v2 task."""

    source_table: str
    entity_column: str
    aggregation: Literal["count", "sum", "mean", "min"]
    days: int
    value_column: str | None = None
    include_column: str | None = None
    include_values: tuple[Any, ...] | None = None
    exclude_column: str | None = None
    exclude_value: Any = None
    value_join: tuple[str, str, str, str] | None = None


# These are the target expressions from Kumo RFM's released v2 RelBench
# classification and regression scripts. Kumo's autoregressive transform drops
# entity WHERE/ASSUMING clauses while preserving filters inside the aggregate.
LAG_SPECS = {
    ("rel-amazon", "item-churn"): LagSpec(
        source_table="review",
        entity_column="product_id",
        aggregation="count",
        days=91,
    ),
    ("rel-event", "user-ignore"): LagSpec(
        source_table="event_attendees",
        entity_column="user_id",
        aggregation="count",
        days=7,
        include_column="status",
        include_values=("invited",),
    ),
    ("rel-f1", "driver-dnf"): LagSpec(
        source_table="results",
        entity_column="driverId",
        aggregation="count",
        days=30,
        exclude_column="statusId",
        exclude_value=1,
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
    ("rel-stack", "user-badge"): LagSpec(
        source_table="badges",
        entity_column="UserId",
        aggregation="count",
        days=91,
    ),
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
        include_column="status",
        include_values=("yes", "maybe"),
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
    ("rel-stack", "post-votes"): LagSpec(
        source_table="votes",
        entity_column="PostId",
        aggregation="count",
        days=91,
    ),
    ("rel-trial", "study-adverse"): LagSpec(
        source_table="reported_event_totals",
        entity_column="nct_id",
        aggregation="sum",
        days=365,
        value_column="subjects_affected",
        include_column="event_type",
        include_values=("serious", "deaths"),
    ),
}


def _delimiter_multicategorical(series: pd.Series) -> str | None:
    """Return Kumo v2.22's inferred scalar-string list delimiter."""
    values = series.iloc[:500].dropna()
    if (
        values.empty
        or not values.map(lambda value: isinstance(value, str)).all()
    ):
        return None
    counts: dict[str, int] = {}
    for character in "\n".join(values):
        if character in {";", ":", "|", "\t"}:
            counts[character] = counts.get(character, 0) + 1
    if not counts:
        return None
    delimiter = max(counts, key=counts.__getitem__)
    unique_rows = values.nunique()
    unique_members = values.str.split(delimiter).explode().nunique()
    if unique_rows > 1.5 * unique_members and unique_members <= 100:
        return delimiter
    return None


def _native_multicategorical(series: pd.Series) -> bool:
    """Return whether sampled values contain native list-like categories."""
    return any(
        isinstance(value, (list, tuple, set, np.ndarray))
        for value in series.dropna().iloc[:1_000]
    )


def _model_tables(
    db: relbench.base.Database,
) -> dict[str, sdm.TableTensor]:
    """Tensorize the cleaned database and drop unsupported columns."""
    tables = {}
    for name, table in db.table_dict.items():
        sample = table.df.head(10_000)
        overrides = dict.fromkeys(table.fkey_col_to_pkey_table, "id")
        if table.pkey_col is not None:
            overrides[table.pkey_col] = "id"

        dropped = [
            column
            for column in sample.columns
            if column not in overrides
            and (
                column.startswith("Unnamed:")
                or sample[column].isna().all()
                or _native_multicategorical(sample[column])
                or _delimiter_multicategorical(sample[column]) is not None
            )
        ]
        candidates = sample.drop(columns=dropped)
        stypes = sdm.infer_stypes(
            candidates,
            overrides=overrides,
            text="infer",
            unsupported="drop",
        )
        stypes = {
            column: stype
            for column, stype in stypes.items()
            if stype != sdm.Stype.text
        }
        tables[name] = sdm.TableTensor.from_pandas(
            df=table.df,
            stypes=stypes,
        )
    return tables


def _lag_source(
    db: relbench.base.Database,
    spec: LagSpec,
) -> tuple[pd.DataFrame, str]:
    """Return the filtered source rows for one autoregressive target."""
    table = db.table_dict[spec.source_table]
    if table.time_col is None:
        raise ValueError(f"Lag source {spec.source_table!r} is not temporal")
    source = table.df.copy()
    if spec.value_join is not None:
        left, right_table, right, value = spec.value_join
        values = db.table_dict[right_table].df.set_index(right)[value]
        if not values.index.is_unique:
            raise RuntimeError(f"Lag value join key is not unique: {right}")
        copied = source[left].map(values)
        if copied.isna().any():
            raise RuntimeError("Lag value join is incomplete")
        source[cast(str, spec.value_column)] = copied
    if spec.include_column is not None:
        source = source[
            source[spec.include_column].isin(
                cast(tuple[Any, ...], spec.include_values)
            )
        ]
    if spec.exclude_column is not None:
        values = source[spec.exclude_column]
        source = source[values.notna() & values.ne(spec.exclude_value)]
    return source, table.time_col


def _aggregate_lags(
    source: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    entity_column: str,
    seed_entity_column: str,
    source_time_column: str,
    seed_time_column: str,
    aggregation: Literal["count", "sum", "mean", "min"],
    value_column: str | None,
    days: int,
) -> pd.DataFrame:
    """Aggregate ten start-open, end-closed lag bins as float32 values."""
    source = source[
        source[entity_column].notna() & source[source_time_column].notna()
    ]
    if aggregation != "count":
        assert value_column is not None
        source = source[source[value_column].notna()]
    source = source.sort_values(
        [entity_column, source_time_column],
        kind="stable",
        ignore_index=True,
    )
    source_groups = source.groupby(entity_column, sort=False).indices
    fill = 0.0 if aggregation in {"count", "sum"} else np.nan
    output = {
        f"__kumo_arl{lag}__": np.full(len(seeds), fill, dtype=np.float32)
        for lag in range(LAG_TIMESTEPS)
    }
    width = (
        np.timedelta64(days, "D").astype("timedelta64[ns]").astype(np.int64)
    )

    for entity, seed_rows in seeds.groupby(
        seed_entity_column,
        sort=False,
    ).groups.items():
        positions = source_groups.get(entity)
        if positions is None:
            continue
        group = source.iloc[positions]
        rows = np.asarray(list(seed_rows), dtype=np.int64)
        anchors = (
            pd.to_datetime(seeds.loc[rows, seed_time_column])
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        times = (
            pd.to_datetime(group[source_time_column])
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        values = None
        prefix = None
        if aggregation != "count":
            raw = group[cast(str, value_column)]
            if pd.api.types.is_float_dtype(raw.dtype):
                values = raw.to_numpy(dtype=np.float32)
            elif pd.api.types.is_bool_dtype(raw.dtype):
                values = raw.to_numpy(dtype=np.bool_)
            elif pd.api.types.is_integer_dtype(raw.dtype):
                values = raw.to_numpy(dtype=np.int64)
            else:
                raise TypeError(f"Unsupported lag value dtype: {raw.dtype}")
            if aggregation in {"sum", "mean"}:
                cumulative = np.cumsum(values)
                prefix = np.concatenate(
                    [np.zeros(1, dtype=cumulative.dtype), cumulative]
                )

        for lag in range(LAG_TIMESTEPS):
            end = anchors - lag * width
            start = end - width
            left = np.searchsorted(times, start, side="right")
            right = np.searchsorted(times, end, side="right")
            count = right - left
            if aggregation == "count":
                result = count
            elif aggregation == "sum":
                assert prefix is not None
                result = prefix[right] - prefix[left]
            elif aggregation == "mean":
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
            output[f"__kumo_arl{lag}__"][rows] = np.asarray(
                result,
                dtype=np.float32,
            )
    return pd.DataFrame(output)


def _add_lagged_targets(
    db: relbench.base.Database,
    task: relbench.base.EntityTask,
    context: pd.DataFrame,
    query: pd.DataFrame,
    pair: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Attach Kumo v2 lagged targets to the sampled context and test query."""
    spec = LAG_SPECS.get(pair)
    if spec is None:
        return context, query, []
    seeds = pd.concat(
        [
            context[[task.entity_col, task.time_col]],
            query[[task.entity_col, task.time_col]],
        ],
        ignore_index=True,
    )
    source, source_time_column = _lag_source(db, spec)
    lagged = _aggregate_lags(
        source,
        seeds,
        entity_column=spec.entity_column,
        seed_entity_column=task.entity_col,
        source_time_column=source_time_column,
        seed_time_column=task.time_col,
        aggregation=spec.aggregation,
        value_column=spec.value_column,
        days=spec.days,
    )
    split = len(context)
    context = pd.concat(
        [context.reset_index(drop=True), lagged.iloc[:split]],
        axis=1,
    )
    query = pd.concat(
        [
            query.reset_index(drop=True),
            lagged.iloc[split:].reset_index(drop=True),
        ],
        axis=1,
    )
    return context, query, list(lagged.columns)


def run_task(
    dataset_name: str,
    task_name: str,
    num_neighbors: int,
) -> dict[str, float]:
    """Evaluate one symmetric neighbor configuration on the test split."""
    pair = (dataset_name, task_name)
    if pair not in TASKS:
        raise ValueError(f"Unsupported task: {dataset_name}/{task_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device")

    source = relarena.RelBenchDatasetTask(
        dataset_name,
        task_name,
        download=True,
    )
    task = source.task
    # This is the cleaned, test-censored DB and target-free outer test query.
    outer = source.outer_split()
    binary = task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION
    if not binary and task.task_type != relbench.base.TaskType.REGRESSION:
        raise TypeError(f"Unsupported task type: {task.task_type}")

    torch.manual_seed(SEED)
    context_df = pd.concat(
        [outer.train_table.df, outer.val_table.df],
        ignore_index=True,
    )
    keep = torch.randperm(len(context_df))[:CONTEXT_SIZE].tolist()
    context_df = context_df.iloc[keep].reset_index(drop=True)
    query_df = outer.eval_table.df.copy().reset_index(drop=True)
    if task.target_col in query_df:
        raise RuntimeError(
            "RelArena test query unexpectedly contains its target"
        )
    if binary:
        labels = sorted(context_df[task.target_col].dropna().unique())
        context_df[task.target_col] = context_df[task.target_col] == labels[-1]

    context_df, query_df, lag_columns = _add_lagged_targets(
        outer.db_state,
        task,
        context_df,
        query_df,
        pair,
    )
    lag_stypes = dict.fromkeys(lag_columns, "numerical")
    context = sdm.TableTensor.from_pandas(
        df=context_df,
        stypes={
            task.entity_col: "id",
            task.time_col: "datetime",
            task.target_col: "categorical" if binary else "numerical",
            **lag_stypes,
        },
    )
    query = sdm.TableTensor.from_pandas(
        df=query_df,
        stypes={
            task.entity_col: "id",
            task.time_col: "datetime",
            **lag_stypes,
        },
    )

    db = outer.db_state
    data = sdm.RelationalData(
        tables=_model_tables(db),
        relationships=[
            {
                "left_table": left_table,
                "left_column": left_column,
                "right_table": right_table,
                "right_column": cast(
                    str,
                    db.table_dict[right_table].pkey_col,
                ),
            }
            for left_table, table in db.table_dict.items()
            for left_column, right_table in (
                table.fkey_col_to_pkey_table.items()
            )
        ],
    )
    sampler = data.sampler(
        {
            name: table.time_col
            for name, table in db.table_dict.items()
            if table.time_col is not None
        }
    )
    sample_kwargs: dict[str, Any] = {
        "task_link": {
            "task_column": task.entity_col,
            "table": task.entity_table,
            "table_column": cast(
                str, db.table_dict[task.entity_table].pkey_col
            ),
        },
        "num_neighbors": [num_neighbors, num_neighbors],
        "task_time_column": task.time_col,
    }

    device = torch.device("cuda")
    model = sdm.models.NemotronRelational(device=device)
    sampled_context = sampler(context, **sample_kwargs).to(device)
    with torch.amp.autocast("cuda", torch.bfloat16):
        model.fit(
            x=sampled_context.task_table.drop_columns(task.target_col),
            y=sampled_context.task_table[task.target_col],
            related_tables=sampled_context.related_tables,
            num_estimators=NUM_ESTIMATORS,
        )

    predictions = []
    for batch in tqdm.tqdm(
        query.split(BATCH_SIZE),
        desc=f"{dataset_name}/{task_name} n={num_neighbors}",
    ):
        with torch.amp.autocast("cuda", torch.bfloat16):
            output = model.predict(*sampler(batch, **sample_kwargs).to(device))
        prediction = (
            output["True"].numerical.squeeze(-1)
            if binary
            else output["q500"].numerical.squeeze(-1)
        )
        predictions.append(prediction.float().cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float32, copy=False)
    if len(prediction) != len(query_df):
        raise RuntimeError("Prediction count does not match the test query")
    return {
        name: float(value)
        for name, value in task.evaluate(
            prediction,
            None,
            metrics=list(task.metrics),
        ).items()
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the import-safe command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=relarena.RELBENCH_V1_DATASETS,
        required=True,
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--num_neighbors",
        type=int,
        choices=NEIGHBOR_SWEEP,
        default=16,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one task and symmetric neighbor configuration."""
    args = build_parser().parse_args(argv)
    metrics = run_task(args.dataset, args.task, args.num_neighbors)
    for name, score in metrics.items():
        print(
            f"{args.dataset}/{args.task} n={args.num_neighbors} "
            f"{name}: {score:.4f}"
        )
    relbench.base.Dataset.get_db.cache_clear()
    relbench.tasks.get_task.cache_clear()
    relbench.datasets.get_dataset.cache_clear()


if __name__ == "__main__":
    main()
