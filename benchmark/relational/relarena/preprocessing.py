# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing for the RelArena KumoRelational benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from relarena.featurization import DFS_MAX_DEPTH, build_dfs_features
from relbench.base import Database, EntityTask, Table, TaskType

import sdm

DFS_DEPTH = 3
_HISTORY_TABLE = "_kumo_target_history"


@dataclass(frozen=True)
class RelationalBundle:
    """Tensorized relational data and its temporal columns."""

    data: sdm.RelationalData
    time_columns: dict[str, str]


@dataclass(frozen=True)
class DFSFeatures:
    """DFS features in anchor-row order."""

    frame: pd.DataFrame
    stypes: dict[str, str]


def _relationships(db: Database) -> list[dict[str, str]]:
    return [
        {
            "left_table": table_name,
            "left_column": column,
            "right_table": related_name,
            "right_column": cast(str, db.table_dict[related_name].pkey_col),
        }
        for table_name, table in db.table_dict.items()
        for column, related_name in table.fkey_col_to_pkey_table.items()
    ]


@lru_cache(maxsize=1)
def _base_bundle(db: Database) -> RelationalBundle:
    tables = {}
    for name, table in db.table_dict.items():
        stypes = sdm.infer_stypes(
            table.df.head(10_000),
            overrides={
                cast(str, table.pkey_col): "id",
                **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
            },
            text="drop",
            unsupported="drop",
        )
        tables[name] = sdm.TableTensor.from_pandas(
            df=table.df,
            stypes=stypes,
        )

    return RelationalBundle(
        data=sdm.RelationalData(
            tables=tables,
            relationships=_relationships(db),
        ),
        time_columns={
            name: cast(str, table.time_col)
            for name, table in db.table_dict.items()
            if table.time_col is not None
        },
    )


def prepare_bundle(
    task: EntityTask,
    db: Database,
    history: Table,
) -> RelationalBundle:
    """Return relational data with leak-safe target history attached."""
    bundle = _base_bundle(db)
    if task.time_col is None:
        return bundle

    frame = history.df[
        [task.entity_col, task.time_col, task.target_col]
    ].copy()
    frame[task.time_col] = pd.to_datetime(frame[task.time_col]) + pd.Timedelta(
        task.timedelta
    )
    target_stype = (
        "numerical" if task.task_type == TaskType.REGRESSION else "categorical"
    )
    tables = {
        **bundle.data.tables,
        _HISTORY_TABLE: sdm.TableTensor.from_pandas(
            df=frame,
            stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
                task.target_col: target_stype,
            },
        ),
    }
    relationships: list[Any] = [
        *bundle.data.relationships,
        {
            "left_table": _HISTORY_TABLE,
            "left_column": task.entity_col,
            "right_table": task.entity_table,
            "right_column": cast(
                str, db.table_dict[task.entity_table].pkey_col
            ),
        },
    ]
    return RelationalBundle(
        data=sdm.RelationalData(
            tables=tables,
            relationships=relationships,
        ),
        time_columns={
            **bundle.time_columns,
            _HISTORY_TABLE: task.time_col,
        },
    )


def build_dfs(
    task: EntityTask,
    db: Database,
    table: Table,
    history: Table,
    *,
    stypes: dict[str, str] | None = None,
) -> DFSFeatures:
    """Build DFS features and pin their schema across fit and predict."""
    features, _ = build_dfs_features(
        task,
        db,
        table,
        depth=DFS_DEPTH,
        max_depth=DFS_MAX_DEPTH,
        history_table=history,
        keep_anchor_columns=False,
    )
    if len(features) != len(table.df):
        raise ValueError(
            f"DFS produced {len(features)} rows; expected {len(table.df)}"
        )
    features = features.reset_index(drop=True)
    if stypes is None:
        stypes = {
            column: str(stype)
            for column, stype in sdm.infer_stypes(
                features.head(10_000),
                text="drop",
                unsupported="drop",
            ).items()
        }
    return DFSFeatures(
        frame=features.reindex(columns=stypes),
        stypes=stypes,
    )


def label_table_to_tensor(
    task: EntityTask,
    table: Table,
    dfs: DFSFeatures,
) -> sdm.TableTensor:
    """Append DFS features and tensorize a RelBench label table."""
    frame = pd.concat(
        [table.df.reset_index(drop=True), dfs.frame],
        axis=1,
    )
    stypes = {
        task.entity_col: "id",
        cast(str, task.time_col): "datetime",
        **dfs.stypes,
    }
    if task.target_col in frame:
        stypes[task.target_col] = (
            "numerical"
            if task.task_type == TaskType.REGRESSION
            else "categorical"
        )
    return sdm.TableTensor.from_pandas(df=frame, stypes=stypes)


def context_indices(
    num_rows: int,
    context_size: int,
    seed: int,
    *,
    strategy: str,
    context_time: np.ndarray | None,
    pool_inflation: float,
    num_estimators: int | None,
) -> np.ndarray | None:
    """Return seeded context indices, optionally per estimator."""
    if num_rows <= context_size:
        return None

    def draw(draw_seed: int) -> np.ndarray:
        if strategy in {"hard_pool", "soft_pool"} and context_time is not None:
            size = round(context_size * pool_inflation)
            if num_rows > size:
                times = pd.to_datetime(context_time).astype("int64").to_numpy()
                order = np.argsort(-times, kind="stable")
                if strategy == "hard_pool":
                    pool = order[:size]
                else:
                    ranks = np.empty(num_rows, dtype=np.int64)
                    ranks[order] = np.arange(num_rows)
                    weights = np.exp(
                        -np.log(2.0) * ranks / max(num_rows - 1, 1) / 0.1
                    )
                    rng = np.random.default_rng(draw_seed)
                    keys = np.log(rng.random(num_rows)) / weights
                    pool = np.argpartition(keys, -size)[-size:]
                return np.random.default_rng(draw_seed).choice(
                    pool,
                    size=context_size,
                    replace=False,
                )
        generator = torch.Generator().manual_seed(draw_seed)
        return torch.randperm(
            num_rows,
            generator=generator,
        )[:context_size].numpy()

    if num_estimators is None:
        return draw(seed)
    seeds = np.random.SeedSequence(seed).generate_state(num_estimators)
    return np.stack([draw(int(member_seed)) for member_seed in seeds])


def sampler_kwargs(
    task: EntityTask,
    db: Database,
    *,
    num_neighbors: list[int],
    temporal_strategy: str,
) -> dict[str, Any]:
    """Build arguments for relational neighborhood sampling."""
    return {
        "task_link": {
            "task_column": task.entity_col,
            "table": task.entity_table,
            "table_column": cast(
                str, db.table_dict[task.entity_table].pkey_col
            ),
        },
        "num_neighbors": num_neighbors,
        "task_time_column": task.time_col,
        "temporal_strategy": temporal_strategy,
    }
