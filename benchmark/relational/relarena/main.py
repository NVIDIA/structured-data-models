# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run SDM KumoRelational on RelArena."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from relarena.model import RelArenaModel
from relarena.registry import register_model
from relarena.search_space import SearchSpace, TaskStats
from relbench.base import Database, EntityTask, Table, TaskType

from benchmark.relational.relarena.preprocessing import (
    build_dfs,
    context_indices,
    label_table_to_tensor,
    prepare_bundle,
    sampler_kwargs,
)
from benchmark.relational.relarena.recipe import (
    recipe_by_name,
    recipe_callbacks,
)
from sdm import Stype, TableTensor
from sdm.evaluation import to_binary_class
from sdm.models import KumoRelational

_PREDICT_BATCH_SIZE = 1_000


def _config(
    *,
    context_size: int,
    context_strategy: str,
    num_hops: int,
    num_neighbors: list[int],
    recipe: str,
    num_estimators: int = 8,
    per_estimator_context: bool = True,
    pool_inflation: float = 4.0,
) -> dict[str, Any]:
    return {
        "context_size": context_size,
        "context_strategy": context_strategy,
        "num_hops": num_hops,
        "num_neighbors": num_neighbors,
        "recipe": recipe,
        "num_estimators": num_estimators,
        "per_estimator_context": per_estimator_context,
        "pool_inflation": pool_inflation,
        "predict_batch_size": _PREDICT_BATCH_SIZE,
    }


def _targeted_grid(stats: TaskStats) -> list[dict[str, Any]]:
    if stats.num_train_nodes < 10_000:
        configs = [
            (10_000, "soft_pool", 2, [32, 32], "robust", 8, True, 4.0),
            (10_000, "hard_pool", 2, [10, 10], "robust", 32, False, 1.5),
            (1_000, "hard_pool", 2, [32, 32], "quantile", 8, False, 1.5),
            (
                1_000,
                "hard_pool",
                2,
                [32, 32],
                "softmax_mean_or_4way_standardize",
                8,
                False,
                1.5,
            ),
        ]
        if stats.task_type == TaskType.BINARY_CLASSIFICATION:
            configs.append(
                (
                    1_000,
                    "soft_pool",
                    0,
                    [],
                    "cls_identity_or_reg_base",
                    8,
                    True,
                    4.0,
                )
            )
    elif stats.num_train_nodes < 100_000:
        configs = [
            (10_000, "hard_pool", 2, [32, 32], "base", 8, False, 1.0),
            (20_000, "soft_pool", 2, [32, 32], "base", 8, True, 4.0),
            (
                10_000,
                "soft_pool",
                0,
                [],
                "cls_base_or_reg_quantile",
                8,
                False,
                1.5,
            ),
            (10_000, "soft_pool", 2, [32, 32], "base", 8, False, 1.0),
            (10_000, "soft_pool", 2, [2, 2], "robust", 8, True, 4.0),
            (5_000, "hard_pool", 0, [], "base", 8, True, 4.0),
        ]
        if (
            stats.num_train_nodes < 15_000
            and stats.task_type == TaskType.BINARY_CLASSIFICATION
        ):
            configs.append(
                (
                    1_000,
                    "soft_pool",
                    0,
                    [],
                    "cls_identity_or_reg_base",
                    8,
                    True,
                    4.0,
                )
            )
    else:
        configs = [
            (10_000, "soft_pool", 0, [], "robust", 8, True, 4.0),
            (10_000, "hard_pool", 0, [], "best_hard", 8, True, 4.0),
            (5_000, "soft_pool", 0, [], "base", 8, True, 4.0),
            (20_000, "soft_pool", 2, [], "base", 8, True, 4.0),
        ]

    return [
        _config(
            context_size=context_size,
            context_strategy=context_strategy,
            num_hops=num_hops,
            num_neighbors=num_neighbors,
            recipe=recipe,
            num_estimators=num_estimators,
            per_estimator_context=per_estimator_context,
            pool_inflation=pool_inflation,
        )
        for (
            context_size,
            context_strategy,
            num_hops,
            num_neighbors,
            recipe,
            num_estimators,
            per_estimator_context,
            pool_inflation,
        ) in configs
    ]


def search_space(stats: TaskStats) -> SearchSpace:
    """Return the targeted DFS and linked-history grid."""
    grid = _targeted_grid(stats)
    return SearchSpace(default_overrides=grid[0], fixed_grid=grid)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _temporal_strategy(task: EntityTask) -> str:
    if task.task_type == TaskType.REGRESSION:
        return "last"
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return "uniform"
    raise ValueError(f"unsupported task type: {task.task_type}")


def _positive_class(train_table: Table, target_col: str) -> Any:
    series = train_table.df[target_col].dropna()
    values = series.unique()
    if len(values) == 0:
        raise ValueError(f"no training labels in column {target_col!r}")
    if series.dtype == bool or values.dtype == bool:
        return True
    classes = np.asarray(values).tolist()
    if any(isinstance(value, (bool, np.bool_)) for value in classes):
        return True
    if 1 in classes:
        return 1
    if True in classes:
        return True
    return values[-1]


def _has_column(table: TableTensor, column: str) -> bool:
    return any(column in columns for columns in table.columns.values())


def _drop_target(table: TableTensor, target_col: str) -> TableTensor:
    if _has_column(table, target_col):
        return table.drop_columns(target_col)
    return table


def _regression_chunk(out: TableTensor) -> np.ndarray:
    columns = out.columns.get(Stype.numerical, ())
    tensor = out["q500"].numerical if "q500" in columns else out.numerical
    return tensor.squeeze(-1).detach().cpu().numpy()


@register_model(search_space=search_space)
class KumoRelationalModel(RelArenaModel):
    """RelArena adapter for the SDM relational foundation model."""

    name = "kumo-relational"
    refit_on_full_data = True

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        """Prepare relational context and fit the in-context model."""
        del val_table, time_limit

        device = _device()
        bundle = prepare_bundle(task, db, train_table)
        sampler = bundle.data.sampler(time_columns=bundle.time_columns)
        kwargs = sampler_kwargs(
            task,
            db,
            num_neighbors=list(self.config["num_neighbors"]),
            temporal_strategy=_temporal_strategy(task),
        )
        dfs = build_dfs(task, db, train_table, train_table)
        context = label_table_to_tensor(task, train_table, dfs)

        num_estimators = int(self.config["num_estimators"])
        keep = context_indices(
            len(context),
            int(self.config["context_size"]),
            seed,
            strategy=str(self.config["context_strategy"]),
            context_time=(
                train_table.df[task.time_col].to_numpy()
                if task.time_col is not None
                else None
            ),
            pool_inflation=float(self.config["pool_inflation"]),
            num_estimators=(
                num_estimators
                if self.config["per_estimator_context"]
                else None
            ),
        )
        if keep is not None:
            context = context[torch.as_tensor(keep, dtype=torch.long)]

        sampled = sampler(context, **kwargs).to(device)
        members = num_estimators if context.dim() == 3 else None
        recipe_name = str(self.config["recipe"])
        model = KumoRelational(device=device)
        with torch.amp.autocast(
            device,
            torch.float16,
            enabled=device == "cuda",
        ):
            model.fit(
                x=sampled.task_table.drop_columns(task.target_col),
                y=sampled.task_table[task.target_col],
                related_tables=sampled.related_tables,
                recipe=recipe_by_name(recipe_name, task.task_type),
                num_estimators=None if members else num_estimators,
                num_hops=int(self.config["num_hops"]),
                generator=_generator(device, seed),
            )

        self._model = model
        self._history = train_table
        self._dfs_stypes = dfs.stypes
        self._sampler_kwargs = kwargs
        self._context_members = members
        self._device = device
        self._callbacks = recipe_callbacks(recipe_name, task.task_type)
        self._positive_class = (
            _positive_class(train_table, task.target_col)
            if task.task_type == TaskType.BINARY_CLASSIFICATION
            else None
        )

    def predict(
        self,
        task: EntityTask,
        db: Database,
        table: Table,
    ) -> np.ndarray:
        """Predict a RelBench label table in relational batches."""
        bundle = prepare_bundle(task, db, self._history)
        sampler = bundle.data.sampler(time_columns=bundle.time_columns)
        dfs = build_dfs(
            task,
            db,
            table,
            self._history,
            stypes=self._dfs_stypes,
        )
        query = label_table_to_tensor(task, table, dfs)
        members = self._context_members
        if members is None:
            batches = query.split(int(self.config["predict_batch_size"]))
        else:
            query = query.expand(members, *query.size())
            batches = query.split(
                int(self.config["predict_batch_size"]),
                dim=-2,
            )

        predictions = []
        for batch in batches:
            sampled = sampler(batch, **self._sampler_kwargs).to(self._device)
            task_rows = sampled.task_table
            if members is not None:
                task_rows = task_rows[0]
            with torch.amp.autocast(
                self._device,
                torch.float16,
                enabled=self._device == "cuda",
            ):
                out = self._model.predict(
                    _drop_target(sampled.task_table, task.target_col),
                    sampled.related_tables,
                    callbacks=self._callbacks or None,
                )

            if task.task_type == TaskType.REGRESSION:
                chunk = _regression_chunk(out)
            elif task.task_type == TaskType.BINARY_CLASSIFICATION:
                if _has_column(task_rows, task.target_col):
                    chunk, _ = to_binary_class(
                        out,
                        task_rows[task.target_col],
                        positive_class=self._positive_class,
                    )
                    chunk = chunk.detach().cpu().numpy()
                else:
                    chunk = (
                        out[str(self._positive_class)]
                        .numerical.squeeze(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            else:
                raise ValueError(f"unsupported task type: {task.task_type}")
            predictions.append(np.asarray(chunk, dtype=float))

        return np.concatenate(predictions)


if __name__ == "__main__":
    from relarena.cli import main

    main(["--model", KumoRelationalModel.name, *sys.argv[1:]])
