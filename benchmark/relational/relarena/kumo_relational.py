# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from functools import lru_cache
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
import tqdm
from relarena_core.model import RelArenaModel
from relarena_core.registry import register_model
from relarena_core.search_space import SearchSpace, TaskStats
from relbench.base import Database, EntityTask, Table, TaskType

import sdm


def search_space(stats: TaskStats) -> SearchSpace:
    return SearchSpace(
        default_overrides={},
        fixed_grid=[
            {
                "subgraph_size": subgraph_size,
                "context_size": context_size,
            }
            for context_size in (
                ["small"] if stats.num_train_nodes <= 20_000 else ["large"]
            )
            for subgraph_size in ["small", "medium", "large"]
        ],
    )


REGRESSION = TaskType.REGRESSION
CLASSIFICATION = TaskType.BINARY_CLASSIFICATION

CONTEXT_SIZE = {
    "small": 2_000,
    "large": 20_000,
}

NUM_NEIGHBORS = {
    # Vary number of neighbor grid search based on task type + context size:
    # Regression generally benefits from a wide range of neighbors, while
    # entity table features + lag target features is a strong baseline:
    (REGRESSION, "small"): {"small": 0, "medium": 8, "large": 64},
    (REGRESSION, "large"): {"small": 0, "medium": 8, "large": 64},
    # Classification excels with less number of neighbors. On smaller context
    # size, it can be beneficial to search over decreased number of neighbors:
    (CLASSIFICATION, "small"): {"small": 1, "medium": 4, "large": 16},
    (CLASSIFICATION, "large"): {"small": 1, "medium": 8, "large": 32},
}


@lru_cache(maxsize=1)
def get_task_table(
    task_table: Table,
    history_table: Table,
    task: EntityTask,
    num_lags: int,
    is_train: bool,
) -> sdm.TableTensor:
    """Return the task table with added historical target features."""
    stypes = {
        task.entity_col: "id",
        task.time_col: "datetime",
    }
    if is_train and task.task_type == REGRESSION:
        stypes[task.target_col] = "numerical"
    elif is_train:
        stypes[task.target_col] = "categorical"

    history_time_col = "__history_time__"
    lookup_time_col = "__lookup_time__"
    row_col = "__row__"

    out = task_table.df.copy()
    right = history_table.df[[task.entity_col, task.time_col, task.target_col]]
    right = right.rename(columns={task.time_col: history_time_col})
    right = right.sort_values([history_time_col, task.entity_col])
    left = pd.DataFrame(
        {
            task.entity_col: out[task.entity_col].to_numpy(),
            lookup_time_col: out[task.time_col].to_numpy(),
            row_col: np.arange(len(out)),
        }
    )

    for lag in range(1, num_lags + 1):
        lag_col = f"{task.target_col}_lag_{lag}"
        if task.task_type == REGRESSION:
            out[lag_col] = np.nan
            stypes[lag_col] = "numerical"
        else:
            out[lag_col] = pd.Series(pd.NA, index=out.index, dtype="object")
            stypes[lag_col] = "categorical"

        left = left.sort_values([lookup_time_col, task.entity_col])
        merged = pd.merge_asof(
            left,
            right,
            by=task.entity_col,
            left_on=lookup_time_col,
            right_on=history_time_col,
            direction="backward",
            allow_exact_matches=False,
        )

        mask = merged[history_time_col].notna()
        if mask.any():
            rows = merged.loc[mask, row_col].to_numpy()
            out.loc[out.index[rows], lag_col] = merged.loc[
                mask, task.target_col
            ].to_numpy()

        left = merged.loc[mask, [task.entity_col, row_col, history_time_col]]
        left = left.rename(columns={history_time_col: lookup_time_col})

    return sdm.TableTensor.from_pandas(out, stypes)


@lru_cache(maxsize=1)
def get_sampler(
    db: Database,
    text: Literal["off", "drop"],
) -> sdm.relational.RelationalSampler:
    r"""Initialize the relational sampler to gather time-aware subgraphs."""
    tables = {}
    for name, table in db.table_dict.items():
        stypes = sdm.infer_stypes(
            table.df.head(10_000),
            overrides={
                table.pkey_col: "id",
                **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
            },
            text=text,
            unsupported="drop",
        )
        if name == "drivers" and text == "drop":
            # Discard misclassified text columns:
            del stypes["forename"], stypes["surname"]
        tables[name] = sdm.TableTensor.from_pandas(table.df, stypes)

    relationships = [
        {
            "left_table": name,
            "left_column": column,
            "right_table": other,
            "right_column": db.table_dict[other].pkey_col,
        }
        for name, table in db.table_dict.items()
        for column, other in table.fkey_col_to_pkey_table.items()
    ]

    return sdm.RelationalData(tables, relationships).sampler(
        time_columns={
            name: table.time_col
            for name, table in db.table_dict.items()
            if table.time_col is not None
        }
    )


@lru_cache(maxsize=1)
def get_model(device: torch.device | str) -> sdm.models.KumoRelational:
    return sdm.models.KumoRelational(device=device)


@register_model(search_space=search_space)
class KumoRelationalModel(RelArenaModel):
    name = "kumo-relational"

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

        self.device = torch.device("cuda:0")
        cpu_generator = torch.Generator().manual_seed(seed)
        cuda_generator = torch.Generator(self.device).manual_seed(seed)

        self.history_table = train_table
        context = get_task_table(
            task_table=train_table,
            history_table=self.history_table,
            task=task,
            num_lags=20,
            is_train=True,
        )

        num_estimators = 8
        context_size = self.config["context_size"]
        subgraph_size = self.config["subgraph_size"]
        N = NUM_NEIGHBORS[(task.task_type, context_size)][subgraph_size]
        context_size = CONTEXT_SIZE[self.config["context_size"]]
        self.num_neighbors = [N] * 2

        if (
            task.task_type != REGRESSION
            and len(context) > context_size * num_estimators
        ):  # Sort by recency for classification tasks:
            perm = context.datetime.view(-1).argsort(descending=True)
            context = context[perm]
            context = context[: context_size * num_estimators]

        self.expand_query = False
        if len(context) > 1.2 * context_size:  # Differ context per estimator:
            repeats = math.ceil(context_size * num_estimators / len(context))
            perm = torch.cat(
                [
                    torch.randperm(len(context), generator=cpu_generator)
                    for _ in range(repeats)
                ]
            )
            context = context[perm[: context_size * num_estimators]]
            context = context.unflatten(0, (num_estimators, context_size))
            self.expand_query = True
            num_estimators = None
        else:
            perm = torch.randperm(len(context), generator=cpu_generator)
            context = context[perm[:context_size]]

        self.sampler = get_sampler(
            db=db,
            # Stype inference misclassifies text columns in rel-trial:
            text="off" if task.entity_table == "facilities" else "drop",
        )

        context, related_tables = self.sampler(
            context,
            task_link={
                "task_column": task.entity_col,
                "table": task.entity_table,
                "table_column": db.table_dict[task.entity_table].pkey_col,
            },
            num_neighbors=self.num_neighbors,
            task_time_column=task.time_col,
        ).to(self.device)

        self.model = get_model(self.device)
        with torch.amp.autocast(self.device.type, torch.float16):
            self.model.fit(
                x=context.drop_columns(task.target_col),
                y=context[task.target_col],
                related_tables=related_tables,
                num_estimators=num_estimators,
                generator=cuda_generator,
            )

    def predict(
        self,
        task: EntityTask,
        db: Database,
        table: Table,
    ) -> np.ndarray:

        query = get_task_table(
            task_table=table,
            history_table=self.history_table,
            task=task,
            num_lags=20,
            is_train=False,
        )
        if self.expand_query:
            query = query.expand(8, *query.size())

        outs = []
        for batch in tqdm.tqdm(query.split(10_000, dim=-2)):
            batch, related_tables = self.sampler(
                batch,
                task_link={
                    "task_column": task.entity_col,
                    "table": task.entity_table,
                    "table_column": db.table_dict[task.entity_table].pkey_col,
                },
                num_neighbors=self.num_neighbors,
                task_time_column=task.time_col,
            ).to(self.device)

            with torch.amp.autocast(self.device.type, torch.float16):
                outs.append(self.model.predict(batch, related_tables))
        out = cast(sdm.TableTensor, torch.cat(outs, dim=-2))

        if task.task_type == REGRESSION:
            out = out["q500"].numerical.squeeze(-1)
        elif "1" in out.column_names:
            out = out["1"].numerical.squeeze(-1)
        else:
            out = out["True"].numerical.squeeze(-1)

        return out.cpu().numpy()


if __name__ == "__main__":
    from relarena.cli import main

    main(["--model", KumoRelationalModel.name, *sys.argv[1:]])
