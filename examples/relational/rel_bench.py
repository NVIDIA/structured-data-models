# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
from typing import Any, cast

import numpy as np
import pandas as pd
import relbench
import torch
import torchmetrics
from relbench.base import EntityTask
from tqdm import tqdm

import sdm


def get_task_table(
    df: pd.DataFrame,
    history: pd.DataFrame,
    task: EntityTask,
    num_lags: int,
) -> sdm.TableTensor:
    """Add target lags from strictly earlier observations per entity."""
    target_stype = (
        "numerical"
        if task.task_type == relbench.base.TaskType.REGRESSION
        else "categorical"
    )
    stypes = {
        task.entity_col: "id",
        task.time_col: "datetime",
        task.target_col: target_stype,
    }
    history_time_col = "__history_time__"
    lookup_time_col = "__lookup_time__"
    row_col = "__row__"

    out = df.copy()
    right = history[[task.entity_col, task.time_col, task.target_col]]
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
        stypes[lag_col] = target_stype
        if target_stype == "numerical":
            out[lag_col] = np.nan
        else:
            out[lag_col] = pd.Series(pd.NA, index=out.index, dtype="object")

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
        rows = merged.loc[mask, row_col].to_numpy()
        out.loc[out.index[rows], lag_col] = merged.loc[
            mask, task.target_col
        ].to_numpy()
        left = merged.loc[mask, [task.entity_col, row_col, history_time_col]]
        left = left.rename(columns={history_time_col: lookup_time_col})

    return sdm.TableTensor.from_pandas(df=out, stypes=stypes)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--max_test_steps", type=int, default=None)
parser.add_argument("--num_neighbors", type=int, nargs="*", default=[16, 16])
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--num_lags", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collect Relational Data #####################################################
dataset = relbench.load_dataset(args.dataset)
task = dataset.load_task(args.task)
db = task.get_db(upto_test_timestamp=False)
data = sdm.RelationalData(
    tables={
        name: sdm.TableTensor.from_pandas(
            df=table.df,
            stypes=sdm.infer_stypes(
                table.df.head(10_000),
                overrides={
                    cast(str, table.pkey_col): "id",
                    **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
                },
                text="drop",
                unsupported="drop",
            ),
        )
        for name, table in db.table_dict.items()
    },
    relationships=[
        {
            "left_table": left_table,
            "left_column": left_column,
            "right_table": right_table,
            "right_column": cast(str, db.table_dict[right_table].pkey_col),
        }
        for left_table, table in db.table_dict.items()
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
    ],
)
sampler = data.sampler(
    time_columns={
        name: table.time_col
        for name, table in db.table_dict.items()
        if table.time_col is not None
    }
)

# Collect Task Table ##########################################################
if task.task_type not in {
    relbench.base.TaskType.BINARY_CLASSIFICATION,
    relbench.base.TaskType.MULTICLASS_CLASSIFICATION,
    relbench.base.TaskType.REGRESSION,
}:
    quit(f"Unsupported task {task.task_type.value!r}")

dfs = [
    task.get_table(split, mask_input_cols=False).df
    for split in ["train", "val", "test"]
]
df = pd.concat(dfs, ignore_index=True)
if task.task_type == relbench.base.Task.REGRESSION:
    target_stype = "numerical"
else:
    target_stype = "categorical"

# Add lag target features to task table:
history = df.rename(columns={task.time_col: "__history_time__"})
history = history.sort_values(["__history_time__", task.entity_col])
lookup = pd.DataFrame(
    {
        task.entity_col: df[task.entity_col].to_numpy(),
        "__lookup_time__": df[task.time_col].to_numpy(),
        "__row__": np.arange(len(df)),
    }
)
for lag in range(1, args.num_lags + 1):
    column = f"{task.target_col}_lag_{lag}"
    if target_stype == "numerical":
        df[column] = np.nan
    else:
        df[column] = pd.Series(pd.NA, index=df.index, dtype="object")

    lookup = lookup.sort_values(["__lookup_time__", task.entity_col])
    merged = pd.merge_asof(
        lookup,
        history,
        by=task.entity_col,
        left_on="__lookup_time__",
        right_on="__history_time__",
        direction="backward",
        allow_exact_matches=False,
    )
    mask = merged["__history_time__"].notna()
    rows = merged.loc[mask, "__row__"].to_numpy()
    df.loc[rows, column] = merged.loc[mask, task.target_col].to_numpy()

    # Find the next lag strictly before the previous observation.
    lookup = merged.loc[
        mask, [task.entity_col, "__row__", "__history_time__"]
    ].rename(columns={"__history_time__": "__lookup_time__"})

task_table = get_task_table(
    df=pd.concat(dfs, ignore_index=True),
    # Use the full context history before subsampling; never use test targets.
    history=pd.concat(dfs, ignore_index=True),
    task=task,
    num_lags=args.num_lags,
)
context, query = task_table.split([len(dfs[0]) + len(dfs[1]), len(dfs[2])])

num_estimators = args.num_estimators
if len(context) > args.context_size:  # Sample different context per estimator:
    repeats = math.ceil(args.context_size * num_estimators / len(context))
    perm = torch.cat([torch.randperm(len(context)) for _ in range(repeats)])
    context = context[perm[: args.context_size * num_estimators]]
    if num_estimators > 1:
        context = context.unflatten(0, (num_estimators, args.context_size))
        query = query.expand(num_estimators, *query.size())
        num_estimators = None

# Execute Model ###############################################################
model = sdm.models.KumoRelational(device=device)
kwargs: dict[str, Any] = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
    },
    "num_neighbors": args.num_neighbors,
    "task_time_column": task.time_col,
}

context, related_tables = sampler(context, **kwargs).to(device)
with torch.amp.autocast(device.type, torch.float16, enabled=True):
    model.fit(
        x=context.drop_columns(task.target_col),
        y=context[task.target_col],
        related_tables=related_tables,
        num_estimators=num_estimators,
    )

if task.task_type == relbench.base.TaskType.REGRESSION:
    metric = torchmetrics.regression.MeanAbsoluteError()
elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
    metric = torchmetrics.classification.BinaryAUROC()
    positive_class = dfs[0][task.target_col].unique()[-1].item()
else:
    metric = torchmetrics.classification.MulticlassAccuracy(average="micro")
metric = metric.to(device)
for batch in tqdm(query.split(args.batch_size, -2)[: args.max_test_steps]):
    x_query = batch.drop_columns(task.target_col)
    y_query = batch[task.target_col].to(device)
    if num_estimators is None:
        y_query = y_query[0]
    with torch.amp.autocast(device.type, torch.float16, enabled=True):
        out = model.predict(*sampler(x_query, **kwargs).to(device))

    if task.task_type == relbench.base.TaskType.REGRESSION:
        pred = out["q500"].numerical.squeeze(-1)  # Median prediction.
        target = y_query.numerical.squeeze(-1)
    elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
        pred, target = sdm.evaluation.to_binary_class(
            out, y_query, positive_class
        )
    else:
        pred, target = sdm.evaluation.to_class_indices(out, y_query)
    metric.update(pred, target)

if task.task_type == relbench.base.TaskType.REGRESSION:
    print(f"MAE: {metric.compute():.4f}")
elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
    print(f"AUROC: {metric.compute():.4f}")
else:
    print(f"ACC: {metric.compute():.4f}")
