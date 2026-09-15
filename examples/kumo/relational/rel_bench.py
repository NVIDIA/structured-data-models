# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark KumoRelational on RelBench entity tasks.

Without arguments, this runs every supported entity task in the public
``rel-*`` datasets except MIMIC-IV and SALT. Pass ``--dataset`` to run one
dataset or both ``--dataset`` and ``--task`` to run one task.

Examples:
    python rel_bench.py
    python rel_bench.py --dataset rel-f1 --task driver-dnf
    python rel_bench.py --dataset rel-f1 --num_neighbors 32
    python rel_bench.py --dataset rel-f1 --num_neighbors 16 16

Each ``--num_neighbors`` value configures one hop: ``32`` is one hop,
``16 16`` is two hops, and ``16 16 8`` is three hops.
"""

import argparse
from typing import Any, cast

import pandas as pd
import relbench
import torch
import torchmetrics
import tqdm

import sdm

DEFAULT_DATASETS = [
    "rel-amazon",
    "rel-arxiv",
    "rel-avito",
    "rel-event",
    "rel-f1",
    "rel-hm",
    "rel-ratebeer",
    "rel-stack",
    "rel-trial",
]

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--dataset")
parser.add_argument("--task")
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--num_neighbors", type=int, nargs="+", default=[16, 16])
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.task and not args.dataset:
    parser.error("'--task' requires '--dataset'")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_task(dataset: Any, task_name: str) -> None:
    dataset_name = str(dataset.name_or_path)
    task = dataset.load_task(task_name)
    if not isinstance(task, relbench.base.EntityTask):
        print(f"{dataset_name}/{task_name}: skipped (not an entity task)")
        return
    if task.task_type not in {
        relbench.base.TaskType.BINARY_CLASSIFICATION,
        relbench.base.TaskType.MULTICLASS_CLASSIFICATION,
        relbench.base.TaskType.REGRESSION,
    }:
        print(f"{dataset_name}/{task_name}: skipped ({task.task_type.value})")
        return

    torch.manual_seed(args.seed)
    binary = task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION
    multiclass = (
        task.task_type == relbench.base.TaskType.MULTICLASS_CLASSIFICATION
    )
    classification = binary or multiclass

    # Task-owned DB removes target leakage and adds any required row key.
    db = task.get_db(upto_test_timestamp=False)
    tables = {}
    for name, table in db.table_dict.items():
        tables[name] = sdm.TableTensor.from_pandas(
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

    data = sdm.RelationalData(
        tables=tables,
        relationships=[
            {
                "left_table": left_table,
                "left_column": left_column,
                "right_table": right_table,
                "right_column": cast(str, db.table_dict[right_table].pkey_col),
            }
            for left_table, table in db.table_dict.items()
            for left_column, right_table in (
                table.fkey_col_to_pkey_table.items()
            )
        ],
    )
    time_columns = {
        name: table.time_col
        for name, table in db.table_dict.items()
        if table.time_col is not None
    }
    sampler = data.sampler(time_columns)

    dfs = [
        task.get_table(split, mask_input_cols=False).df
        for split in ["train", "val", "test"]
    ]
    context_df = pd.concat(dfs[:2], ignore_index=True)
    task_df = pd.concat(dfs, ignore_index=True)
    if binary:
        # Normalize 0/1, Boolean, and f/t labels so AUROC scores True.
        labels = sorted(context_df[task.target_col].dropna().unique())
        task_df[task.target_col] = task_df[task.target_col] == labels[-1]
    # Multiclass targets are zero-based IDs; preserve gaps without test labels.
    num_classes = (
        int(context_df[task.target_col].max()) + 1 if multiclass else 0
    )
    task_table = sdm.TableTensor.from_pandas(
        df=task_df,
        stypes={
            task.entity_col: "id",
            task.time_col: "datetime",
            task.target_col: (
                "categorical" if classification else "numerical"
            ),
        },
    )
    context, query = task_table.split([len(dfs[0]) + len(dfs[1]), len(dfs[2])])
    context = context[torch.randperm(len(context))[: args.context_size]]

    model = sdm.models.KumoRelational(device=device)
    kwargs: dict[str, Any] = {
        "task_link": {
            "task_column": task.entity_col,
            "table": task.entity_table,
            "table_column": cast(
                str, db.table_dict[task.entity_table].pkey_col
            ),
        },
        "num_neighbors": args.num_neighbors,
        "task_time_column": task.time_col,
    }
    context, related_tables = sampler(context, **kwargs).to(device)
    with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
        model.fit(
            x=context.drop_columns(task.target_col),
            y=context[task.target_col],
            related_tables=related_tables,
            num_estimators=args.num_estimators,
        )

    predictions = []
    targets = []
    for batch in tqdm.tqdm(
        query.split(args.batch_size),
        desc=f"{dataset_name}/{task_name}",
    ):
        y_query = batch[task.target_col].to(device)
        with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
            out = model.predict(
                *sampler(
                    batch.drop_columns(task.target_col),
                    **kwargs,
                ).to(device)
            )
        if binary:
            pred, target = sdm.evaluation.to_binary_class(
                out,
                y_query,
                positive_class=True,
            )
        elif multiclass:
            pred = out.numerical
            class_ids = torch.tensor(
                [int(column) for column in out.columns[sdm.Stype.numerical]],
                dtype=torch.long,
                device=pred.device,
            )
            scores = pred.new_zeros(pred.size(0), num_classes)
            pred = scores.index_copy(-1, class_ids, pred)
        else:
            pred = out["q500"].numerical.squeeze(-1)
            target = y_query.numerical.squeeze(-1)
        predictions.append(pred.cpu())
        if not multiclass:
            targets.append(target.cpu())

    pred = torch.cat(predictions)
    if multiclass:
        for name, score in task.evaluate(pred.numpy()).items():
            print(f"{dataset_name}/{task_name} {name}: {score:.4f}")
        return

    name = "AUROC" if binary else "MAE"
    target = torch.cat(targets)
    score = (
        torchmetrics.classification.BinaryAUROC()(pred, target)
        if binary
        else torchmetrics.regression.MeanAbsoluteError()(pred, target)
    )
    print(f"{dataset_name}/{task_name} {name}: {score:.4f}")


datasets = [args.dataset] if args.dataset else DEFAULT_DATASETS
for dataset_name in datasets:
    dataset = relbench.load_dataset(dataset_name)
    task_names = [args.task] if args.task else sorted(dataset.get_task_names())
    for task_name in task_names:
        run_task(dataset, task_name)
