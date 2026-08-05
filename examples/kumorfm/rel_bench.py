"""Benchmark KumoRFM on RelBench entity tasks.

Without arguments, this runs every binary-classification and regression task
in the public ``rel-*`` datasets except MIMIC-IV, which requires separate
credentials. Pass ``--dataset`` to run one dataset or both ``--dataset`` and
``--task`` to run one task. Text and unsupported multi-categorical features
are omitted from the benchmark.
"""

import argparse
from typing import cast

import pandas as pd
import relbench
import torch
import torchmetrics
import tqdm

import sdm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", choices=relbench.datasets.get_dataset_names())
parser.add_argument("--task")
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--num_neighbors", type=int, default=16)
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.task and not args.dataset:
    parser.error("'--task' requires '--dataset'")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_task(dataset_name: str, task_name: str) -> None:
    task = relbench.tasks.get_task(dataset_name, task_name, download=True)
    if not isinstance(task, relbench.base.EntityTask):
        print(f"{dataset_name}/{task_name}: skipped (not an entity task)")
        return
    if task.task_type not in {
        relbench.base.TaskType.BINARY_CLASSIFICATION,
        relbench.base.TaskType.REGRESSION,
    }:
        print(f"{dataset_name}/{task_name}: skipped ({task.task_type.value})")
        return

    torch.manual_seed(args.seed)
    classification = (
        task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION
    )

    # Task-owned DB removes autocomplete leakage and adds any required row key.
    db = task.dataset.get_db(upto_test_timestamp=False)
    tables = {}
    for name, table in db.table_dict.items():
        id_columns = {
            table.pkey_col,
            *table.fkey_col_to_pkey_table,
        }
        stypes = {}
        for column in table.df:
            try:
                stype = sdm.infer_stypes(
                    table.df[[column]].head(1000),
                    overrides={column: "id"} if column in id_columns else None,
                    allow_text=True,
                )[column]
            except TypeError:
                print(f"{name}.{column}: skipped (unsupported type)")
                continue
            if stype == sdm.Stype.text:
                print(f"{name}.{column}: skipped (text)")
                continue
            stypes[column] = stype
        tables[name] = sdm.TableTensor.from_pandas(
            df=table.df[list(stypes)],
            stypes=stypes,
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
    sampler = data.sampler(
        temporal=(
            sdm.TemporalSamplingConfig(
                time_columns=time_columns,
                strategy="last",
            )
            if time_columns
            else None
        ),
    )

    dfs = [
        task.get_table(split, mask_input_cols=False).df
        for split in ["train", "val", "test"]
    ]
    task_df = pd.concat(dfs, ignore_index=True)
    if classification:
        # Normalize 0/1, Boolean, and f/t labels so AUROC scores True.
        labels = sorted(
            pd.concat(dfs[:2], ignore_index=True)[task.target_col]
            .dropna()
            .unique()
        )
        task_df[task.target_col] = task_df[task.target_col] == labels[-1]
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

    model = sdm.models.KumoRFM(device=device)
    kwargs = {
        "task_link": {
            "task_column": task.entity_col,
            "table": task.entity_table,
            "table_column": cast(
                str, db.table_dict[task.entity_table].pkey_col
            ),
        },
        "num_neighbors": [args.num_neighbors] * 2,
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
        if classification:
            pred, target = sdm.evaluation.to_binary_class(
                out,
                y_query,
                positive_class=True,
            )
        else:
            pred = out["q500"].numerical.squeeze(-1)
            target = y_query.numerical.squeeze(-1)
        predictions.append(pred.cpu())
        targets.append(target.cpu())

    name = "AUROC" if classification else "MAE"
    pred = torch.cat(predictions)
    target = torch.cat(targets)
    score = (
        torchmetrics.classification.BinaryAUROC()(pred, target)
        if classification
        else torchmetrics.regression.MeanAbsoluteError()(pred, target)
    )
    print(f"{dataset_name}/{task_name} {name}: {score:.4f}")


datasets = (
    [args.dataset]
    if args.dataset
    else sorted(
        name
        for name in relbench.datasets.get_dataset_names()
        if name.startswith("rel-") and name not in {"rel-mimic", "rel-salt"}
    )
)
for dataset_name in datasets:
    task_names = (
        [args.task]
        if args.task
        else sorted(relbench.tasks.get_task_names(dataset_name))
    )
    for task_name in task_names:
        run_task(dataset_name, task_name)
        relbench.base.Dataset.get_db.cache_clear()
        relbench.tasks.get_task.cache_clear()
        relbench.datasets.get_dataset.cache_clear()
