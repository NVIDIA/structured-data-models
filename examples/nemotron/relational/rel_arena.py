"""Benchmark NemotronRelational on RelArena's RelBench v1 task grid.

Without arguments, this runs every supported binary classification and
regression task. Pass ``--dataset`` to run one dataset or both ``--dataset``
and ``--task`` to run one task.

Examples:
    python rel_arena.py
    python rel_arena.py --dataset rel-f1 --task driver-dnf
    python rel_arena.py --dataset rel-f1 --num_neighbors 32
    python rel_arena.py --dataset rel-f1 --num_neighbors 16 16

Each ``--num_neighbors`` value configures one hop: ``32`` is one hop,
``16 16`` is two hops, and ``16 16 8`` is three hops.

The default context caps match Kumo RFM for shared classification tasks;
regression and RelArena-only tasks use 5,000 rows. This follows
``rel_bench.py`` by using seeded random context samples and every test row.
Kumo's classification script instead varies sampling by task, and both Kumo
scripts cap the number of test rows.
"""

import argparse
from typing import Any, cast

import numpy as np
import pandas as pd
import relarena
import relbench
import torch
import tqdm

import sdm

CONTEXT_SIZE = {
    ("rel-amazon", "item-churn"): 10_000,
    ("rel-amazon", "item-ltv"): 5_000,
    ("rel-amazon", "user-churn"): 5_000,
    ("rel-amazon", "user-ltv"): 5_000,
    ("rel-avito", "ad-ctr"): 5_000,
    ("rel-avito", "user-clicks"): 5_000,
    ("rel-avito", "user-visits"): 5_000,
    ("rel-event", "user-attendance"): 5_000,
    ("rel-event", "user-ignore"): 3_000,
    ("rel-event", "user-repeat"): 3_000,
    ("rel-f1", "driver-dnf"): 5_000,
    ("rel-f1", "driver-position"): 5_000,
    ("rel-f1", "driver-top3"): 5_000,
    ("rel-hm", "item-sales"): 5_000,
    ("rel-hm", "user-churn"): 10_000,
    ("rel-stack", "post-votes"): 5_000,
    ("rel-stack", "user-badge"): 5_000,
    ("rel-stack", "user-engagement"): 3_000,
    ("rel-trial", "site-success"): 5_000,
    ("rel-trial", "study-adverse"): 5_000,
    ("rel-trial", "study-outcome"): 5_000,
}

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--dataset", choices=relarena.RELBENCH_V1_DATASETS)
parser.add_argument("--task")
parser.add_argument(
    "--context_size",
    type=int,
    help="override the task-specific context cap",
)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--num_neighbors", type=int, nargs="+", default=[16, 16])
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

    pair = (dataset_name, task_name)
    if pair not in CONTEXT_SIZE:
        print(f"{dataset_name}/{task_name}: skipped (no context cap)")
        return

    torch.manual_seed(args.seed)
    binary = task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION
    context_size = (
        CONTEXT_SIZE[pair] if args.context_size is None else args.context_size
    )

    # Use the full relational database, matching the existing RelBench example.
    db = task.dataset.get_db(upto_test_timestamp=False)
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

    train_df = task.get_table("train", mask_input_cols=False).df
    val_df = task.get_table("val", mask_input_cols=False).df
    test_table = task.get_table("test", mask_input_cols=False)
    context_df = pd.concat([train_df, val_df], ignore_index=True)
    if binary:
        # Normalize 0/1, Boolean, and f/t labels so AUROC scores True.
        labels = sorted(context_df[task.target_col].dropna().unique())
        context_df[task.target_col] = context_df[task.target_col] == labels[-1]
    context = sdm.TableTensor.from_pandas(
        df=context_df,
        stypes={
            task.entity_col: "id",
            task.time_col: "datetime",
            task.target_col: "categorical" if binary else "numerical",
        },
    )
    query = sdm.TableTensor.from_pandas(
        df=test_table.df[[task.entity_col, task.time_col]].copy(),
        stypes={
            task.entity_col: "id",
            task.time_col: "datetime",
        },
    )
    context = context[torch.randperm(len(context))[:context_size]]

    model = sdm.models.NemotronRelational(device=device)
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
    with torch.amp.autocast(
        device.type,
        torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        model.fit(
            x=context.drop_columns(task.target_col),
            y=context[task.target_col],
            related_tables=related_tables,
            num_estimators=args.num_estimators,
        )

    predictions = []
    for batch in tqdm.tqdm(
        query.split(args.batch_size),
        desc=f"{dataset_name}/{task_name}",
    ):
        with torch.amp.autocast(
            device.type,
            torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            out = model.predict(*sampler(batch, **kwargs).to(device))
        pred = (
            out["True"].numerical.squeeze(-1)
            if binary
            else out["q500"].numerical.squeeze(-1)
        )
        predictions.append(pred.float().cpu().numpy())

    pred = np.concatenate(predictions)
    metrics = task.evaluate(pred, test_table, metrics=list(task.metrics))
    for name, score in metrics.items():
        print(f"{dataset_name}/{task_name} {name}: {score:.4f}")


datasets = (
    [args.dataset] if args.dataset else list(relarena.RELBENCH_V1_DATASETS)
)
specs = relarena.list_entity_tasks(datasets)
if args.task:
    specs = [spec for spec in specs if spec.task == args.task]

for spec in specs:
    run_task(spec.dataset, spec.task)
    relbench.base.Dataset.get_db.cache_clear()
    relbench.tasks.get_task.cache_clear()
    relbench.datasets.get_dataset.cache_clear()
