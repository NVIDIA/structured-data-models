import argparse
from typing import cast

import pandas as pd
import relbench
import torch
from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task, get_task_names
from sdm import (
    RelationalData,
    TableTensor,
    TemporalSamplingConfig,
    infer_stypes,
)
from sdm.models import KumoRFM
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import MeanAbsoluteError
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description=(
        "Benchmark KumoRFM on RelBench. Omit --dataset and --task to run "
        "every rel-* task; rel-mimic requires access credentials."
    )
)
parser.add_argument("--dataset", choices=get_dataset_names())
parser.add_argument("--task")
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.task and not args.dataset:
    parser.error("'--task' requires '--dataset'")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_task(dataset_name: str, task_name: str) -> None:
    """Evaluate one supported RelBench task."""
    task = get_task(dataset_name, task_name, download=True)
    if not isinstance(task, relbench.base.EntityTask):
        print(f"{dataset_name}/{task_name}: skipped (not an entity task)")
        return
    if task.task_type not in {
        relbench.base.TaskType.BINARY_CLASSIFICATION,
        relbench.base.TaskType.REGRESSION,
    }:
        print(f"{dataset_name}/{task_name}: skipped ({task.task_type.value})")
        return
    classification = (
        task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION
    )

    torch.manual_seed(args.seed)

    # Collect Relational Data #################################################
    db = get_dataset(dataset_name, download=True).get_db(
        upto_test_timestamp=False
    )
    data = RelationalData(
        tables={
            name: TableTensor.from_pandas(
                df=table.df,
                stypes=infer_stypes(table.df),
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
            TemporalSamplingConfig(
                time_columns=time_columns,
                strategy="last",
            )
            if time_columns
            else None
        ),
    )

    # Collect Task Table ######################################################
    dfs = [
        task.get_table(split, mask_input_cols=False).df
        for split in ["train", "val", "test"]
    ]
    task_table = TableTensor.from_pandas(
        df=pd.concat(dfs, ignore_index=True),
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

    # Execute Model ###########################################################
    model = KumoRFM(device=device)
    kwargs = {
        "task_link": {
            "task_column": task.entity_col,
            "table": task.entity_table,
            "table_column": cast(
                str, db.table_dict[task.entity_table].pkey_col
            ),
        },
        "num_neighbors": [16, 16],
        "task_time_column": task.time_col,
    }
    context, related_tables = sampler(context, **kwargs).to(device)
    model.fit(
        x=context.drop_columns(task.target_col),
        y=context[task.target_col],
        related_tables=related_tables,
        num_estimators=1,
    )

    metric = (BinaryAUROC() if classification else MeanAbsoluteError()).to(
        device
    )
    for batch in tqdm(
        query.split(args.batch_size),
        desc=f"{dataset_name}/{task_name}",
    ):
        x_query = batch.drop_columns(task.target_col)
        y_query = batch[task.target_col].to(device)
        out = model.predict(*sampler(x_query, **kwargs).to(device))
        if classification:
            out = out["1"].numerical  # Positive class.
            y_query = y_query.categorical.categories[0][
                y_query.categorical.code
            ]
        else:
            out = out["q500"].numerical  # Median prediction.
            y_query = y_query.numerical
        metric.update(out, y_query)

    if context.stype(task.target_col) == "categorical":
        print(f"{dataset_name}/{task_name} AUROC: {metric.compute():.4f}")
    else:
        print(f"{dataset_name}/{task_name} MAE: {metric.compute():.4f}")
    model.clear()


datasets = (
    [args.dataset]
    if args.dataset
    else sorted(
        name for name in get_dataset_names() if name.startswith("rel-")
    )
)
for dataset_name in datasets:
    task_names = (
        [args.task] if args.task else sorted(get_task_names(dataset_name))
    )
    for task_name in task_names:
        run_task(dataset_name, task_name)
        relbench.base.Dataset.get_db.cache_clear()
        get_task.cache_clear()
        get_dataset.cache_clear()
