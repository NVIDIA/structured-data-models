import argparse
from collections.abc import Sequence
from typing import cast

import pandas as pd
import torch
from relbench.base import Dataset
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from sdm import (
    RelationalData,
    Stype,
    TableTensor,
    TemporalSamplingConfig,
    infer_stypes,
)
from sdm.models import KumoRFM

SALT_DATASET = "rel-salt"
SALT_PRESETS = {
    "item-plant": ([32, 32, 8], 500),
    "item-shippoint": ([32, 32, 8], 500),
    "item-incoterms": ([32, 32, 8], 500),
    "sales-office": ([64, 64, 8], 1_000),
    "sales-group": ([64, 64, 8], 1_000),
    "sales-payterms": ([64, 64, 8], 1_000),
    "sales-shipcond": ([64, 64, 8], 1_000),
    "sales-incoterms": ([64, 64, 8], 1_000),
}

parser = argparse.ArgumentParser(
    description="Benchmark KumoRFM on every RelBench SALT task"
)
parser.add_argument("--task", choices=SALT_PRESETS)
parser.add_argument("--context_size", type=int, default=1_000)
parser.add_argument("--num_neighbors", type=int)
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_task(task_name: str) -> None:
    """Evaluate one SALT task."""
    torch.manual_seed(args.seed)
    task = get_task(SALT_DATASET, task_name, download=True)
    db = get_dataset(SALT_DATASET, download=True).get_db(
        upto_test_timestamp=False
    )
    data = RelationalData(
        tables={
            name: TableTensor.from_pandas(
                df=table.df,
                stypes=infer_stypes(
                    table.df,
                    overrides={
                        column: "id"
                        for column in (
                            table.pkey_col,
                            *table.fkey_col_to_pkey_table,
                        )
                        if column is not None
                    },
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

    frames = [
        task.get_table(split, mask_input_cols=False).df
        for split in ["train", "val", "test"]
    ]
    task_table = TableTensor.from_pandas(
        df=pd.concat(frames, ignore_index=True),
        stypes={
            task.entity_col: "id",
            task.time_col: "datetime",
            task.target_col: "categorical",
        },
    )
    train_end = len(frames[0])
    val_end = train_end + len(frames[1])
    task_tables: Sequence[TableTensor] = (
        task_table[:train_end],
        task_table[train_end:val_end],
        task_table[val_end:],
    )
    class_to_index = {
        str(value): index
        for index, value in enumerate(
            task_table.categorical.categories[0].tolist()
        )
    }
    context = torch.cat(task_tables[:2], dim=0)
    perm = torch.randperm(len(context))[: args.context_size]
    context = cast(TableTensor, context[perm])
    num_neighbors = SALT_PRESETS[task_name][0]
    if args.num_neighbors is not None:
        num_neighbors = [args.num_neighbors] * 2 + num_neighbors[2:]
    kwargs = {
        "task_link": {
            "task_column": task.entity_col,
            "table": task.entity_table,
            "table_column": cast(
                str, db.table_dict[task.entity_table].pkey_col
            ),
        },
        "num_neighbors": num_neighbors,
        "task_time_column": task.time_col,
    }
    context, related_context = sampler(context, **kwargs).to(device)
    x_context = context.drop_columns(task.target_col)
    y_context = context[task.target_col]

    model = KumoRFM(device=device)
    metric = MulticlassAccuracy(
        num_classes=cast(int, task.num_classes),
        average="micro",
    ).to(device)
    batch_size = SALT_PRESETS[task_name][1]
    for query in tqdm(
        task_tables[-1].split(batch_size),
        desc=f"{SALT_DATASET}/{task_name}",
    ):
        y_query = query[task.target_col].to(device)
        query, related_query = sampler(
            query.drop_columns(task.target_col),
            **kwargs,
        ).to(device)
        out = model(
            x_context=x_context,
            y_context=y_context,
            x_query=query,
            related_context_tables=related_context,
            related_query_tables=related_query,
            num_estimators=args.num_estimators,
        )
        class_indices = torch.tensor(
            [
                class_to_index[column]
                for column in out.columns[Stype.numerical]
            ],
            device=device,
        )
        pred = class_indices[out.numerical.argmax(dim=-1)]
        metric.update(pred.view(-1), y_query.categorical.code.view(-1))

    print(f"{SALT_DATASET}/{task_name} accuracy: {metric.compute():.4f}")
    model.clear()


task_names = [args.task] if args.task else SALT_PRESETS
for task_name in task_names:
    run_task(task_name)
    Dataset.get_db.cache_clear()
    get_task.cache_clear()
    get_dataset.cache_clear()
