import argparse
from typing import cast

import pandas as pd
import relbench
import torch
import torchmetrics
import tqdm

import sdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def peak():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_host_memory_stats()
    torch.cuda.synchronize(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    print(f"Allocated: {peak_alloc / 1000**2:.1f}MB")
    print(f"Reserved: {peak_reserved / 1000**2:.1f}MB")


peak()


# Collect Relational Data #####################################################
dataset = relbench.datasets.get_dataset(args.dataset, download=True)
db = dataset.get_db(upto_test_timestamp=False)
data = sdm.RelationalData(
    tables={
        name: sdm.TableTensor.from_pandas(
            df=table.df,
            stypes=sdm.infer_stypes(table.df),
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
time_columns = {
    name: table.time_col
    for name, table in db.table_dict.items()
    if table.time_col is not None
}
print("Relational Data")
peak()
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
print("Sampler Init")
peak()

# Collect Task Table ##########################################################
task = relbench.tasks.get_task(args.dataset, args.task, download=True)
dfs = [
    task.get_table(split, mask_input_cols=False).df
    for split in ["train", "val", "test"]
]
task_table = sdm.TableTensor.from_pandas(
    df=pd.concat(dfs, ignore_index=True),
    stypes={
        task.entity_col: "id",
        task.time_col: "datetime",
        task.target_col: "numerical"
        if task.task_type == relbench.base.TaskType.REGRESSION
        else "categorical",
    },
)
context, query = task_table.split([len(dfs[0]) + len(dfs[1]), len(dfs[2])])
context = context[torch.randperm(len(context))[: args.context_size]]
print("Task Table")
peak()

# Execute Model ###############################################################
model = sdm.models.KumoRFM(device=device)
print("Model")
peak()

kwargs = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
    },
    "num_neighbors": [64, 64],
    "task_time_column": task.time_col,
}
context, related_tables = sampler(context, **kwargs).to(device)
print("Before fit + Sampler")
peak()
with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
    model.fit(
        x=context.drop_columns(task.target_col),
        y=context[task.target_col],
        related_tables=related_tables,
        num_estimators=1,
    )
print("After fit")
peak()

if task.task_type == relbench.base.TaskType.REGRESSION:
    metric = torchmetrics.regression.MeanAbsoluteError().to(device)
else:
    metric = torchmetrics.classification.BinaryAUROC().to(device)
for batch in tqdm.tqdm(query.split(args.batch_size)):
    x_query = batch.drop_columns(task.target_col)
    y_query = batch[task.target_col].to(device)
    with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
        out = model.predict(*sampler(x_query, **kwargs).to(device))
    if task.task_type == relbench.base.TaskType.REGRESSION:
        pred = out["q500"].numerical.squeeze(-1)  # Median prediction.
        target = y_query.numerical.squeeze(-1)
    else:
        pred, target = sdm.evaluation.to_binary_class(
            out, y_query, positive_class=1
        )
    metric.update(pred, target)
if task.task_type == relbench.base.TaskType.REGRESSION:
    print(f"MAE: {metric.compute():.4f}")
else:
    print(f"AUROC: {metric.compute():.4f}")
