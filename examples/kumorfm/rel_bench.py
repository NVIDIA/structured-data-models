import argparse
from collections.abc import Sequence
from typing import cast

import relbench
import torch
from relbench.datasets import get_dataset
from relbench.tasks import get_task
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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collect Relational Data #####################################################
db = get_dataset(args.dataset, download=True).get_db(upto_test_timestamp=False)
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
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
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

# Collect Task Table ##########################################################
task_tables: Sequence[TableTensor] = []
task = get_task(args.dataset, args.task, download=True)
for split in ["train", "val", "test"]:
    task_table = TableTensor.from_pandas(
        df=task.get_table(split, mask_input_cols=False).df,
        stypes={
            task.entity_col: "id",
            task.time_col: "datetime",
            task.target_col: "numerical"
            if task.task_type == relbench.base.TaskType.REGRESSION
            else "categorical",
        },
    )
    task_tables.append(task_table)

context = torch.cat(task_tables[:2], dim=0)
perm = torch.randperm(len(context))[: args.context_size]
context = cast(TableTensor, context[perm])

# Execute Model ###############################################################
model = KumoRFM(device=device)

kwargs = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
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

if context.stype(task.target_col) == "categorical":
    metric = BinaryAUROC().to(device)
else:
    metric = MeanAbsoluteError().to(device)
for query in tqdm(task_tables[-1].split(args.batch_size)):
    x_query = query.drop_columns(task.target_col)
    y_query = query[task.target_col].to(device)
    out = model.predict(*sampler(x_query, **kwargs).to(device))
    if y_query.stype(task.target_col).value == "categorical":
        out = out["1"].as_tensor().view(-1)  # Positive class.
        y_query = y_query.as_tensor().decode().view(-1)  # Decode ground-truth.
    else:
        out = out["q500"].as_tensor().view(-1)  # Median prediction.
        y_query = y_query.as_tensor().view(-1)
    metric.update(out, y_query)
if context.stype(task.target_col) == "categorical":
    print(f"AUROC: {metric.compute():.4f}")
else:
    print(f"MAE: {metric.compute():.4f}")
