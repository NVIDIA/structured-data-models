import argparse
from collections.abc import Sequence
from typing import cast

import relbench
import torch
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from sdm import RelationalData, TableTensor, infer_stypes
from sdm.models import KumoRFM

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=1000)
args = parser.parse_args()

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
sampler = data.sampler(
    time_columns={
        name: table.time_col
        for name, table in db.table_dict.items()
        if table.time_col is not None
    },
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

train_table = torch.cat(task_tables[:2], dim=0)
perm = torch.randperm(len(train_table))[: args.context_size]
train_table = cast(TableTensor, train_table[perm])

# Execute Model ###############################################################
model = KumoRFM(pretrained=False, device=device)

kwargs = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
    },
    "num_neighbors": [16, 16],
    "task_time_column": task.time_col,
}
train_table, related_tables = sampler(train_table, **kwargs).to(device)
model.fit(
    x=train_table.drop_columns(task.target_col),
    y=train_table[task.target_col],
    related_tables=related_tables,
)

test_table = task_tables[-1].drop_columns(task.target_col)
for test_batch in test_table.split(args.batch_size):
    model.predict(*sampler(test_batch, **kwargs).to(device))
model.clear()
