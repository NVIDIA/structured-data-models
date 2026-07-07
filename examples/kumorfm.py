from time import perf_counter

import torch

t = perf_counter()
import argparse
from collections.abc import Sequence

import pandas as pd
import relbench
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from sdm import TableTensor
from torch_geometric.data import HeteroData
from torch_geometric.sampler import NeighborSampler

print("import", perf_counter() - t)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=1000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collect Task Table ##########################################################
t = perf_counter()
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
train_table = train_table[
    torch.randperm(len(train_table))[: args.context_size]
]
test_table = task_tables[-1]
print("task", perf_counter() - t)

# Collect Related Tables ######################################################
t = perf_counter()
tables: dict[str, TableTensor] = {}
db = get_dataset(args.dataset, download=True).get_db(upto_test_timestamp=False)
for name, table in db.table_dict.items():
    stypes: dict[str, str] = {}
    for column in table.df.columns:
        if column == table.pkey_col or column in table.fkey_col_to_pkey_table:
            stypes[column] = "id"
        elif column == table.time_col:
            stypes[column] = "datetime"
        elif pd.api.types.is_string_dtype(table.df[column]):
            stypes[column] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(table.df[column]):
            stypes[column] = "datetime"
        else:
            stypes[column] = "numerical"
    tables[name] = TableTensor.from_pandas(table.df, stypes)
print("graph", perf_counter() - t)

# Create PyG-based Neighbor Sampler ###########################################
t = perf_counter()
hetero_data = HeteroData()
for src_table_name, table in db.table_dict.items():
    hetero_data[src_table_name].num_nodes = len(table.df)
    if table.time_col is not None:
        time = table.df[table.time_col].astype("int64").to_numpy()
        hetero_data[src_table_name].time = torch.from_numpy(time)

    for fkey, dst_table_name in table.fkey_col_to_pkey_table.items():
        mask = torch.from_numpy(table.df[fkey].notna().to_numpy())
        src = torch.arange(len(table.df))[mask]
        dst = torch.from_numpy(table.df[fkey].to_numpy())[mask].long()
        edge_type = (src_table_name, fkey, dst_table_name)
        hetero_data[edge_type].edge_index = torch.stack([src, dst], dim=0)
        edge_type = (dst_table_name, f"rev_{fkey}", src_table_name)
        hetero_data[edge_type].edge_index = torch.stack([dst, src], dim=0)
sampler = NeighborSampler(
    hetero_data,
    num_neighbors=[10, 10],
    disjoint=True,
    temporal_strategy="last",
    time_attr="time",
)

# sampler.sample_from_nodes(
#         node=
#
#         )


print("sampler", perf_counter() - t)
print(train_table)
