import argparse
from collections.abc import Sequence

import relbench
import torch
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from sdm import RelatedTables, TableTensor, infer_stypes
from torch_geometric.data import HeteroData
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=1000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_table = train_table[
    torch.randperm(len(train_table))[: args.context_size]
]
test_table = task_tables[-1]

# Collect Related Tables ######################################################
db = get_dataset(args.dataset, download=True).get_db(upto_test_timestamp=False)
related_tables = RelatedTables(
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
            "right_column": db.table_dict[right_table].pkey_col,
        }
        for left_table, table in db.table_dict.items()
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
    ]
    + [
        {
            "left_column": task.entity_col,
            "right_table": task.entity_table,
            "right_column": task.entity_col,
        }
    ],
)

print(related_tables)
quit()

# Create PyG-based Neighbor Sampler ###########################################
from time import perf_counter

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

out = sampler.sample_from_nodes(
    NodeSamplerInput(
        input_id=None,
        node=torch.arange(10),
        time=train_table[task.time_col].datetime.view(-1),
        input_type=task.entity_table,
    )
)
print(out)


print("sampler", perf_counter() - t)
