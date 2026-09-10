import argparse
from typing import Any, cast

import pandas as pd
import torch
import torchmetrics
import tqdm
from datasets import load_dataset

import sdm

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--max_test_steps", type=int, default=None)
parser.add_argument("--num_neighbors", type=int, nargs="*", default=[16, 16])
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sale_tasks = [
    "SALESOFFICE",
    "SALESGROUP",
    "CUSTOMERPAYMENTTERMS",
    "SHIPPINGCONDITION",
    "HEADERINCOTERMSCLASSIFICATION",
]
item_tasks = [
    "PLANT",
    "SHIPPINGPOINT",
    "ITEMINCOTERMSCLASSIFICATION",
]


# Load and sanitize raw data ##################################################
def get_df(name: str, split: str = "train") -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        load_dataset("sap-ai-research/SALT", name, split=split).to_pandas(),
    )


sales_train = get_df("salesdocuments", split="train")
sales_test = get_df("salesdocuments", split="test")
items_train = get_df("salesdocument_items", split="train")
items_test = get_df("salesdocument_items", split="test")

dfs: dict[str, pd.DataFrame] = {}
dfs["sales"] = pd.concat([sales_train, sales_test], ignore_index=True)
dfs["items"] = pd.concat([items_train, items_test], ignore_index=True)
dfs["customers"] = get_df("customers")
dfs["addresses"] = get_df("addresses")

# 1. Merge timestamps into single `datetime` column:
date = dfs["sales"]["CREATIONDATE"].astype(str)
time = dfs["sales"]["CREATIONTIME"].astype(str)
dfs["sales"]["CREATIONDATETIME"] = pd.to_datetime(date + " " + time)
del dfs["sales"]["CREATIONDATE"]
del dfs["sales"]["CREATIONTIME"]
# 2. Add timestamp to items:
dfs["items"] = pd.merge(
    left=dfs["items"],
    right=dfs["sales"][["SALESDOCUMENT", "CREATIONDATETIME"]],
    how="left",
    left_on="SALESDOCUMENT",
    right_on="SALESDOCUMENT",
)
# 3. Remove auto-generated columns:
del dfs["sales"]["__index_level_0__"]
del dfs["items"]["__index_level_0__"]
del dfs["customers"]["__index_level_0__"]
del dfs["addresses"]["__index_level_0__"]
# 4. Add missing primary key:
dfs["items"]["ID"] = range(len(dfs["items"]))
# 5. Rename columns to align with task name:
dfs["sales"] = dfs["sales"].rename(
    columns={"INCOTERMSCLASSIFICATION": "HEADERINCOTERMSCLASSIFICATION"}
)
dfs["items"] = dfs["items"].rename(
    columns={"INCOTERMSCLASSIFICATION": "ITEMINCOTERMSCLASSIFICATION"}
)
# 7. Join 1:1 mapping between customers and addresses:
dfs["customers"] = pd.merge(
    left=dfs["customers"],
    right=dfs["addresses"],
    how="left",
    left_on="ADDRESSID",
    right_on="ADDRESSID",
).drop(columns=["ADDRESSID"])

# Collect Task Table ##########################################################
if args.task.upper() in sale_tasks:
    context, query = sdm.TableTensor.from_pandas(
        df=dfs["sales"],
        stypes={
            "SALESDOCUMENT": "id",
            "CREATIONDATETIME": "datetime",
            args.task.upper(): "categorical",
        },
    ).split(len(sales_train), dim=-2)
    task_link = {
        "task_column": "SALESDOCUMENT",
        "table": "sales",
        "table_column": "SALESDOCUMENT",
    }
else:
    context, query = sdm.TableTensor.from_pandas(
        df=dfs["items"],
        stypes={
            "ID": "id",
            "CREATIONDATETIME": "datetime",
            args.task.upper(): "categorical",
        },
    ).split(len(items_train), dim=-2)
    task_link = {
        "task_column": "ID",
        "table": "items",
        "table_column": "ID",
    }

context = context[torch.randperm(len(context))[: args.context_size]]

# Build Relational Data #######################################################
dfs["sales"] = dfs["sales"].drop(labels=sale_tasks, axis=1)
dfs["items"] = dfs["items"].drop(labels=item_tasks, axis=1)

data = sdm.RelationalData(
    tables={
        name: sdm.TableTensor.from_pandas(
            df=df,
            stypes=sdm.infer_stypes(
                df.head(10_000),
                overrides={
                    "ID": "id",
                    "SALESDOCUMENT": "id",
                    "CUSTOMER": "id",
                    "SOLDTOPARTY": "id",
                    "SHIPTOPARTY": "id",
                    "PAYERPARTY": "id",
                    "BILLTOPARTY": "id",
                },
            ),
        )
        for name, df in dfs.items()
    },
    relationships=[
        {
            "left_table": left_table,
            "left_column": left_column,
            "right_table": right_table,
            "right_column": right_column,
        }
        for left_table, left_column, right_table, right_column in [
            ("items", "SALESDOCUMENT", "sales", "SALESDOCUMENT"),
            ("items", "SOLDTOPARTY", "customers", "CUSTOMER"),
            ("items", "SHIPTOPARTY", "customers", "CUSTOMER"),
            ("items", "BILLTOPARTY", "customers", "CUSTOMER"),
            ("items", "PAYERPARTY", "customers", "CUSTOMER"),
        ]
    ],
)
sampler = data.sampler(
    time_columns={
        "sales": "CREATIONDATETIME",
        "items": "CREATIONDATETIME",
    },
)

# Execute Model ###############################################################
model = sdm.models.KumoRelational(task="classification", device=device)
kwargs: dict[str, Any] = {
    "task_link": task_link,
    "num_neighbors": args.num_neighbors,
    "task_time_column": "CREATIONDATETIME",
}

context, related_tables = sampler(context, **kwargs).to(device)
with torch.amp.autocast(device.type, torch.float16, enabled=True):
    model.fit(
        x=context.drop_columns(args.task.upper()),
        y=context[args.task.upper()],
        related_tables=related_tables,
        num_estimators=args.num_estimators,
    )

metric = torchmetrics.aggregation.MeanMetric().to(device)
for batch in tqdm.tqdm(query.split(args.batch_size)[: args.max_test_steps]):
    x_query = batch.drop_columns(args.task.upper())
    y_query = batch[args.task.upper()].to(device)
    with torch.amp.autocast(device.type, torch.float16, enabled=True):
        out = model.predict(*sampler(x_query, **kwargs).to(device))
    pred, target = sdm.evaluation.to_class_indices(out, y_query)
    match = pred.argsort(dim=-1, descending=True) == target.unsqueeze(-1)
    metric.update((match.argmax(dim=-1) + 1).reciprocal())

print(f"MRR: {metric.compute():.4f}")
