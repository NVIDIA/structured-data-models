# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
from typing import Any, cast

import pandas as pd
import relbench
import torch
import torchmetrics
from tqdm import tqdm

import sdm
import sdm.processing as sp


def embed_entity_text(
    entity: sdm.TableTensor,
    task_ids: pd.Series,
    pkey: str,
    *,
    model_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor] | None:
    """Embed task entities and return CPU embeddings."""
    if len(entity.columns[sdm.Stype.text]) == 0:
        return None
    entity_ids = entity.select_columns(pkey).to_pandas()[pkey]
    in_task = torch.from_numpy(
        entity_ids.isin(pd.unique(task_ids)).to_numpy(dtype=bool)
    )
    text = entity.select_stypes(sdm.Stype.text)[in_task]
    if text.size(0) == 0:
        return None
    encoder = sp.SentenceTransformer(
        model_name=model_name,
        batch_size=32,
        max_seq_length=2048,
    )
    row_chunk = 1024
    parts = []
    columns: tuple[str, ...] = ()
    with torch.inference_mode():
        for start in range(0, text.size(0), row_chunk):
            embedded = encoder(text[start : start + row_chunk].to(device))
            columns = embedded.columns[sdm.Stype.numerical]
            parts.append(embedded.numerical.cpu())
    del encoder, embedded, text
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0), columns, in_task


def attach_text(
    entity: sdm.TableTensor,
    embeddings: torch.Tensor,
    columns: tuple[str, ...],
    in_task: torch.Tensor,
    context_ids: pd.Series,
    pkey: str,
    text_dim: int,
) -> sdm.TableTensor:
    """Project embeddings with a PCA fit on the context entities."""
    entity_ids = entity.select_columns(pkey).to_pandas()[pkey]
    in_context = torch.from_numpy(
        entity_ids.isin(pd.unique(context_ids)).to_numpy(dtype=bool)
    )
    pca = sp.PCA(text_dim)
    pca.fit(
        sdm.TableTensor(
            columns={"numerical": columns},
            numerical=embeddings[in_context[in_task]],
        )
    )
    reduced = pca.transform(
        sdm.TableTensor(
            columns={"numerical": columns},
            numerical=embeddings,
        )
    )
    values = torch.full(
        (entity.size(0), reduced.numerical.size(-1)),
        torch.nan,
        dtype=reduced.numerical.dtype,
    )
    values[in_task] = reduced.numerical
    return cast(
        sdm.TableTensor,
        torch.cat(
            [
                entity.drop_stypes(sdm.Stype.text),
                sdm.TableTensor(
                    columns={
                        "numerical": reduced.columns[sdm.Stype.numerical],
                    },
                    numerical=values,
                ),
            ],
            dim=-1,
        ),
    )


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--max_test_steps", type=int, default=None)
parser.add_argument("--num_neighbors", type=int, nargs="*", default=[16, 16])
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--num_lags", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--text", action="store_true")
parser.add_argument("--text_dim", type=int, default=64)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collect Relational Data #####################################################
dataset = relbench.load_dataset(args.dataset)
task = dataset.load_task(args.task)
db = task.get_db(upto_test_timestamp=False)
tables = {
    name: sdm.TableTensor.from_pandas(
        df=table.df,
        stypes=sdm.infer_stypes(
            table.df.head(10_000),
            overrides={
                cast(str, table.pkey_col): "id",
                **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
            },
            text=(
                "infer" if args.text and name == task.entity_table else "drop"
            ),
            unsupported="drop",
        ),
    )
    for name, table in db.table_dict.items()
}
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
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
    ],
)
time_columns = {
    name: table.time_col
    for name, table in db.table_dict.items()
    if table.time_col is not None
}

# Collect Task Table ##########################################################
if task.task_type not in {
    relbench.base.TaskType.BINARY_CLASSIFICATION,
    relbench.base.TaskType.MULTICLASS_CLASSIFICATION,
    relbench.base.TaskType.REGRESSION,
}:
    quit(f"Unsupported task {task.task_type.value!r}")

dfs = [
    task.get_table(split, mask_input_cols=False).df
    for split in ["train", "val", "test"]
]
df = pd.concat(dfs, ignore_index=True)

# Add lag target features to task table:
right = df.sort_values(task.time_col)
left = pd.DataFrame(
    {
        task.entity_col: df[task.entity_col],
        "__lookup_time__": df[task.time_col],
        "__row__": range(len(df)),
    }
)
for lag in range(1, args.num_lags + 1):
    column = f"{task.target_col}_lag_{lag}"
    if task.task_type == relbench.base.TaskType.REGRESSION:
        df[column] = float("NaN")
    else:
        df[column] = pd.Series(pd.NA, index=df.index, dtype="object")

    merged = pd.merge_asof(
        left.sort_values("__lookup_time__"),
        right,
        by=task.entity_col,
        left_on="__lookup_time__",
        right_on=task.time_col,
        direction="backward",
        allow_exact_matches=False,
    )
    mask = merged[task.time_col].notna()
    rows = merged.loc[mask, "__row__"].to_numpy()
    df.loc[rows, column] = merged.loc[mask, task.target_col].to_numpy()

    # Find the next lag strictly before the previous observation.
    left = merged.loc[mask, [task.entity_col, "__row__", task.time_col]]
    left = left.rename(columns={task.time_col: "__lookup_time__"})

if task.task_type == relbench.base.TaskType.REGRESSION:
    target_stype = "numerical"
else:
    target_stype = "categorical"

task_table = sdm.TableTensor.from_pandas(
    df=df,
    stypes={
        task.entity_col: "id",
        task.time_col: "datetime",
        task.target_col: target_stype,
        **{
            f"{task.target_col}_lag_{lag}": target_stype
            for lag in range(1, args.num_lags + 1)
        },
    },
)
context, query = task_table.split([len(dfs[0]) + len(dfs[1]), len(dfs[2])])

num_estimators = args.num_estimators
if len(context) > args.context_size:  # Sample different context per estimator:
    repeats = math.ceil(args.context_size * num_estimators / len(context))
    perm = torch.cat([torch.randperm(len(context)) for _ in range(repeats)])
    context = context[perm[: args.context_size * num_estimators]]
    if num_estimators > 1:
        context = context.unflatten(0, (num_estimators, args.context_size))
        query = query.expand(num_estimators, *query.size())
        num_estimators = None

# Execute Model ###############################################################
if args.text:
    entity_pkey = cast(str, db.table_dict[task.entity_table].pkey_col)
    entity = tables[task.entity_table]
    encoded = embed_entity_text(
        entity=entity,
        task_ids=df[task.entity_col],
        pkey=entity_pkey,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )
    if encoded is not None:
        embeddings, columns, in_task = encoded
        tables[task.entity_table] = attach_text(
            entity=entity,
            embeddings=embeddings,
            columns=columns,
            in_task=in_task,
            context_ids=pd.concat(
                [dfs[0][task.entity_col], dfs[1][task.entity_col]],
                ignore_index=True,
            ),
            pkey=entity_pkey,
            text_dim=args.text_dim,
        )

sampler = data.sampler(time_columns=time_columns)
model = sdm.models.KumoRelational(device=device)
recipe = model.default_recipe()
kwargs: dict[str, Any] = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
    },
    "num_neighbors": args.num_neighbors,
    "task_time_column": task.time_col,
}

context, related_tables = sampler(context, **kwargs).to(device)
with torch.amp.autocast(device.type, torch.float16, enabled=True):
    model.fit(
        x=context.drop_columns(task.target_col),
        y=context[task.target_col],
        related_tables=related_tables,
        num_estimators=num_estimators,
        recipe=recipe,
    )

if task.task_type == relbench.base.TaskType.REGRESSION:
    metric = torchmetrics.regression.MeanAbsoluteError()
elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
    metric = torchmetrics.classification.BinaryAUROC()
    positive_class = dfs[0][task.target_col].unique()[-1].item()
else:
    metric = torchmetrics.classification.MulticlassAccuracy(average="micro")
metric = metric.to(device)
for batch in tqdm(query.split(args.batch_size, -2)[: args.max_test_steps]):
    x_query = batch.drop_columns(task.target_col)
    y_query = batch[task.target_col].to(device)
    if num_estimators is None:
        y_query = y_query[0]
    with torch.amp.autocast(device.type, torch.float16, enabled=True):
        out = model.predict(*sampler(x_query, **kwargs).to(device))

    if task.task_type == relbench.base.TaskType.REGRESSION:
        pred = out["q500"].numerical.squeeze(-1)  # Median prediction.
        target = y_query.numerical.squeeze(-1)
    elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
        pred, target = sdm.evaluation.to_binary_class(
            out, y_query, positive_class
        )
    else:
        pred, target = sdm.evaluation.to_class_indices(out, y_query)
    metric.update(pred, target)

if task.task_type == relbench.base.TaskType.REGRESSION:
    print(f"nMAE: {metric.compute() / task.nmae_std:.4f}")
elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
    print(f"AUROC: {metric.compute():.4f}")
else:
    print(f"ACC: {metric.compute():.4f}")
