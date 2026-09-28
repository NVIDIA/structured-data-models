# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune KumoRelational on the RelBench rel-f1/driver-top3 task."""

import argparse
from pathlib import Path
from typing import cast

import pandas as pd
import relbench
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import sdm
import sdm.processing as sp

parser = argparse.ArgumentParser()
parser.add_argument("--max-epochs", type=int, default=5)
parser.add_argument("--steps-per-epoch", type=int, default=10)
parser.add_argument("--context-size", type=int, default=500)
parser.add_argument("--query-size", type=int, default=32)
parser.add_argument("--test-size", type=int, default=64)
parser.add_argument("--num-neighbors", type=int, nargs="+", default=[8, 8])
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--checkpoint", type=Path, default=Path("ckpt-kumo-relational.pt")
)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = relbench.load_dataset("stanford-star/relbench-v1/rel-f1")
task = dataset.load_task("driver-top3")
db = dataset.get_db()

data = sdm.RelationalData(
    tables={
        name: sdm.TableTensor.from_pandas(
            table.df,
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
        for name, table in db.table_dict.items()
    },
    relationships=[
        {
            "left_table": name,
            "left_column": foreign_key,
            "right_table": parent,
            "right_column": cast(str, db.table_dict[parent].pkey_col),
        }
        for name, table in db.table_dict.items()
        for foreign_key, parent in table.fkey_col_to_pkey_table.items()
    ],
)
sampler = data.sampler(
    time_columns={
        name: table.time_col
        for name, table in db.table_dict.items()
        if table.time_col is not None
    }
)

frames = [
    task.get_table(split, mask_input_cols=False).df
    for split in ("train", "val", "test")
]
target = task.target_col
task_table = sdm.TableTensor.from_pandas(
    pd.concat(frames, ignore_index=True),
    stypes={
        task.entity_col: "id",
        task.time_col: "datetime",
        target: "categorical",
    },
)
train_table, val_table, test_table = task_table.split(
    [len(df) for df in frames]
)

ordered = frames[0].sort_values(
    [task.time_col, task.entity_col], kind="mergesort"
)
val_context = train_table[
    torch.as_tensor(ordered.tail(args.context_size).index.to_numpy())
]
val_query = val_table[: args.test_size]

# The test context uses only train and validation labels.
test_context_rows = (
    pd.concat(frames[:2], ignore_index=True)
    .sort_values([task.time_col, task.entity_col], kind="mergesort")
    .tail(args.context_size)
)
test_context = task_table[: len(train_table) + len(val_table)][
    torch.as_tensor(test_context_rows.index.to_numpy())
]
test_query = test_table[: args.test_size]

task_link = {
    "task_column": task.entity_col,
    "table": task.entity_table,
    "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
}


def sample(
    table: sdm.TableTensor,
) -> tuple[sdm.TableTensor, sdm.RelatedTables[sdm.TableTensor]]:
    return sampler(
        task_table=table,
        task_link=task_link,
        num_neighbors=args.num_neighbors,
        task_time_column=task.time_col,
    ).to(device)


val_context, val_related_context = sample(val_context)
val_query, val_related_query = sample(val_query)
test_context, test_related_context = sample(test_context)
test_query, test_related_query = sample(test_query)

model = sdm.models.KumoRelational(task="classification", device=device)
train_recipe = model.default_recipe()
train_recipe.target = sp.StypeDispatch(
    categorical=[sp.AlignCategories(), sp.ShuffleCategories(method="shift")],
)
train_recipe.output = sp.Identity()  # Cross-entropy consumes raw class scores.


def evaluate(
    context: sdm.TableTensor,
    query: sdm.TableTensor,
    related_context: sdm.RelatedTables[sdm.TableTensor],
    related_query: sdm.RelatedTables[sdm.TableTensor],
) -> float:
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device.type, enabled=device.type == "cuda"),
    ):
        out = model(
            x_context=context.drop_columns(target),
            y_context=context[target],
            x_query=query.drop_columns(target),
            related_context_tables=related_context,
            related_query_tables=related_query,
            num_hops=len(args.num_neighbors),
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
    score, label = sdm.evaluation.to_binary_class(
        out, query[target], positive_class=1
    )
    return float(roc_auc_score(label.cpu(), score.cpu()))


val_auc = evaluate(
    val_context, val_query, val_related_context, val_related_query
)
print(f"epoch=0/{args.max_epochs} val_auroc={val_auc:.4f}")
torch.save(model.state_dict(), args.checkpoint)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=0.01
)
for epoch in range(1, args.max_epochs + 1):
    model.train()
    total_loss = 0.0
    for _ in range(args.steps_per_epoch):
        end = int(
            torch.randint(
                args.context_size + args.query_size,
                len(ordered) + 1,
                (1,),
            )
        )
        query_rows = ordered.iloc[end - args.query_size : end]
        context_rows = ordered.iloc[: end - args.query_size]
        context_rows = context_rows.loc[
            context_rows[task.time_col] < query_rows[task.time_col].min()
        ].tail(args.context_size)
        train_context = train_table[
            torch.as_tensor(context_rows.index.to_numpy())
        ]
        train_query = train_table[torch.as_tensor(query_rows.index.to_numpy())]
        train_context, train_related_context = sample(train_context)
        train_query, train_related_query = sample(train_query)

        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(
                x_context=train_context.drop_columns(target),
                y_context=train_context[target],
                x_query=train_query.drop_columns(target),
                related_context_tables=train_related_context,
                related_query_tables=train_related_query,
                recipe=train_recipe,
                num_hops=len(args.num_neighbors),
            )[0]
            logits, label = sdm.evaluation.to_class_indices(
                out, train_query[target], missing_score=-torch.inf
            )
            loss = F.cross_entropy(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach())
    metric = evaluate(
        val_context, val_query, val_related_context, val_related_query
    )
    print(
        f"epoch={epoch}/{args.max_epochs} "
        f"val_auroc={metric:.4f} "
        f"train_loss={total_loss / args.steps_per_epoch:.4f}"
    )
    if metric > val_auc:
        val_auc = metric
        torch.save(model.state_dict(), args.checkpoint)

model.load_state_dict(
    torch.load(args.checkpoint, map_location=device, weights_only=True)
)
test_auc = evaluate(
    test_context, test_query, test_related_context, test_related_query
)
print(f"test_auroc={test_auc:.4f}")
