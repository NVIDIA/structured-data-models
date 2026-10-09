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
parser.add_argument("--eval-batch-size", type=int, default=64)
parser.add_argument("--num-neighbors", type=int, nargs="+", default=[8, 8])
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--checkpoint", type=Path, default=Path("ckpt-kumo-relational.pt")
)
args = parser.parse_args()
if args.context_size < 2:
    raise ValueError("--context-size must be at least 2 for this binary task")

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = relbench.load_dataset("stanford-star/relbench-v1/rel-f1")
task = dataset.load_task("driver-top3")
# Keep later test history; the sampler enforces each query's timestamp cutoff.
db = task.get_db(upto_test_timestamp=False)

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


def context_rows(available: pd.DataFrame) -> pd.DataFrame:
    rows = available.tail(args.context_size)
    if rows[target].nunique() == 2:
        return rows
    missing = available.loc[~available[target].isin(rows[target])].tail(1)
    if missing.empty:
        raise ValueError("Context needs an earlier row from each target class")
    return pd.concat([rows.iloc[1:], missing]).sort_values(
        [task.time_col, task.entity_col], kind="mergesort"
    )


val_context = train_table[
    torch.as_tensor(context_rows(ordered).index.to_numpy())
]

# The test context uses only train and validation labels.
test_context_rows = pd.concat(frames[:2], ignore_index=True).sort_values(
    [task.time_col, task.entity_col], kind="mergesort"
)
test_context = task_table[: len(train_table) + len(val_table)][
    torch.as_tensor(context_rows(test_context_rows).index.to_numpy())
]

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
test_context, test_related_context = sample(test_context)

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
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with (
        torch.inference_mode(),
        torch.autocast(device.type, enabled=device.type == "cuda"),
    ):
        model.fit(
            x=context.drop_columns(target),
            y=context[target],
            related_tables=related_context,
            num_hops=len(args.num_neighbors),
            generator=generator,
        )
        for batch in query.split(args.eval_batch_size, dim=-2):
            batch, related_query = sample(batch)
            out = model.predict(
                x=batch.drop_columns(target),
                related_tables=related_query,
            )
            score, label = sdm.evaluation.to_binary_class(
                out, batch[target], positive_class=1
            )
            scores.append(score.cpu())
            labels.append(label.cpu())
    model.clear()
    return float(roc_auc_score(torch.cat(labels), torch.cat(scores)))


val_auc = evaluate(val_context, val_table, val_related_context)
print(f"epoch=0/{args.max_epochs} val_auroc={val_auc:.4f}")
torch.save(model.state_dict(), args.checkpoint)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=0.01
)
# Queries begin after both classes have appeared in the available history.
first_query = ordered[task.time_col].searchsorted(
    ordered.groupby(target)[task.time_col].min().max(), side="right"
)
min_end = max(args.context_size, first_query) + args.query_size
for epoch in range(1, args.max_epochs + 1):
    model.train()
    total_loss = 0.0
    for _ in range(args.steps_per_epoch):
        end = int(
            torch.randint(
                min_end,
                len(ordered) + 1,
                (1,),
            )
        )
        query_rows = ordered.iloc[end - args.query_size : end]
        available = ordered.iloc[: end - args.query_size]
        available = available.loc[
            available[task.time_col] < query_rows[task.time_col].min()
        ]
        train_context = train_table[
            torch.as_tensor(context_rows(available).index.to_numpy())
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
        optimizer.step()
        total_loss += float(loss.detach())
    metric = evaluate(val_context, val_table, val_related_context)
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
test_auc = evaluate(test_context, test_table, test_related_context)
print(f"test_auroc={test_auc:.4f}")
