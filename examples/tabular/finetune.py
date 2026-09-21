# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full fine-tuning of KumoTabular ("kumo-small") or TabICLv2."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split

import sdm
import sdm.processing as sp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=("kumo-small", "tabiclv2"),
    default="tabiclv2",
)
parser.add_argument(
    "--task",
    choices=("classification", "regression"),
    default="classification",
)
parser.add_argument("--max-epochs", type=int, default=10)
parser.add_argument("--steps-per-epoch", type=int, default=30)
parser.add_argument("--context-size", type=int, default=256)
parser.add_argument("--query-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--num-estimators", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.task == "classification":
    df = load_digits(as_frame=True).frame
    target = "target"
    stypes = sdm.infer_stypes(df, overrides={target: "categorical"})
else:
    df = fetch_california_housing(as_frame=True).frame
    target = "MedHouseVal"
    stypes = sdm.infer_stypes(df)

table = sdm.TableTensor.from_pandas(df, stypes=stypes, device=device)

# Split dataset into training/validation/test (60/20/20):
train_index, rest_index = train_test_split(
    torch.arange(len(df)).numpy(),
    test_size=0.4,
    random_state=args.seed,
    stratify=df[target] if args.task == "classification" else None,
)
val_index, test_index = train_test_split(
    rest_index,
    test_size=0.5,
    random_state=args.seed,
    stratify=df[target].iloc[rest_index]
    if args.task == "classification"
    else None,
)
train_table = table[torch.from_numpy(train_index).to(device)]
val_table = table[torch.from_numpy(val_index).to(device)]
test_table = table[torch.from_numpy(test_index).to(device)]

if args.model == "kumo-small":
    model = sdm.models.KumoTabular(task=args.task, size="small", device=device)
else:
    model = sdm.models.TabICLv2(task=args.task, device=device)

train_recipe = model.default_recipe()
train_recipe.target = sp.StypeDispatch(  # No target flipping.
    categorical=[
        sp.AlignCategories(),
        sp.ShuffleCategories(method="shift"),
    ],
    numerical=sp.Standardize(),
)
train_recipe.output = sp.Identity()  # No post-processing.


def evaluate(context: sdm.TableTensor, query: sdm.TableTensor) -> float:
    model.eval()
    out = model(
        x_context=context.drop_columns(target),
        y_context=context[target],
        x_query=query.drop_columns(target),
        num_estimators=args.num_estimators,
    )
    if args.task == "classification":
        prob, y = sdm.evaluation.to_class_indices(
            out, query[target], missing_score=0.0
        )
        return float((prob.argmax(dim=-1) == y).float().mean())

    pred = out.numerical.mean(dim=-1)
    y = query[:, target].numerical.squeeze(-1)
    return float((pred - y).pow(2).mean().sqrt())  # RMSE


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
quantile_levels = torch.linspace(0.001, 0.999, 999, device=device)
higher_is_better = args.task == "classification"
metric_name = "acc" if args.task == "classification" else "rmse"
ckpt_path = Path(f"ckpt-{args.model}-{args.task}.pt")

val_metric = evaluate(context=train_table, query=val_table)
print(f"epoch=0/{args.max_epochs} val_{metric_name}={val_metric:.4f}")
torch.save(model.state_dict(), ckpt_path)

for epoch in range(1, args.max_epochs + 1):
    model.train()
    total_loss = 0.0
    for _ in range(args.steps_per_epoch):
        index = torch.randperm(train_table.size(0), device=device)
        index = index[: args.context_size + args.query_size]
        context = train_table[index[: args.context_size]]
        query = train_table[index[args.context_size :]]

        optimizer.zero_grad()
        out = model(
            x_context=context.drop_columns(target),
            y_context=context[target],
            x_query=query.drop_columns(target),
            recipe=train_recipe,
        )[0]
        if args.task == "classification":
            logits, y = sdm.evaluation.to_class_indices(
                out, query[target], missing_score=-torch.inf
            )
            loss = F.cross_entropy(logits, y)
        else:  # Pinball loss over the model's 999 fixed quantile levels:
            diff = query[target].numerical - out.numerical
            loss = torch.maximum(
                quantile_levels * diff,
                (quantile_levels - 1) * diff,
            ).mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())

    metric = evaluate(context=train_table, query=val_table)
    print(
        f"epoch={epoch}/{args.max_epochs} "
        f"val_{metric_name}={metric:.4f} "
        f"train_loss={total_loss / args.steps_per_epoch:.4f}"
    )
    if metric > val_metric if higher_is_better else metric < val_metric:
        val_metric = metric
        torch.save(model.state_dict(), ckpt_path)

best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(best_state)
test_metric = evaluate(context=train_table, query=test_table)
print(f"test_{metric_name}={test_metric:.4f}")
