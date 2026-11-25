# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full fine-tuning of KumoTabular ("kumo-small") or TabICLv2."""

import argparse
from pathlib import Path
from typing import cast

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
# Optimal values for the hyperparameters below depend on the model and
# dataset. Select them based on validation performance.
parser.add_argument("--max-epochs", type=int, default=10)
parser.add_argument("--max-chunk-size", type=int, default=10_000)
parser.add_argument("--query-fraction", type=float, default=0.2)
parser.add_argument("--num-train-estimators", type=int, default=4)
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


@torch.inference_mode()
def evaluate(context: sdm.TableTensor, query: sdm.TableTensor) -> float:
    model.eval()
    with torch.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        out = model(
            x_context=context.drop_columns(target),
            y_context=context[target],
            x_query=query.drop_columns(target),
            num_estimators=args.num_estimators,
            generator=torch.Generator(device).manual_seed(args.seed),
        )
    if args.task == "classification":
        prob, y = sdm.evaluation.to_class_indices(
            out, query[target], missing_score=0.0
        )
        return float((prob.argmax(dim=-1) == y).float().mean())

    pred = out.numerical.float().mean(dim=-1)
    y = query[target].numerical.squeeze(-1)
    return float((pred - y).pow(2).mean().sqrt())  # RMSE


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
higher_is_better = args.task == "classification"
metric_name = "acc" if args.task == "classification" else "rmse"
ckpt_path = Path(f"ckpt-{args.model}-{args.task}.pt")
train_labels = df[target].iloc[train_index].to_numpy()

val_metric = evaluate(context=train_table, query=val_table)
print(f"epoch=0/{args.max_epochs} val_{metric_name}={val_metric:.4f}")
torch.save(model.state_dict(), ckpt_path)

for epoch in range(1, args.max_epochs + 1):
    model.train()
    total_loss = 0.0
    steps = 0
    for chunk in torch.randperm(len(train_table)).split(args.max_chunk_size):
        if len(chunk) < 50:
            continue
        context_rows, query_rows = train_test_split(
            chunk,
            test_size=args.query_fraction,
            stratify=train_labels[chunk]
            if args.task == "classification"
            else None,
        )
        context = train_table[context_rows]
        query = train_table[query_rows]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = model(
                x_context=context.drop_columns(target),
                y_context=context[target],
                x_query=query.drop_columns(target),
                recipe=train_recipe,
                num_estimators=args.num_train_estimators,
                estimator_batch_size=None,
            )
        y_query = query[target]
        if args.task == "classification":
            y_views = cast(
                sdm.TableTensor, y_query.expand(out.size(0), *y_query.shape)
            )
            logits, y = sdm.evaluation.to_class_indices(
                out, y_views, missing_score=-torch.inf
            )
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        else:  # Pinball loss over the models' 999 fixed quantile levels.
            diff = y_query.numerical - out.numerical.float()
            level = torch.linspace(0.001, 0.999, 999, device=device)
            loss = torch.maximum(level * diff, (level - 1) * diff).mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        steps += 1

    metric = evaluate(context=train_table, query=val_table)
    print(
        f"epoch={epoch}/{args.max_epochs} "
        f"val_{metric_name}={metric:.4f} "
        f"train_loss={total_loss / steps:.4f}"
    )
    if metric > val_metric if higher_is_better else metric < val_metric:
        val_metric = metric
        torch.save(model.state_dict(), ckpt_path)

best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(best_state)
test_metric = evaluate(context=train_table, query=test_table)
print(f"test_{metric_name}={test_metric:.4f}")
