# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full fine-tuning of KumoTabular ("kumo-small").

Fine-tunes every parameter of ``sdm.models.KumoTabular(size="small")`` with
gradient descent on resampled in-context batches, using ``torch.optim.AdamW``
directly against the model's public ``forward()`` API. No changes to `sdm`
are required: a custom ``Recipe`` swaps target processing for an identity
pass-through so query labels (encoded once, consistently, from the full
labeled pool) can be compared against the model's raw per-estimator output
inside a training ``Callback``, mirroring how ``sdm.explain.GradientExplainer``
already differentiates through the model.
"""

import argparse

import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor

import sdm
import sdm.processing as sp
from sdm.models.callback import Callback


class _TrainingCallback(Callback):
    requires_grad = True

    def __init__(
        self,
        task: str,
        y_query: Tensor,
        quantile_levels: Tensor | None = None,
    ) -> None:
        self._task = task
        self._y_query = y_query
        self._quantile_levels = quantile_levels
        self.loss: Tensor | None = None

    def on_model_forward_end(
        self,
        model: torch.nn.Module,
        out: sdm.TableTensor,
    ) -> sdm.TableTensor:
        if self._task == "classification":
            self.loss = F.cross_entropy(out.numerical, self._y_query)
        else:
            # Pinball loss over the model's 999 fixed quantile levels.
            assert self._quantile_levels is not None
            diff = self._y_query.unsqueeze(-1) - out.numerical
            self.loss = torch.maximum(
                self._quantile_levels * diff,
                (self._quantile_levels - 1) * diff,
            ).mean()
        self.loss.backward()
        return out


def finetune(
    model: sdm.models.ICLModel,
    recipe: sp.Recipe,
    pool: sdm.TableTensor,
    target_column: str,
    *,
    task: str,
    steps: int,
    context_size: int,
    query_size: int,
    lr: float,
    generator: torch.Generator,
) -> None:
    """Full fine-tune every parameter of ``model`` in place.

    Each step resamples a disjoint context/query batch from ``pool``, runs
    the model's differentiable ``forward()`` path, and backpropagates a
    supervised loss on the query rows computed inside a training callback.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    quantile_levels = torch.linspace(0.001, 0.999, 999, device=pool.device)
    log_every = max(1, steps // 10)

    for step in range(1, steps + 1):
        row = torch.randperm(
            pool.size(0),
            generator=generator,
            device=pool.device,
        )[: context_size + query_size]
        context, query = pool[row].split(context_size)
        x_context = context.drop_columns(target_column)
        y_context = context[:, target_column]
        x_query = query.drop_columns(target_column)

        if task == "classification":
            y_query = query[:, target_column].categorical.code
            y_query = y_query.squeeze(-1).long()
        else:
            y_query = query[:, target_column].numerical.squeeze(-1)

        callback = _TrainingCallback(task, y_query, quantile_levels)
        optimizer.zero_grad()
        model(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            recipe=recipe,
            num_estimators=1,
            callbacks=[callback],
            generator=generator,
        )
        optimizer.step()

        if step % log_every == 0 or step == steps:
            assert callback.loss is not None
            print(f"  step {step:4d}/{steps}  loss={callback.loss.item():.4f}")


def evaluate(
    model: sdm.models.ICLModel,
    context: sdm.TableTensor,
    query: sdm.TableTensor,
    target_column: str,
    *,
    task: str,
    num_estimators: int,
    generator: torch.Generator,
) -> float:
    """Zero-shot in-context evaluation via the model's own default recipe."""
    model.eval()
    with torch.inference_mode():
        out = model(
            x_context=context.drop_columns(target_column),
            y_context=context[:, target_column],
            x_query=query.drop_columns(target_column),
            num_estimators=num_estimators,
            generator=generator,
        )
        if task == "classification":
            scores, indices = sdm.evaluation.to_class_indices(
                out,
                query[:, target_column],
                missing_score=0.0,
            )
            return (scores.argmax(dim=-1) == indices).float().mean().item()

        pred = out.numerical.mean(dim=-1)
        target = query[:, target_column].numerical.squeeze(-1)
        return (pred - target).pow(2).mean().sqrt().item()  # RMSE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--query-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    if args.task == "classification":
        frame = load_digits(as_frame=True).frame
        target_column = "target"
        stypes = sdm.infer_stypes(
            frame,
            overrides={target_column: "categorical"},
        )
    else:
        frame = fetch_california_housing(as_frame=True).frame
        target_column = "MedHouseVal"
        stypes = sdm.infer_stypes(frame)

    stratify = frame[target_column] if args.task == "classification" else None
    train_frame, eval_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify,
    )
    train_frame = train_frame.reset_index(drop=True)
    eval_frame = eval_frame.reset_index(drop=True)

    # Zero-shot evaluation always uses raw label/target values: the model's
    # own default recipe fits `AlignCategories`/`Standardize` on the context
    # it is given, so no manual target encoding is needed here.
    eval_context = sdm.TableTensor.from_pandas(
        df=train_frame.iloc[: args.context_size],
        stypes=stypes,
        device=device,
    )
    eval_query = sdm.TableTensor.from_pandas(
        df=eval_frame,
        stypes=stypes,
        device=device,
    )

    # The training pool pre-encodes the target once (stable category order
    # for classification; pool-level standardization for regression) so it
    # can be compared consistently against the model's raw per-step output.
    train_pool_frame = train_frame.copy()
    if args.task == "regression":
        target_mean = train_pool_frame[target_column].mean()
        target_std = train_pool_frame[target_column].std()
        train_pool_frame[target_column] = (
            train_pool_frame[target_column] - target_mean
        ) / target_std
    train_pool = sdm.TableTensor.from_pandas(
        df=train_pool_frame,
        stypes=stypes,
        device=device,
    )

    model = sdm.models.KumoTabular(task=args.task, size="small", device=device)

    baseline = evaluate(
        model,
        eval_context,
        eval_query,
        target_column,
        task=args.task,
        num_estimators=args.num_estimators,
        generator=generator,
    )
    metric = "accuracy" if args.task == "classification" else "RMSE"
    print(f"zero-shot {metric}: {baseline:.4f}")

    default_recipe = model.default_recipe()
    train_recipe = sp.Recipe(
        features=default_recipe.features,
        target=sp.Identity(),
        output=default_recipe.output,
    )
    print(f"fine-tuning for {args.steps} steps...")
    finetune(
        model,
        train_recipe,
        train_pool,
        target_column,
        task=args.task,
        steps=args.steps,
        context_size=args.context_size,
        query_size=args.query_size,
        lr=args.lr,
        generator=generator,
    )

    finetuned = evaluate(
        model,
        eval_context,
        eval_query,
        target_column,
        task=args.task,
        num_estimators=args.num_estimators,
        generator=generator,
    )
    print(f"fine-tuned {metric}: {finetuned:.4f}")


if __name__ == "__main__":
    main()
