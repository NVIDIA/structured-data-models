# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full fine-tuning of KumoTabular ("kumo-small") or TabICLv2.

Fine-tunes every parameter of the selected model with gradient descent on
resampled in-context batches per iteration. Evaluates in-context against a
fixed validation split every epoch, checkpoints whenever that metric
improves, and reports the checkpointed model's performance on a held-out
test split once fine-tuning is done.
"""

import argparse
from pathlib import Path

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


def finetune(
    model: sdm.models.ICLModel,
    recipe: sp.Recipe,
    pool: sdm.TableTensor,
    val_context: sdm.TableTensor,
    val_query: sdm.TableTensor,
    target_column: str,
    *,
    task: str,
    max_epochs: int,
    steps_per_epoch: int,
    context_size: int,
    query_size: int,
    lr: float,
    num_estimators: int,
    checkpoint_path: Path,
    generator: torch.Generator,
) -> None:
    """Full fine-tune every parameter of ``model`` in place.

    Each step resamples a disjoint context/query batch from ``pool``, runs
    the model's differentiable ``forward()`` path, and backpropagates a
    supervised loss on the query rows computed inside a training callback.
    After every epoch, ``model`` is evaluated in-context against the fixed
    ``val_context``/``val_query`` split and checkpointed to
    ``checkpoint_path`` whenever that metric improves on the best seen so
    far. Before returning, ``model`` is restored to its best checkpoint.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    quantile_levels = torch.linspace(0.001, 0.999, 999, device=pool.device)
    higher_is_better = task == "classification"
    metric_name = "val_accuracy" if task == "classification" else "val_RMSE"

    def current_metric() -> float:
        return evaluate(
            model,
            val_context,
            val_query,
            target_column,
            task=task,
            num_estimators=num_estimators,
            generator=generator,
        )

    best_metric = current_metric()
    print(f"epoch   0 (zero-shot)  {metric_name}={best_metric:.4f}")
    torch.save(model.state_dict(), checkpoint_path)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
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
            assert callback.loss is not None
            total_loss += callback.loss.item()

        metric_value = current_metric()
        improved = (
            metric_value > best_metric
            if higher_is_better
            else metric_value < best_metric
        )
        status = "  (improved, checkpointed)" if improved else ""
        print(
            f"epoch {epoch:3d}/{max_epochs}  "
            f"loss={total_loss / steps_per_epoch:.4f}  "
            f"{metric_name}={metric_value:.4f}{status}"
        )
        if improved:
            best_metric = metric_value
            torch.save(model.state_dict(), checkpoint_path)

    best_state = torch.load(
        checkpoint_path,
        map_location=pool.device,
        weights_only=True,
    )
    model.load_state_dict(best_state)


def main() -> None:
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
    parser.add_argument(
        "--context-size",
        type=int,
        default=256,
        help="Context size sampled per fine-tuning iteration.",
    )
    parser.add_argument(
        "--query-size",
        type=int,
        default=128,
        help="Query size sampled per fine-tuning iteration.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path or Path(
        f"checkpoint-{args.model}-{args.task}.pt"
    )

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

    # 80% train (also the fine-tuning pool) / 10% val (checkpoint selection
    # during training) / 10% test (reported only once, after fine-tuning).
    stratify = frame[target_column] if args.task == "classification" else None
    train_frame, holdout_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify,
    )
    holdout_stratify = (
        holdout_frame[target_column] if args.task == "classification" else None
    )
    val_frame, test_frame = train_test_split(
        holdout_frame,
        test_size=0.5,
        random_state=args.seed,
        stratify=holdout_stratify,
    )
    train_frame = train_frame.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    # Zero-shot/validation/test evaluation always uses raw label/target
    # values: the model's own default recipe fits `AlignCategories`/
    # `Standardize` on the context it is given, so no manual target
    # encoding is needed here.
    context = sdm.TableTensor.from_pandas(
        df=train_frame,
        stypes=stypes,
        device=device,
    )
    val_query = sdm.TableTensor.from_pandas(
        df=val_frame,
        stypes=stypes,
        device=device,
    )
    test_query = sdm.TableTensor.from_pandas(
        df=test_frame,
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

    if args.model == "kumo-small":
        model = sdm.models.KumoTabular(
            task=args.task,
            size="small",
            device=device,
        )
    else:
        model = sdm.models.TabICLv2(task=args.task, device=device)

    default_recipe = model.default_recipe()
    train_recipe = sp.Recipe(
        features=default_recipe.features,
        target=sp.Identity(),
        output=default_recipe.output,
    )
    finetune(
        model,
        train_recipe,
        train_pool,
        context,
        val_query,
        target_column,
        task=args.task,
        max_epochs=args.max_epochs,
        steps_per_epoch=args.steps_per_epoch,
        context_size=args.context_size,
        query_size=args.query_size,
        lr=args.lr,
        num_estimators=args.num_estimators,
        checkpoint_path=checkpoint_path,
        generator=generator,
    )

    test_metric = evaluate(
        model,
        context,
        test_query,
        target_column,
        task=args.task,
        num_estimators=args.num_estimators,
        generator=generator,
    )
    metric_name = "accuracy" if args.task == "classification" else "RMSE"
    print(f"test_{metric_name}={test_metric:.4f}")


if __name__ == "__main__":
    main()
