# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full fine-tuning of KumoTabular ("kumo-small") or TabICLv2.

Fine-tunes every parameter of the selected model with gradient descent on
resampled in-context batches per iteration, using the model's own default
recipe throughout (``model.train()`` makes ``forward()`` differentiable
end-to-end, including post-processing, so no custom recipe or callback is
needed). Evaluates in-context against a fixed validation split every epoch,
checkpoints whenever that metric improves, and reports the checkpointed
model's performance on a held-out test split once fine-tuning is done.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split

import sdm

_EPS = 1e-12


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

    Each step resamples a disjoint context/query batch from ``pool`` and
    runs ``model.train()``-mode ``forward()``, which returns an
    already-decoded prediction (real class names / original-scale
    quantiles) that's differentiable end-to-end. After every epoch,
    ``model`` is evaluated in-context against the fixed ``val_context``/
    ``val_query`` split and checkpointed to ``checkpoint_path`` whenever
    that metric improves on the best seen so far. Before returning,
    ``model`` is restored to its best checkpoint.
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
            y_query = query[:, target_column]

            optimizer.zero_grad()
            out = model(
                x_context=context.drop_columns(target_column),
                y_context=context[:, target_column],
                x_query=query.drop_columns(target_column),
                num_estimators=1,
                generator=generator,
            )
            if task == "classification":
                scores, indices = sdm.evaluation.to_class_indices(
                    out,
                    y_query,
                    missing_score=0.0,
                )
                loss = F.nll_loss(scores.clamp_min(_EPS).log(), indices.long())
            else:
                # Pinball loss over the model's 999 fixed quantile levels,
                # already back in the target's original scale.
                target = y_query.numerical.squeeze(-1).unsqueeze(-1)
                diff = target - out.numerical
                loss = torch.maximum(
                    quantile_levels * diff,
                    (quantile_levels - 1) * diff,
                ).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

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

    # The model's own default recipe (AlignCategories/Standardize/etc.)
    # handles both training and evaluation now that forward() is
    # differentiable end-to-end in train mode, so context/pool share one
    # table built straight from raw labels -- no manual pre-encoding.
    train_pool = sdm.TableTensor.from_pandas(
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

    if args.model == "kumo-small":
        model = sdm.models.KumoTabular(
            task=args.task,
            size="small",
            device=device,
        )
    else:
        model = sdm.models.TabICLv2(task=args.task, device=device)

    finetune(
        model,
        train_pool,
        train_pool,
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
        train_pool,
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
