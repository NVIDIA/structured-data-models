# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared full fine-tuning core for the tabular benchmark harness.

Fine-tunes every parameter of an :class:`~sdm.models.ICLModel` with gradient
descent. Generic over ``x``/``y`` :class:`TableTensor`
pools built however each benchmark adapter already builds them, so the same
:func:`full_finetune` call can be dropped in right before each adapter's
existing ``model.fit(...)`` call.
"""

import copy
from typing import Literal

import torch
import torch.nn.functional as F

import sdm

_EPS = 1e-12

# Single source of truth for the fine-tuning defaults shared by every
# benchmark adapter (TabArena/BeyondArena, ScoringBench, TALENT).
FINETUNE_EPOCHS = 75
FINETUNE_ITERS_PER_EPOCH = 10
FINETUNE_LR = 1e-6
FINETUNE_TRAIN_SIZE = 10_000
FINETUNE_CONTEXT_FRAC = 0.8
FINETUNE_VAL_FRAC = 0.2


def kumo_small_binary_epochs(
    default: int,
    *,
    is_kumo_small: bool,
    is_binary: bool,
) -> int:
    """Epoch count for fine-tuning, with a kumo-small binary override.

    Empirically found to need fewer epochs than the shared default to avoid
    overfitting on binary classification.
    """
    return 50 if is_kumo_small and is_binary else default


def evaluate(
    model: sdm.models.ICLModel,
    context_x: sdm.TableTensor,
    context_y: sdm.TableTensor,
    query_x: sdm.TableTensor,
    query_y: sdm.TableTensor,
    *,
    task: str,
    num_estimators: int,
    generator: torch.Generator | None,
) -> float:
    """Zero-shot in-context evaluation via the model's own default recipe."""
    model.eval()
    with torch.inference_mode():
        out = model(
            x_context=context_x,
            y_context=context_y,
            x_query=query_x,
            num_estimators=num_estimators,
            generator=generator,
        )
        if task == "classification":
            scores, indices = sdm.evaluation.to_class_indices(
                out,
                query_y,
                missing_score=0.0,
            )
            return (scores.argmax(dim=-1) == indices).float().mean().item()

        pred = out.numerical.mean(dim=-1)
        target = query_y.numerical.squeeze(-1)
        return (pred - target).pow(2).mean().sqrt().item()  # RMSE


def full_finetune(
    model: sdm.models.ICLModel,
    x_pool: sdm.TableTensor,
    y_pool: sdm.TableTensor,
    *,
    task: Literal["classification", "regression"],
    max_epochs: int = FINETUNE_EPOCHS,
    iters_per_epoch: int = FINETUNE_ITERS_PER_EPOCH,
    train_size: int = FINETUNE_TRAIN_SIZE,
    context_frac: float = FINETUNE_CONTEXT_FRAC,
    val_frac: float = FINETUNE_VAL_FRAC,
    lr: float = FINETUNE_LR,
    num_estimators: int,
    max_val_context_size: int | None = None,
    generator: torch.Generator | None,
) -> float:
    """Full fine-tune every parameter of ``model`` in place.

    Carves a fixed ``val_frac`` validation split out of ``x_pool``/``y_pool``
    up front; the remainder is the training portion. Each epoch runs
    ``iters_per_epoch`` training iterations before validating/checkpointing
    once. Each iteration independently resamples up to ``train_size`` rows
    from the training portion (or all of it, if smaller), splits that sample
    ``context_frac``/``1 - context_frac`` into context/query, and runs one
    differentiable ``model(...)`` forward/backward call. Validation always
    uses the full training portion as context (optionally capped by
    ``max_val_context_size``) against the held-out val split as query.
    Leaves ``model`` in eval mode holding its best-validation-metric weights.
    Returns that best validation metric (accuracy for classification, pinball
    loss for regression -- lower is better for regression).
    """
    device = x_pool.device
    n = x_pool.size(0)

    perm = torch.randperm(n, generator=generator, device=device)
    n_val = int(val_frac * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    x_train, y_train = x_pool[train_idx], y_pool[train_idx]
    x_val, y_val = x_pool[val_idx], y_pool[val_idx]

    val_context_x, val_context_y = x_train, y_train
    if (
        max_val_context_size is not None
        and val_context_x.size(0) > max_val_context_size
    ):
        cap = torch.randperm(
            val_context_x.size(0), generator=generator, device=device
        )
        cap = cap[:max_val_context_size]
        val_context_x, val_context_y = val_context_x[cap], val_context_y[cap]

    n_train = x_train.size(0)
    epoch_size = min(train_size, n_train)
    n_context = int(context_frac * epoch_size)

    if n_val < 1:
        raise ValueError(
            f"val_frac={val_frac} leaves no validation rows out of "
            f"{n} pool rows; pass a larger val_frac or pool."
        )
    if n_train < 2:
        raise ValueError(
            f"Only {n_train} training rows remain after the validation "
            "split; need at least 2 to form a non-empty context and query."
        )
    if n_context < 1 or epoch_size - n_context < 1:
        raise ValueError(
            f"context_frac={context_frac} splits {epoch_size} sampled rows "
            f"into an empty context or query ({n_context} / "
            f"{epoch_size - n_context}); pick a context_frac strictly "
            "between 0 and 1 relative to the sample size."
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    higher_is_better = task == "classification"

    def current_metric() -> float:
        return evaluate(
            model,
            val_context_x,
            val_context_y,
            x_val,
            y_val,
            task=task,
            num_estimators=num_estimators,
            generator=generator,
        )

    best_metric = current_metric()
    best_state = copy.deepcopy(model.state_dict())

    for _epoch in range(max_epochs):
        model.train()
        for _iter in range(iters_per_epoch):
            row = torch.randperm(n_train, generator=generator, device=device)
            row = row[:epoch_size]
            x_context, y_context = (
                x_train[row[:n_context]],
                y_train[row[:n_context]],
            )
            x_query, y_query = (
                x_train[row[n_context:]],
                y_train[row[n_context:]],
            )

            optimizer.zero_grad()
            out = model(
                x_context=x_context,
                y_context=y_context,
                x_query=x_query,
                num_estimators=1,
                generator=generator,
            )
            if task == "classification":
                scores, indices = sdm.evaluation.to_class_indices(
                    out,
                    y_query,
                    missing_score=0.0,
                )
                loss = F.nll_loss(scores.clamp_min(_EPS).log(), indices)
            else:
                # Pinball loss over the model's quantile output columns,
                # already back in the target's original scale; a model with
                # a single point-estimate column (e.g. TabFM) has no
                # quantiles to score against, so this reduces to squared
                # error instead.
                diff = y_query.numerical - out.numerical
                n_quantiles = out.numerical.size(-1)
                if n_quantiles > 1:
                    levels = torch.linspace(
                        0.001, 0.999, n_quantiles, device=device
                    )
                    loss = torch.maximum(
                        levels * diff,
                        (levels - 1) * diff,
                    ).mean()
                else:
                    loss = diff.pow(2).mean()
            loss.backward()
            optimizer.step()

        metric_value = current_metric()
        improved = (
            metric_value > best_metric
            if higher_is_better
            else metric_value < best_metric
        )
        if improved:
            best_metric = metric_value
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return best_metric
