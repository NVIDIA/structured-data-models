# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared full fine-tuning core for the tabular benchmark harness.

Fine-tunes every parameter of an :class:`~sdm.models.ICLModel` with gradient
descent, using ``model.train()`` to make ``forward()`` differentiable
end-to-end against the model's own default recipe (no training ``Callback``
or ``Recipe`` override needed). Generic over ``x``/``y`` :class:`TableTensor`
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
    max_epochs: int = 150,
    iters_per_epoch: int = 10,
    train_size: int = 10_000,
    context_frac: float = 0.8,
    val_frac: float = 0.2,
    lr: float = 1e-5,
    num_estimators: int,
    max_val_context_size: int | None = None,
    generator: torch.Generator | None,
) -> None:
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    quantile_levels = torch.linspace(0.001, 0.999, 999, device=device)
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
