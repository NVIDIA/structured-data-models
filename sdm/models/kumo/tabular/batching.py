# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal, cast

import torch

from sdm import TableTensor, Task
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.model import KumoTabular, _KumoTabular


def estimate_batch_size(
    model: KumoTabular,
    x: TableTensor,
    y: TableTensor,
    *,
    memory_budget: int,
    num_estimators: int = 1,
    mode: Literal["fit", "predict"] = "fit",
    estimator_batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Estimate an ensemble batch for fitting or query rows for prediction.

    For default-recipe inference. Uses half the budget and returns at least
    one item; the estimate does not guarantee that it fits in memory.

    Args:
        model: Kumo Tabular model with the target task initialized.
        x: Unprocessed context features of shape ``[R, C]``.
        y: Context targets of shape ``[R, 1]``. Categorical metadata defines
            all classes; numerical targets select regression.
        memory_budget: Available bytes excluding weights, inputs, and context
            caches, including overlapping transfers during prediction.
        num_estimators: Total number of ensemble members.
        mode: ``"fit"`` estimates ensemble members; ``"predict"`` query rows.
        estimator_batch_size: Ensemble members run together during prediction.
            Ignored for fitting.
        dtype: Neural execution or autocast dtype.

    Returns:
        At least one item, capped at ``num_estimators`` for fitting.
    """
    num_rows, num_columns = x.shape[-2:]
    num_classes = (
        len(y.categorical.categories[0]) if y.categorical.size(-1) else 0
    )
    workspace, cache = _row_bytes(
        model=model,
        num_columns=num_columns,
        num_classes=num_classes,
        dtype=dtype,
    )
    budget = memory_budget // 2
    if mode == "fit":
        capacity = budget // max(num_rows * (workspace + cache), 1)
        return max(1, min(num_estimators, capacity))
    if mode == "predict":
        # Preprocessing and output reduction retain all ensemble members.
        row_bytes = workspace * estimator_batch_size
        row_bytes += (
            num_estimators * 8 * (8 * num_columns + 4 * (num_classes or 999))
        )
        return max(1, budget // max(row_bytes, 1))
    raise AssertionError(f"Unknown batch estimation mode: {mode!r}")


def _row_bytes(
    model: KumoTabular,
    num_columns: int,
    num_classes: int,
    dtype: torch.dtype,
) -> tuple[int, int]:
    task = Task.classification if num_classes else Task.regression
    network = cast(_KumoTabular, model.models[task])
    row = network.row_embedding
    icl = network.icl_block
    layer = cast(KumoTabularTransformerBlock, icl.layers[0])
    tasks = 1
    if num_classes and num_classes > model.ecoc.max_classes:
        tasks = max(
            math.ceil(num_classes / (model.ecoc.max_classes - 1)),
            4 * math.ceil(math.log(num_classes, model.ecoc.max_classes)),
        )
    # The default recipe adds count columns and keeps at most 500 features.
    columns = min(2 * num_columns, 500)
    element_size = dtype.itemsize
    workspace = tasks * (
        element_size * 4 * (columns + row.readout_token.size(0)) * row.channels
        + layer.peak_bytes_per_example(
            element_size=element_size, query_length=1
        )
    )
    # Fit projects all heads before retaining the smaller query KV heads.
    cache = tasks * element_size * 2 * layer.attn.q_dim * len(icl.layers)
    return workspace, cache
