# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import cast

import torch

from sdm import TableTensor, Task
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.model import KumoTabular, _KumoTabular
from sdm.nn import InducedTransformerBlock


def estimate_fit_batch_size(
    model: KumoTabular,
    x: TableTensor,
    y: TableTensor,
    *,
    memory_budget: int,
    num_estimators: int = 1,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Estimate how many ensemble members to fit together.

    For default-recipe inference. Uses half the budget; the estimate does
    not guarantee that it fits in memory.

    Args:
        model: Kumo Tabular model with the target task initialized.
        x: Unprocessed context features of shape ``[R, C]``.
        y: Context targets of shape ``[R, 1]``. Categorical metadata defines
            all classes; numerical targets select regression.
        memory_budget: Available bytes excluding weights and inputs.
        num_estimators: Total number of ensemble members.
        dtype: Neural execution or autocast dtype.

    Returns:
        At least one member, capped at ``num_estimators``. Pass this as
        ``estimator_batch_size`` to :meth:`~sdm.models.KumoTabular.fit`.
    """
    num_rows, num_columns = x.shape[-2:]
    num_classes = (
        len(y.categorical.categories[0]) if y.categorical.size(-1) else 0
    )
    workspace, cache, member = _row_bytes(
        model=model,
        num_columns=num_columns,
        num_classes=num_classes,
        dtype=dtype,
    )
    capacity = (memory_budget // 2) // max(
        num_rows * (workspace + cache) + member, 1
    )
    return max(1, min(num_estimators, capacity))


def estimate_predict_batch_size(
    model: KumoTabular,
    x: TableTensor,
    y: TableTensor,
    *,
    memory_budget: int,
    num_estimators: int = 1,
    estimator_batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Estimate how many query rows to predict together.

    For default-recipe inference. Uses half the budget; the estimate does
    not guarantee that it fits in memory.

    Args:
        model: Kumo Tabular model with the target task initialized.
        x: Unprocessed context features of shape ``[R, C]``.
        y: Context targets of shape ``[R, 1]``. Categorical metadata defines
            all classes; numerical targets select regression.
        memory_budget: Available bytes excluding weights, inputs, and context
            caches, including overlapping transfers.
        num_estimators: Total number of ensemble members used for fitting.
        estimator_batch_size: Ensemble members run together, as set during fit.
        dtype: Neural execution or autocast dtype.

    Returns:
        At least one query row per prediction call.
    """
    num_columns = x.size(-1)
    num_classes = (
        len(y.categorical.categories[0]) if y.categorical.size(-1) else 0
    )
    workspace, _, _ = _row_bytes(
        model=model,
        num_columns=num_columns,
        num_classes=num_classes,
        dtype=dtype,
    )
    # Preprocessing and output reduction retain all ensemble members.
    row_bytes = workspace * estimator_batch_size
    row_bytes += (
        num_estimators * 8 * (8 * num_columns + 4 * (num_classes or 999))
    )
    return max(1, (memory_budget // 2) // max(row_bytes, 1))


def _row_bytes(
    model: KumoTabular,
    num_columns: int,
    num_classes: int,
    dtype: torch.dtype,
) -> tuple[int, int, int]:
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
    # Fit records column-block caches per estimator, not per row.
    inducing = sum(
        cast(InducedTransformerBlock, block).inducing_points.size(0)
        for block in row.col_blocks
    )
    member = tasks * element_size * 2 * columns * inducing * row.channels
    return workspace, cache, member
