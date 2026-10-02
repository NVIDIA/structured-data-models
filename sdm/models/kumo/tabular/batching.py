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
    """Estimate a batch size for Kumo Tabular's default recipe.

    For ``mode="fit"``, return the number of ensemble members to run together
    as ``fit(..., estimator_batch_size=...)``. For ``mode="predict"``, return
    the number of query rows to pass to each ``predict`` call. Read context
    dimensions and target classes from table metadata without processing data.

    This inference heuristic uses half the supplied budget and returns at
    least one item, even when it may not fit. Custom recipes and autograd
    execution require their own estimates; this is not an OOM guarantee.

    Args:
        model: Kumo Tabular model with the context's target task initialized.
        x: Unprocessed context features of shape ``[R, C]``, with ``R`` rows
            and ``C`` feature columns.
        y: Context targets of shape ``[R, 1]``. A categorical target's category
            vector defines all classes, including those absent from its rows.
            A numerical target selects regression with 999 output quantiles.
        memory_budget: Available device bytes after weights and input tables.
            For prediction, also subtract resident or staged context caches,
            including overlapping transfers. The caller determines this budget.
        num_estimators: Total number of ensemble members used for fitting.
        mode: Whether to estimate an ensemble batch for ``"fit"`` or a query
            row batch for ``"predict"``.
        estimator_batch_size: Maximum number of ensemble members run together
            during prediction, as selected during fitting. Ignored for fitting.
        dtype: Neural execution dtype, including any autocast setting.

    Returns:
        For fitting, a batch size between one and ``num_estimators``. For
        prediction, at least one query row; cap it to the query table's size.
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
