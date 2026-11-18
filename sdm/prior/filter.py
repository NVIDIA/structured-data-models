# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import Task


def should_drop(
    x: Tensor,
    y: Tensor,
    *,
    task_type: Task,
    drop_trivial: bool = False,
    drop_unlearnable: bool = False,
    trivial_threshold: float = 0.05,
    num_trees: int = 25,
    num_resamples: int = 200,
    max_rows: int | None = 10_000,
) -> bool:
    """Decide whether to drop a dataset based on an ExtraTrees probe.

    Compare out-of-bag squared errors with a mean baseline,
    one-hot encoding classification targets. A bootstrap test measures
    whether the probe improves on the baseline.

    Args:
        x: Features with shape ``[R, C]`` for rows and columns.
        y: Targets with shape ``[R]``; classification labels are
            nonnegative integer codes.
        task_type: Classification or regression task.
        drop_trivial: Discard datasets solved almost perfectly by the probe.
        drop_unlearnable: Discard datasets where the probe does not
            significantly improve on the mean baseline.
        trivial_threshold: Maximum probe-to-baseline RMSE ratio for a trivial
            dataset.
        num_trees: Number of trees in the probe.
        num_resamples: Number of bootstrap resamples for the improvement test.
        max_rows: Maximum rows per tree bootstrap and deterministic
            scoring subset. ``None`` uses all rows.

    Returns:
        Whether the dataset should be dropped.
    """
    if not (drop_trivial or drop_unlearnable):
        return False
    from sklearn.ensemble import ExtraTreesRegressor  # noqa: PLC0415

    score_index: np.ndarray | None = None
    if max_rows is not None and x.size(0) > max_rows:
        score_index = np.sort(
            np.random.default_rng(0).choice(
                x.size(0), size=max_rows, replace=False
            )
        )

    # Use a float [num_rows, num_targets] matrix for the shared MSE probe.
    # Binary classification needs only one of the two one-hot columns.
    if task_type == Task.classification:
        labels = y.to(torch.long)
        num_classes = int(labels.max().item() + 1)
        targets = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
        if num_classes == 2:
            targets = targets[:, :1]
    else:
        targets = y.reshape(-1, 1).to(torch.float32)

    # Give missing values a distinct value without changing training data.
    features = x.to(torch.float).nan_to_num(nan=-9999.0).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    probe = ExtraTreesRegressor(
        n_estimators=num_trees,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
        random_state=1,
        max_depth=6,
        max_samples=(max_rows if score_index is not None else None),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Some inputs do not have OOB scores",
            category=UserWarning,
        )
        probe.fit(
            features,
            (
                np.squeeze(targets, axis=-1)
                if targets.shape[-1] == 1
                else targets
            ),
        )
    predictions = probe.oob_prediction_
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if score_index is not None:
        predictions = predictions[score_index]
        targets = targets[score_index]

    valid_rows = ~np.isnan(predictions).any(axis=1)
    valid_targets = targets[valid_rows]
    baseline_errors = (
        (valid_targets - targets.mean(axis=0, keepdims=True)) ** 2
    ).sum(axis=1)
    probe_errors = ((valid_targets - predictions[valid_rows]) ** 2).sum(axis=1)

    is_trivial = bool(
        np.sqrt(probe_errors.mean())
        <= trivial_threshold * np.sqrt(baseline_errors.mean())
    )
    if drop_trivial and is_trivial:
        return True

    if not drop_unlearnable:
        return False

    improvements = baseline_errors - probe_errors
    resample_index = np.random.default_rng(0).integers(
        0,
        len(improvements),
        size=(num_resamples, len(improvements)),
    )
    p_value = float(np.mean(improvements[resample_index].mean(axis=1) <= 0.0))
    return p_value >= 0.05
