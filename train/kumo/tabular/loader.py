# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from sdm import Stype, Task
from sdm.prior import UniformInt, should_drop
from sdm.prior.tabular import SCMPrior
from sdm.prior.tabular.transform._table import (
    categorical_mask,
    category_values,
    feature_order,
)

COLUMN_BUCKETS = (*range(8, 97, 8), 100)

MAX_COLUMNS_LADDER = (
    (10_240, None),
    (20_000, 80),
    (30_000, 60),
    (40_000, 40),
    (50_000, 30),
    (60_000, 20),
    (65_000, 15),
    (None, 10),
)


class SamplingGroup(NamedTuple):
    """Dense batch of sampled tables with a shared row count and context size.

    Feature columns are padded; ``num_columns`` records each table's valid
    width and ``categorical_mask`` identifies categorical columns.
    """

    x: torch.Tensor  # [B, R, padded_columns]
    y: torch.Tensor  # [B, R]
    train_size: int
    categorical_mask: torch.Tensor  # [B, padded_columns]
    num_columns: torch.Tensor  # [B]
    max_column_width: int


@dataclass
class PriorDataset(IterableDataset[list[SamplingGroup]]):
    prior: SCMPrior
    batch_size: int
    group_size: int
    min_rows: int
    max_rows: int
    min_train_size: float
    max_train_size: float
    replay_small: bool
    seed: int
    rank: int
    world_size: int
    keep_trivial_prob: float = 0.5
    min_regression_train_std: float = 0.1
    robust_regression_probe: bool = True
    max_columns_ladder: tuple[tuple[int | None, int | None], ...] = (
        MAX_COLUMNS_LADDER
    )

    def __iter__(self) -> Iterator[list[SamplingGroup]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        # Layout draws match across ranks; content streams remain distinct.
        rng = np.random.default_rng([self.seed, worker_id])
        seed = self.seed + self.rank + worker_id * self.world_size
        random.seed(seed)
        np.random.seed(seed % 2**32)  # noqa: NPY002
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        trivial_rng = np.random.default_rng(seed + 0x7E1)
        columns = self.prior.schema_sampler.num_columns
        assert isinstance(columns, UniformInt)
        column_buckets = [
            width
            for width in COLUMN_BUCKETS
            if columns.low <= width < columns.high
        ]
        column_buckets.append(columns.high)
        while True:
            low, high = self.min_rows, self.max_rows
            if self.replay_small:
                probability = rng.random()
                if probability < 0.05:
                    low = high = min(int(rng.integers(200, 1_000)), high)
                elif probability < 0.3:
                    low, high = min(1_000, high), min(10_000, high)
            if low == high:
                num_rows = low
            else:
                num_rows = int(np.exp(rng.uniform(np.log(low), np.log(high))))
            max_columns = columns.high
            for limit, cap in self.max_columns_ladder:
                if limit is None or num_rows <= limit:
                    if cap is not None:
                        max_columns = min(max_columns, cap)
                    break
            max_columns = max(columns.low, max_columns)
            groups = []
            # Sample and pad each group; normalize on context rows only.
            for _ in range(self.batch_size // self.group_size):
                fraction = torch.rand(
                    (), generator=generator, dtype=torch.float64
                ).item()
                fraction = self.min_train_size + fraction * (
                    self.max_train_size - self.min_train_size
                )
                split = min(max(int(num_rows * fraction), 1), num_rows - 1)
                classes = self.prior.schema_sampler.num_classes
                num_classes = (
                    classes.sample(generator=generator)
                    if classes is not None
                    else None
                )
                column_draw = int(rng.integers(columns.low, max_columns + 1))
                bucket = bisect_left(column_buckets, column_draw)
                max_column_width = min(column_buckets[bucket], max_columns)
                column_sampler = UniformInt(
                    low=column_buckets[bucket - 1] + 1
                    if bucket
                    else columns.low,
                    high=max_column_width,
                )
                samples = [
                    self._sample_dataset(
                        column_sampler=column_sampler,
                        num_classes=num_classes,
                        num_rows=num_rows,
                        train_size=split,
                        generator=generator,
                        trivial_rng=trivial_rng,
                    )
                    for _ in range(self.group_size)
                ]
                width = max(x.size(-1) for x, _, _ in samples)
                x_pad = samples[0][0].new_zeros(
                    self.group_size, num_rows, width
                )
                categorical_mask = torch.zeros(
                    self.group_size,
                    width,
                    dtype=torch.bool,
                    device=x_pad.device,
                )
                num_columns = torch.tensor(
                    [x.size(-1) for x, _, _ in samples], device=x_pad.device
                )
                targets = []
                for index, (x, y, categorical) in enumerate(samples):
                    x_pad[index, :, : x.size(-1)] = x
                    categorical_mask[index, : x.size(-1)] = categorical
                    targets.append(y)
                x_pad = _standardize_features(x_pad, split)
                target = torch.stack(targets)
                if target.is_floating_point():
                    context = target[:, :split]
                    mean = context.mean(dim=-1, keepdim=True)
                    scale = context.std(dim=-1, keepdim=True, correction=0)
                    scale = torch.where(
                        scale < 1e-8, torch.ones_like(scale), scale
                    )
                    target = (target - mean) / scale
                groups.append(
                    SamplingGroup(
                        x=x_pad,
                        y=target,
                        train_size=split,
                        categorical_mask=categorical_mask,
                        num_columns=num_columns,
                        max_column_width=max_column_width,
                    )
                )
            yield groups

    def _sample_dataset(
        self,
        column_sampler: UniformInt,
        num_classes: int | None,
        num_rows: int,
        train_size: int,
        generator: torch.Generator,
        trivial_rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample features, targets, and a categorical mask."""
        num_columns = column_sampler.sample(generator=generator)
        for attempt in range(100):
            schema = self.prior.schema_sampler.sample(
                num_columns=num_columns,
                num_classes=num_classes,
                generator=generator,
            )
            sample = self.prior.sample(
                num_rows=num_rows,
                schema=schema,
                generator=generator,
            )
            if sample is None:
                continue
            x, y = sample
            if not x.active_stypes <= {Stype.numerical, Stype.categorical}:
                raise ValueError(
                    "Training features must be numerical or categorical"
                )
            features = torch.cat(
                (
                    x.numerical,
                    category_values(x.categorical, x.numerical.dtype),
                ),
                dim=-1,
            )[..., feature_order(x)]
            categorical = categorical_mask(x)
            if y.categorical.size(-1):
                target = y.categorical.code.squeeze(-1).long()
                # Both splits must contain the same classes.
                candidate = target
                for permutation in range(11):
                    if permutation:
                        order = torch.randperm(num_rows, generator=generator)
                        candidate = target[order]
                    context = candidate[:train_size].unique()
                    query = candidate[train_size:].unique()
                    if len(context) >= 2 and torch.equal(context, query):
                        if permutation:
                            features, target = features[order], candidate
                        break
                else:
                    continue
                target = target.unique(return_inverse=True)[1]
                task_type = Task.classification
            else:
                target = y.numerical.squeeze(-1)
                task_type = Task.regression
                if (
                    target.std(correction=0) < 1e-8
                    or target[:train_size].std(correction=0)
                    <= self.min_regression_train_std
                ):
                    continue
            # NaN is a distinct value; entirely missing columns are constant.
            keep = (features[:train_size] != features[:1]).any(dim=0)
            keep &= ~features[:train_size].isnan().all(dim=0)
            features, categorical = features[:, keep], categorical[keep]
            if features.size(-1) == 0:
                continue
            # Check only the first five attempts to bound probe fits.
            if attempt < 5:
                keep_trivial = self.keep_trivial_prob > 0.0 and (
                    self.keep_trivial_prob >= 1.0
                    or trivial_rng.random() < self.keep_trivial_prob
                )
                if should_drop(
                    x=features,
                    y=(
                        _robust_probe_target(target)
                        if task_type == Task.regression
                        and self.robust_regression_probe
                        else target
                    ),
                    task_type=task_type,
                    drop_trivial=not keep_trivial,
                    drop_unlearnable=True,
                ):
                    continue
            return features, target, categorical
        raise RuntimeError(
            "No valid dataset after 100 attempts "
            f"(rows={num_rows}, split={train_size})"
        )


def _standardize_features(x: torch.Tensor, train_size: int) -> torch.Tensor:
    values = x.double()
    context = values[..., :train_size, :]
    context = context.masked_fill(~context.isfinite(), torch.nan)
    mean = context.nanmean(dim=-2, keepdim=True).nan_to_num(nan=0.0)
    var = (context - mean).square().nanmean(dim=-2, keepdim=True)
    scale = var.nan_to_num(nan=0.0).sqrt() + 1e-6
    return ((values - mean) / scale).clamp(-100.0, 100.0).to(x.dtype)


def _robust_probe_target(y: torch.Tensor) -> torch.Tensor:
    centered = y - y.median()
    mad = 1.4826 * centered.abs().median()
    std = y.std(correction=0)
    scale = torch.where(
        mad > 0, mad, torch.where(std > 0, std, torch.ones_like(std))
    )
    return (centered / scale).clamp(-4.0, 4.0)
