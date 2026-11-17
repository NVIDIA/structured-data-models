# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from sdm import Task
from sdm.prior import HP, Correlated, LogUniformInt


@dataclass(frozen=True)
class Schema:
    """Specification of a dataset.

    Args:
        column_dims: One entry per feature: ``1`` for numerical features,
            otherwise the number of possible categories.
        target_dim: ``1`` for regression, or the number of possible classes
            for classification.
    """

    column_dims: tuple[int, ...]
    target_dim: int

    @property
    def task_type(self) -> Task:
        """Return the task implied by the target size."""
        return Task.classification if self.target_dim > 1 else Task.regression

    @property
    def num_columns(self) -> int:
        """Return the feature count."""
        return len(self.column_dims)


class SchemaSampler:
    """Sample :class:`Schema` instances.

    Args:
        task_type: Prediction task.
        num_columns: Feature count.
        cat_ratio: Categorical fraction, clipped to
            ``[0, 1]``.
        max_cardinality: Upper limit on category counts.
        num_classes: Class count, required for classification.
        independent_cat_ratio: Fraction of categorical
            features with independent cardinalities, clipped to ``[0, 1]``.
        max_correlated_cardinality: Additional cap on correlated cardinalities.
    """

    def __init__(
        self,
        task_type: Task,
        num_columns: HP[int] | int,
        cat_ratio: HP[float] | float,
        max_cardinality: HP[int] | int,
        num_classes: HP[int] | int | None = None,
        independent_cat_ratio: HP[float] | float = 0.0,
        max_correlated_cardinality: HP[int] | int = 9,
    ) -> None:
        if task_type == Task.classification and num_classes is None:
            raise ValueError("Classification tasks require 'num_classes'")

        self.task_type = task_type
        self.num_columns = HP.from_value(num_columns)
        self.cat_ratio = HP.from_value(cat_ratio)
        self.max_cardinality = HP.from_value(max_cardinality)
        self.independent_cat_ratio = HP.from_value(independent_cat_ratio)
        self.max_correlated_cardinality = HP.from_value(
            max_correlated_cardinality
        )
        self.num_classes: HP[int] | None = None
        if num_classes is not None:
            self.num_classes = HP.from_value(num_classes)

    def sample(
        self,
        generator: torch.Generator | None = None,
        *,
        num_columns: int | None = None,
        num_classes: int | None = None,
    ) -> Schema:
        """Sample a schema.

        Args:
            generator: Pseudorandom generator.
            num_columns: Override the feature count for this draw.
            num_classes: Override the class count for a classification draw.
        """
        if num_columns is None:
            num_columns = self.num_columns.sample(generator=generator)
        cat_ratio = self.cat_ratio.sample(generator=generator)
        max_k = self.max_cardinality.sample(generator=generator)
        independent_cat_ratio = self.independent_cat_ratio.sample(
            generator=generator
        )
        max_correlated_k = min(
            max_k,
            self.max_correlated_cardinality.sample(generator=generator),
        )

        cat_ratio = min(max(cat_ratio, 0.0), 1.0)
        independent_cat_ratio = min(max(independent_cat_ratio, 0.0), 1.0)
        num_categorical = round(num_columns * cat_ratio)
        num_independent = round(num_categorical * independent_cat_ratio)

        perm = torch.randperm(num_columns, device="cpu", generator=generator)
        categorical = perm[:num_categorical]
        independent = categorical[:num_independent]
        correlated = categorical[num_independent:]

        is_independent = torch.zeros(
            num_columns, dtype=torch.bool, device="cpu"
        )
        is_independent[independent] = True
        is_correlated = torch.zeros(
            num_columns, dtype=torch.bool, device="cpu"
        )
        is_correlated[correlated] = True

        column_dims = torch.ones(num_columns, dtype=torch.long, device="cpu")
        if num_independent:
            column_dims[is_independent] = LogUniformInt(2, max_k).sample_batch(
                size=num_independent,
                generator=generator,
            )
        num_correlated = num_categorical - num_independent
        if num_correlated:
            column_dims[is_correlated] = Correlated(
                LogUniformInt(2, max_correlated_k)
            ).sample_batch(
                size=num_correlated,
                generator=generator,
            )

        target_dim = 1
        if self.task_type == Task.classification:
            assert self.num_classes is not None
            target_dim = (
                self.num_classes.sample(generator=generator)
                if num_classes is None
                else num_classes
            )
            assert target_dim >= 2

        return Schema(
            column_dims=tuple(column_dims.tolist()),
            target_dim=target_dim,
        )
