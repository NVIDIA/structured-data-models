# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import torch

import sdm.processing as sp

Normalize = Literal["round_robin", "identity", "power", "squash", "quantile"]


def _transform(normalize: Normalize) -> sp.Processor:
    if normalize == "identity":
        return sp.Identity()
    if normalize == "power":
        return sp.PowerTransform()
    if normalize == "squash":
        return sp.SquashTransform()
    if normalize == "quantile":
        return sp.QuantileTransform(n_quantiles=100)
    return sp.Choice(
        sp.Identity(),
        sp.PowerTransform(),
        sp.SquashTransform(),
        method="round_robin",
    )


def _normalize(normalize: Normalize) -> list[sp.Processor]:
    # Standardized columns are normalized in double precision: in single
    # precision the standardized values of a column with a few huge
    # outliers collapse onto one value.
    return [
        sp.Cast(torch.float64),
        sp.Standardize(eps=1e-6),
        sp.Clip(min_value=-100.0, max_value=100.0),
        _transform(normalize),
        sp.ClipSigma(threshold=4.0),
    ]


def default_recipe(
    normalize: Normalize = "round_robin",
    shuffle_categories_max: int | None = None,
) -> sp.Recipe:
    r"""Default recipe.

    ``normalize`` is the numeric transform members use, by default a cycle
    over identity, power and squash. ``shuffle_categories_max`` permutes the
    codes of categorical columns with at most that many levels per member.
    """
    shuffle: list[sp.Processor] = []
    if shuffle_categories_max is not None:
        shuffle = [sp.ShuffleCategories(max_categories=shuffle_categories_max)]
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                # Large, mostly incomplete tables keep their missing cells
                # for the model; every other table is imputed.
                numerical=sp.MissingDispatch(
                    dense=sp.ImputeMean(),
                    min_rows=20_000,
                    min_row_frac=0.5,
                ),
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    *shuffle,
                    sp.AddLevelCounts(min_cardinality=50),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.DropConstantColumns(),
                    *_normalize(normalize),
                    sp.FlipSign(),
                    sp.Cast(torch.float32),
                ],
                # Category codes are normalized like numbers but never
                # sign-flipped.
                categorical=[
                    sp.ToNumerical(),
                    sp.DropConstantColumns(),
                    *_normalize(normalize),
                    sp.Cast(torch.float32),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.ShuffleColumns(method="latin"),
                    sp.SelectColumns(500, method="first"),
                ],
            ),
        ],
        target=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=[
                    sp.Standardize(),
                    sp.FlipSign(),
                ],
            ),
        ],
        output=sp.TaskDispatch(
            classification=[
                sp.ReduceEstimators(method="mean"),
                sp.Softmax(temperature=1.0),
            ],
            regression=[
                sp.ReduceQuantiles(),
                sp.ReduceEstimators(method="trimmed"),
            ],
        ),
    )
