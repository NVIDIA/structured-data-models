# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import torch

import sdm.processing as sp

NumericalMissing = Literal["dispatch", "nan", "mix", "impute"]
RegressionReduction = Literal["scalar_trim", "quantile_trim"]


def _normalize() -> list[sp.Processor]:
    # Members cycle over three normalizations of the standardized columns,
    # fitted in double precision: in single precision the standardized
    # values of a column with a few huge outliers collapse onto one value.
    return [
        sp.Cast(torch.float64),
        sp.Standardize(eps=1e-6),
        sp.Clip(min_value=-100.0, max_value=100.0),
        sp.Choice(
            sp.Identity(),
            sp.PowerTransform(),
            sp.SquashTransform(),
            method="round_robin",
        ),
        sp.ClipSigma(threshold=4.0),
    ]


def _missing(numerical_missing: NumericalMissing) -> sp.Processor:
    # ``nan`` keeps missing cells for the model everywhere (best on
    # BeyondArena; the model is trained with missing values). ``dispatch``
    # imputes all but large, mostly incomplete tables, ``impute`` fills them
    # everywhere, ``mix`` alternates the two across the members.
    if numerical_missing == "dispatch":
        return sp.MissingDispatch(
            dense=sp.ImputeMean(),
            min_rows=20_000,
            min_row_frac=0.5,
        )
    if numerical_missing == "nan":
        return sp.Identity()
    if numerical_missing == "mix":
        return sp.Choice(sp.Identity(), sp.ImputeMean(), method="round_robin")
    return sp.ImputeMean()


def _regression_output(
    regression_reduction: RegressionReduction,
) -> list[sp.Processor]:
    # ``scalar_trim`` averages the quantiles of every member to a point
    # prediction, then takes the trimmed mean over the members.
    # ``quantile_trim`` takes the trimmed mean of every quantile over the
    # members first, then averages the trimmed quantiles.
    trim = sp.ReduceEstimators(method="trimmed")
    if regression_reduction == "quantile_trim":
        return [trim, sp.ReduceQuantiles()]
    return [sp.ReduceQuantiles(), trim]


def default_recipe(
    numerical_missing: NumericalMissing = "nan",
    regression_reduction: RegressionReduction = "scalar_trim",
) -> sp.Recipe:
    r"""Default recipe.

    Args:
        numerical_missing: How missing numerical cells reach the model, see
            :func:`_missing`.
        regression_reduction: Order of the quantile and ensemble reductions
            of regression outputs, see :func:`_regression_output`.
    """
    ecoc = sp.EncodeECOC(alphabet_size=10)
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                numerical=_missing(numerical_missing),
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.AddLevelCounts(min_cardinality=50),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.DropConstantColumns(),
                    *_normalize(),
                    sp.FlipSign(),
                    sp.Cast(torch.float32),
                ],
                # Category codes are normalized like numbers but never
                # sign-flipped.
                categorical=[
                    sp.ToNumerical(),
                    sp.DropConstantColumns(),
                    *_normalize(),
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
                    sp.AlignCategories(shared_categories=True),
                    ecoc,
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
                sp.DecodeECOC(ecoc),
                sp.ReduceEstimators(method="mean"),
                sp.Softmax(temperature=1.0),
            ],
            regression=_regression_output(regression_reduction),
        ),
    )
