# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm.processing as sp


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


def default_recipe() -> sp.Recipe:  # noqa: D103
    ecoc = sp.EncodeECOC(alphabet_size=10)
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
            regression=[
                sp.ReduceQuantiles(),
                sp.ReduceEstimators(method="trimmed"),
            ],
        ),
    )
