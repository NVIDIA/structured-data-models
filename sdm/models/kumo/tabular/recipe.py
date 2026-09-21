# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    def numerical_processor() -> sp.Sequential:
        return sp.Sequential(
            sp.DropConstantColumns(),
            sp.Standardize(eps=1e-6),
            sp.Clip(min_value=-100.0, max_value=100.0),
            sp.Choice(
                sp.Identity(),
                sp.PowerTransform(),
                [
                    sp.RobustScale(),
                    sp.ClipSoft(3.0),
                ],
                method="round_robin",
            ),
            sp.ClipSigma(threshold=4.0),
        )

    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                numerical=[
                    numerical_processor(),
                    sp.FlipSign(),
                ],
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.AddCategoryCounts(),
                    sp.ToNumerical(),
                    numerical_processor(),
                ],
            ),
            sp.ShuffleColumns(method="latin"),
            sp.SelectColumns(500, method="first"),
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
                sp.Softmax(),
            ],
            regression=[
                sp.SortQuantiles(),
                sp.ReduceEstimators(
                    method="trimmed_mean",
                    proportion=0.2,
                ),
            ],
        ),
    )
