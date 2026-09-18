# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def _normalize() -> list[sp.Processor]:
    return [
        sp.Callable(
            lambda table: table.replace_blocks(
                numerical=table.numerical.double()
            )
        ),
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
    ]


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.AddCategoryCounts(),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.DropConstantColumns(),
                    *_normalize(),
                    sp.FlipSign(),
                    sp.Callable(
                        lambda table: table.replace_blocks(
                            numerical=table.numerical.float()
                        )
                    ),
                ],
                categorical=[
                    sp.ToNumerical(),
                    sp.DropConstantColumns(),
                    *_normalize(),
                    sp.Callable(
                        lambda table: table.replace_blocks(
                            numerical=table.numerical.float()
                        )
                    ),
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
                sp.ReduceEstimators(
                    method="trimmed_mean",
                    proportion=0.2,
                ),
            ],
        ),
    )
