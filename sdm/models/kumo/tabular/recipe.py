# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                numerical=[
                    sp.DropConstantColumns(),
                    sp.Standardize(eps=1e-6, dtype=torch.float64),
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
                    sp.FlipSign(),
                    sp.Callable(
                        lambda table: table.replace_blocks(
                            numerical=table.numerical.float()
                        )
                    ),
                ],
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.AddCategoryCounts(),
                    sp.ToNumerical(),
                    sp.DropConstantColumns(),
                    sp.Standardize(eps=1e-6, dtype=torch.float64),
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
                    sp.Callable(
                        lambda table: table.replace_blocks(
                            numerical=table.numerical.float()
                        )
                    ),
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
