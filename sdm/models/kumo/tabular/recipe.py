# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    ecoc = sp.ECOCCategories(alphabet_size=10)
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value", min_frequency=2),
                    sp.ToNumerical(),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.DropConstantColumns(),
                    sp.Standardize(eps=1e-6),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    sp.Choice(
                        sp.Identity(),
                        sp.PowerTransform(),
                        method="round_robin",
                    ),
                    sp.ClipSigma(threshold=4.0),
                    sp.FlipSign(),
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
        output=[
            sp.TaskDispatch(regression=sp.SortQuantiles()),
            sp.DecodeECOC(ecoc),
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(classification=sp.Softmax(temperature=1.0)),
        ],
    )
