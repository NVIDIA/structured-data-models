# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        # Kumo-TS scales features in double precision.
        features=[
            sp.Cast(torch.float64),
            sp.MinMaxScale(),
            sp.Cast(torch.float32),
        ],
    )
