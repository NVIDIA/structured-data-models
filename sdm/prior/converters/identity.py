# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior.converters import NumericalConverter


class IdentityConverter(NumericalConverter):
    """Return single-channel values as numerical columns."""

    def forward(  # noqa: D102
        self,
        slices: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        values = slices.values() if slices.is_nested else slices.squeeze(1)
        return values, slices
