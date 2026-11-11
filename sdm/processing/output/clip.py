# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import EnsembleTable, Stype, TableLike
from sdm.processing import TargetAwareProcessor


class ClipNonnegative(TargetAwareProcessor):
    r"""Clip outputs at zero for targets with nonnegative observations."""

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def _transform(self, table: TableLike) -> TableLike:
        if self._target is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no context targets; "
                "use it in a 'Recipe' through model execution"
            )
        target = (
            self._target.numerical.as_tensor()
            if isinstance(self._target, EnsembleTable)
            else self._target.numerical.unsqueeze(0)
        )
        mask = ~(target < 0).any(dim=-2).any(dim=0)
        mask &= (~target.isnan()).any(dim=-2).any(dim=0)
        mask = mask.unsqueeze(-2).unsqueeze(-1)
        numerical = table.numerical.unflatten(-1, (mask.size(-2), -1))
        numerical = torch.where(mask, numerical.clamp(min=0), numerical)
        return table.replace_blocks(numerical=numerical.flatten(-2))
