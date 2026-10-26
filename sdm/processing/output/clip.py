# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import EnsembleTable, Stype
from sdm.processing import TargetAwareEnsembleProcessor


class ClipNonnegative(TargetAwareEnsembleProcessor):
    r"""Clip outputs at zero for targets with nonnegative observations."""

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _mask: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_mask", torch.empty(0, dtype=torch.bool))

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        nonnegative = []
        observed = []
        for target in ensemble_table._iter_groups():
            values = target.numerical
            valid = ~values.isnan()
            nonnegative.append((values.ge(0) | ~valid).all(dim=-2).all(dim=0))
            observed.append(valid.any(dim=-2).any(dim=0))
        nonnegative = torch.stack(nonnegative).all(dim=0)
        observed = torch.stack(observed).any(dim=0)
        self._set_buffer(
            name="_mask",
            value=(nonnegative & observed).unsqueeze(-2).unsqueeze(-1),
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:

        groups = []
        for group in ensemble_table._iter_groups():
            values = group.numerical.unflatten(-1, (self._mask.size(-2), -1))
            values = torch.where(self._mask, values.clamp_min(0), values)
            groups.append(group.replace_blocks(numerical=values.flatten(-2)))
        return ensemble_table.replace_groups(groups)
