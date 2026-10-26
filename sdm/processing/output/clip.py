# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import EnsembleTable, Stype
from sdm.processing import TargetAwareEnsembleProcessor


class ClipNonnegative(TargetAwareEnsembleProcessor):
    r"""Clip outputs at zero for targets with nonnegative observations."""

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self._target is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no context targets; "
                "use it in a 'Recipe' through model execution"
            )

        negatives, observed = [], []
        for target in self._target._iter_groups():
            negatives.append((target.numerical < 0).any(dim=-2).any(dim=0))
            observed.append((~target.numerical.isnan()).any(dim=-2).any(dim=0))

        mask = ~torch.stack(negatives).any(dim=0)
        mask &= torch.stack(observed).any(dim=0)
        mask = mask.unsqueeze(-2).unsqueeze(-1)

        groups = []
        for group in ensemble_table._iter_groups():
            values = group.numerical.unflatten(-1, (mask.size(-2), -1))
            values = torch.where(mask, values.clamp(min=0), values)
            groups.append(group.replace_blocks(numerical=values.flatten(-2)))
        return ensemble_table.replace_groups(groups)
