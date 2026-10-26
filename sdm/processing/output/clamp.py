# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import EnsembleTable, Stype
from sdm.processing import TargetAwareEnsembleProcessor


class TargetClamper(TargetAwareEnsembleProcessor):
    r"""Clamp predictions to zero for targets with nonnegative observations.

    NaNs are ignored; targets without observations are left unchanged.
    Apply after estimator reduction. When estimators have different contexts,
    observations across all contexts determine whether each target is clamped.
    Numerical output columns must be grouped by target in context column order.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self._target is None:
            raise RuntimeError(
                "'TargetClamper' has no context targets; use it in a 'Recipe' "
                "through model execution"
            )

        nonnegative = []
        observed = []
        for target in self._target._iter_groups():
            values = target.numerical
            valid = ~values.isnan()
            nonnegative.append((values.ge(0) | ~valid).all(dim=-2).all(dim=0))
            observed.append(valid.any(dim=-2).any(dim=0))
        nonnegative = torch.stack(nonnegative).all(dim=0)
        observed = torch.stack(observed).any(dim=0)
        nonnegative = (nonnegative & observed).unsqueeze(-2).unsqueeze(-1)

        groups = []
        for group in ensemble_table._iter_groups():
            values = group.numerical.unflatten(
                -1, (self._target[0].numerical.size(-1), -1)
            )
            values = torch.where(nonnegative, values.clamp_min(0), values)
            groups.append(group.replace_blocks(numerical=values.flatten(-2)))
        return ensemble_table.replace_groups(groups)
