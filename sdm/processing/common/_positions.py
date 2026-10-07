# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from sdm import EnsembleTable


class _ForwardEnsemblePositions:
    """Forward row coordinates through an ensemble processor's own hooks."""

    if TYPE_CHECKING:

        def _fit_ensemble(
            self,
            ensemble_table: EnsembleTable,
            row_positions: Tensor | None = None,
            *,
            generator: torch.Generator | None = None,
        ) -> None: ...

        def _fit_transform_ensemble(
            self,
            ensemble_table: EnsembleTable,
            row_positions: Tensor | None = None,
            *,
            generator: torch.Generator | None = None,
        ) -> EnsembleTable: ...

        def _transform_ensemble(
            self,
            ensemble_table: EnsembleTable,
            row_positions: Tensor | None = None,
        ) -> EnsembleTable: ...

        def _inverse_transform_ensemble(
            self,
            ensemble_table: EnsembleTable,
            row_positions: Tensor | None = None,
        ) -> EnsembleTable: ...

    def _fit_ensemble_with_positions(
        self,
        ensemble_table: EnsembleTable,
        row_positions: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_ensemble(ensemble_table, row_positions, generator=generator)

    def _fit_transform_ensemble_with_positions(
        self,
        ensemble_table: EnsembleTable,
        row_positions: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        return self._fit_transform_ensemble(
            ensemble_table, row_positions, generator=generator
        )

    def _transform_ensemble_with_positions(
        self, ensemble_table: EnsembleTable, row_positions: Tensor
    ) -> EnsembleTable:
        return self._transform_ensemble(ensemble_table, row_positions)

    def _inverse_transform_ensemble_with_positions(
        self, ensemble_table: EnsembleTable, row_positions: Tensor
    ) -> EnsembleTable:
        return self._inverse_transform_ensemble(ensemble_table, row_positions)
