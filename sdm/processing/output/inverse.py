# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor


class InvertTarget(EnsembleProcessor):
    """Invert the fitted target pipeline at this output processing step.

    Bound to ``Recipe.target`` during model execution. Member assignments must
    remain compatible with its fitted state. Classification predictions pass
    through unchanged.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self) -> None:
        super().__init__()
        self._target: EnsembleProcessor | None = None
        self._locations: tuple[tuple[int, int], ...] = ()
        self._ndim = 0

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.dim() == self._ndim:
            return self._transform_ensemble(
                EnsembleTable.from_table(table, num_members=1)
            )[0]

        outputs = self._transform_ensemble(
            EnsembleTable(
                groups=(table,),
                locations=tuple((0, i) for i in range(table.size(0))),
            )
        )
        if outputs.num_groups == 1:
            return outputs.expanded_group(0)
        return cast(
            TableTensor,
            torch.stack(
                tuple(outputs[i] for i in range(len(outputs))), dim=0
            ),
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if not isinstance(self._target, EnsembleInvertibleMixin):
            raise RuntimeError("Target recipe is not invertible")
        if (
            len(ensemble_table) == len(self._locations)
            and ensemble_table._locations != self._locations
        ):
            # Restore fitted groups without changing logical member order.
            groups: list[list[TableTensor]] = [
                [] for _ in range(max(i for i, _ in self._locations) + 1)
            ]
            for member_id, (group_id, _) in enumerate(self._locations):
                groups[group_id].append(ensemble_table[member_id])
            ensemble_table = EnsembleTable(
                groups=tuple(
                    cast(TableTensor, torch.stack(tuple(group), dim=0))
                    for group in groups
                ),
                locations=self._locations,
            )
        return self._target.inverse_transform_ensemble(ensemble_table)
