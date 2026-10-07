# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import EnsembleProcessor


class InvertTarget(EnsembleProcessor):
    """Invert the fitted target pipeline at this output processing step.

    Resolved during recipe execution; classification predictions pass through.
    Without ``member``, inputs retain their leading estimator dimension.
    After reducing estimators, select the target state with ``member``.
    A single fitted estimator also supports reduced inputs without selection.

    Args:
        member: Fitted estimator whose target state applies to the whole input.
            If ``None``, invert each estimator with its own target state.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self, *, member: int | None = None) -> None:
        super().__init__()
        self.member = member
        self._inverse: (
            Callable[[TableTensor, int | None], TableTensor] | None
        ) = None

    def _transform(self, table: TableTensor) -> TableTensor:
        if self._inverse is None:
            raise RuntimeError(
                "'InvertTarget' has no fitted target; use it in "
                "'Recipe.output' through model execution"
            )
        return self._inverse(table, self.member)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table.replace_groups(
            [self._transform(group) for group in ensemble_table._iter_groups()]
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if self.member is None:
            return super().__repr__(indent=indent)
        return f"{' ' * indent}{self.__class__.__name__}(member={self.member})"
