# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch
from torch import Tensor
from torch.nn import ModuleDict

from sdm import EnsembleTable, Stype
from sdm.processing import EnsembleProcessor, Processor
from sdm.processing.common._positions import _ForwardEnsemblePositions


class TableDispatch(_ForwardEnsemblePositions, EnsembleProcessor):
    """Apply separate feature processors to task and related tables.

    :class:`TableDispatch` is resolved only during model execution.

    Args:
        task: Processor used for the task table.
        related: Processor used for related tables.
    """

    def __init__(
        self,
        *,
        task: object = None,
        related: object = None,
    ) -> None:
        super().__init__()
        self.processors: ModuleDict[EnsembleProcessor] = ModuleDict()
        for route, processor in (("task", task), ("related", related)):
            if processor is None:
                continue
            self.processors[route] = EnsembleProcessor.as_processor(processor)

        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )
        self._route: Literal["task", "related"] | None = None

    @property
    def handles_stypes(self) -> frozenset[Stype]:
        r""":meta private:"""  # noqa: D415
        return frozenset(
            stype
            for processor in self.processors.values()
            for stype in processor.handles_stypes
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        row_positions: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if self._route is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved table route; "
                "use it in a 'Recipe' through model execution"
            )
        if self._route in self.processors:
            self.processors[self._route].fit_ensemble(
                ensemble_table,
                row_positions=row_positions,
                generator=generator,
            )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        row_positions: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if self._route is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved table route; "
                "use it in a 'Recipe' through model execution"
            )
        if self._route not in self.processors:
            return ensemble_table
        return self.processors[self._route].fit_transform_ensemble(
            ensemble_table,
            row_positions=row_positions,
            generator=generator,
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        row_positions: Tensor | None = None,
    ) -> EnsembleTable:
        if self._route is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved table route; "
                "use it in a 'Recipe' through model execution"
            )
        if self._route not in self.processors:
            return ensemble_table
        return self.processors[self._route].transform_ensemble(
            ensemble_table, row_positions=row_positions
        )

    def get_extra_state(self) -> str | None:
        r""":meta private:"""  # noqa: D415
        return self._route

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._route = cast(Literal["task", "related"] | None, state)

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self.processors) == 0:
            return super().__repr__(indent=indent)

        reprs = []
        for route, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{route}={processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
