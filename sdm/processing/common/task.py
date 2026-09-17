# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch
from torch.nn import ModuleDict

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import EnsembleProcessor, Processor


class TaskDispatch(EnsembleProcessor):
    """Apply separate processors based on the semantic type of the target.

    :class:`TaskDispatch` is resolved only during model execution.

    Args:
        classification: Processor selected for a categorical target.
        regression: Processor selected for a numerical target.
    """

    def __init__(
        self,
        *,
        classification: object = None,
        regression: object = None,
    ) -> None:
        super().__init__()
        self.processors: ModuleDict[EnsembleProcessor] = ModuleDict()
        for task, processor in (
            ("classification", classification),
            ("regression", regression),
        ):
            if processor is None:
                continue
            processor = EnsembleProcessor.as_processor(processor)
            self.processors[task] = processor

        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )
        self._task: Literal["classification", "regression"] | None = None

    @property
    def handles_stypes(self) -> frozenset[Stype]:
        r""":meta private:"""  # noqa: D415
        return frozenset(
            stype
            for processor in self.processors.values()
            for stype in processor.handles_stypes
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if self._task is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved task; use it "
                "in a 'Recipe' through model execution"
            )
        if self._task in self.processors:
            self.processors[self._task].fit(table, generator=generator)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        if self._task is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved task; use it "
                "in a 'Recipe' through model execution"
            )
        if self._task not in self.processors:
            return table
        return self.processors[self._task].fit_transform(
            table,
            generator=generator,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        if self._task is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved task; use it "
                "in a 'Recipe' through model execution"
            )
        if self._task not in self.processors:
            return table
        return self.processors[self._task].transform(table)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if self._task is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved task; use it "
                "in a 'Recipe' through model execution"
            )
        if self._task in self.processors:
            self.processors[self._task].fit_ensemble(
                ensemble_table,
                generator=generator,
            )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if self._task is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved task; use it "
                "in a 'Recipe' through model execution"
            )
        if self._task not in self.processors:
            return ensemble_table
        return self.processors[self._task].fit_transform_ensemble(
            ensemble_table,
            generator=generator,
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self._task is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved task; use it "
                "in a 'Recipe' through model execution"
            )
        if self._task not in self.processors:
            return ensemble_table
        return self.processors[self._task].transform_ensemble(ensemble_table)

    def get_extra_state(self) -> str | None:
        r""":meta private:"""  # noqa: D415
        return self._task

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._task = cast(
            Literal["classification", "regression"] | None,
            state,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self.processors) == 0:
            return super().__repr__(indent=indent)

        reprs = []
        for task, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{task}={processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
