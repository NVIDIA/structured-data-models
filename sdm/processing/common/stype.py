# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch.nn import ModuleDict

from sdm import EnsembleData, EnsembleTable, Stype, TableLike
from sdm.processing import (
    InvertibleMixin,
    Processor,
)


class StypeDispatch(Processor, InvertibleMixin):
    r"""Apply separate processor pipelines to columns grouped by semantic type.

    Matching columns are passed to each configured route. Route outputs are
    concatenated in semantic type order; unconfigured types pass through.

    Inverse transform supports routes that preserve their semantic type. Every
    active route must be invertible, and routes must not share an output
    semantic type. Passthrough columns are preserved.

    Args:
        numerical: Processor or stateless callable route for numerical
            columns.
        categorical: Processor or stateless callable route for categorical
            columns.
        datetime: Processor or stateless callable route for datetime columns.
        text: Processor or stateless callable route for text columns.
        id: Processor or stateless callable route for identifier columns.
    """

    def __init__(
        self,
        *,
        numerical: object = None,
        categorical: object = None,
        datetime: object = None,
        id: object = None,
        text: object = None,
    ) -> None:
        super().__init__()
        self.processors: ModuleDict[Processor] = ModuleDict()
        for stype, processor in (
            (Stype.numerical, numerical),
            (Stype.categorical, categorical),
            (Stype.datetime, datetime),
            (Stype.text, text),
            (Stype.id, id),
        ):
            if processor is None:
                continue
            processor = Processor.as_processor(processor)
            self.processors[str(stype)] = processor

        self.handles_stypes = tuple(Stype(stype) for stype in self.processors)
        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )
        self._active_routes: tuple[str, ...] | None = None

    def get_extra_state(
        self,
    ) -> tuple[str, ...] | None:
        r""":meta private:"""  # noqa: D415
        return self._active_routes

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._active_routes = cast(tuple[str, ...] | None, state)

    def _find_active_routes(
        self,
        table: TableLike,
    ) -> tuple[str, ...]:
        active_stypes = (
            tuple(stype.value for stype in table.active_stypes)
            if isinstance(table, EnsembleTable)
            else tuple(
                stype.value
                for stype, columns in table.column_groups
                if columns
            )
        )
        return tuple(
            stype for stype in self.processors if stype in active_stypes
        )

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._active_routes = self._find_active_routes(table)
        for stype in self._active_routes:
            self.processors[stype].fit(
                table.select_stypes(stype),
                generator=generator,
            )

    def _fit_transform(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> TableLike:
        self._active_routes = self._find_active_routes(table)
        outputs = [
            self.processors[stype].fit_transform(
                table.select_stypes(stype),
                generator=generator,
            )
            for stype in self._active_routes
        ]
        outputs.append(self._remainder(table))
        nonempty = [output for output in outputs if output.active_stypes]
        if len(nonempty) == 1:
            return nonempty[0]
        return EnsembleData.execute(torch.cat, nonempty or outputs, dim=-1)

    def _transform(
        self,
        table: TableLike,
    ) -> TableLike:
        active_routes = self._active_routes
        if not self.requires_fit or active_routes is None:
            active_routes = self._find_active_routes(table)
            self._active_routes = active_routes
        outputs = [
            self.processors[stype].transform(table.select_stypes(stype))
            for stype in active_routes
        ]
        outputs.append(self._remainder(table))
        nonempty = [output for output in outputs if output.active_stypes]
        if len(nonempty) == 1:
            return nonempty[0]
        return EnsembleData.execute(torch.cat, nonempty or outputs, dim=-1)

    def _inverse_transform(
        self,
        table: TableLike,
    ) -> TableLike:
        active_routes = self._active_routes
        if not self.requires_fit or active_routes is None:
            active_routes = self._find_active_routes(table)
            self._active_routes = active_routes

        outputs = []
        for stype in active_routes:
            processor = self.processors[stype]
            # TODO This is wrong. A processor may not map to same stype back!
            outputs.append(
                processor.inverse_transform(table.select_stypes(stype))
            )

        outputs.append(self._remainder(table))
        nonempty = [output for output in outputs if output.active_stypes]
        if len(nonempty) == 1:
            return nonempty[0]
        return EnsembleData.execute(torch.cat, nonempty or outputs, dim=-1)

    def _remainder(
        self,
        table: TableLike,
    ) -> TableLike:
        return table.select_stypes(
            tuple(
                stype
                for stype in Stype.all()
                if str(stype) not in self.processors
            )
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self.processors) == 0:
            return super().__repr__(indent=indent)

        reprs = []
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{stype}={processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
