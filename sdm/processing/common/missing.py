# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch
from torch.nn import ModuleDict

from sdm import EnsembleTable, Stype
from sdm.processing import EnsembleProcessor, Processor


class MissingDispatch(EnsembleProcessor):
    """Apply separate processors based on the fitting table's missingness.

    The route is decided at fit on the numerical block of the first
    member's fitting table. With ``rows`` its number of rows per batch
    element, ``cell_frac`` the fraction of NaN cells and ``row_frac`` the
    fraction of rows containing at least one NaN, both pooled over all
    leading batch dimensions, the ``sparse`` processor is selected iff
    ``rows >= min_rows`` and ``row_frac >= min_row_frac`` and
    ``cell_frac >= min_cell_frac``, and the ``dense`` processor otherwise.
    A route left ``None`` passes the table through unchanged.

    Args:
        sparse: Processor selected when the fitting table meets every
            threshold.
        dense: Processor selected otherwise.
        min_rows: Minimum number of rows for the ``sparse`` route.
        min_row_frac: Minimum fraction of rows containing at least one NaN
            for the ``sparse`` route.
        min_cell_frac: Minimum fraction of NaN numerical cells for the
            ``sparse`` route.
    """

    requires_fit = True

    def __init__(
        self,
        *,
        sparse: object = None,
        dense: object = None,
        min_rows: int = 0,
        min_row_frac: float = 0.0,
        min_cell_frac: float = 0.0,
    ) -> None:
        super().__init__()
        self.processors: ModuleDict[EnsembleProcessor] = ModuleDict()
        for route, processor in (("sparse", sparse), ("dense", dense)):
            if processor is None:
                continue
            self.processors[route] = EnsembleProcessor.as_processor(processor)

        self.min_rows = min_rows
        self.min_row_frac = min_row_frac
        self.min_cell_frac = min_cell_frac
        self._route: Literal["sparse", "dense"] | None = None

    @property
    def route(self) -> Literal["sparse", "dense"] | None:
        """The route decided at fit, or ``None`` before fitting."""
        return self._route

    @property
    def handles_stypes(self) -> frozenset[Stype]:
        r""":meta private:"""  # noqa: D415
        return frozenset(
            stype
            for processor in self.processors.values()
            for stype in processor.handles_stypes
        ) | {Stype.numerical}

    def _resolve(self, ensemble_table: EnsembleTable) -> None:
        missing = ensemble_table.table(0).numerical.isnan()
        rows = missing.size(-2)
        if missing.numel() == 0:
            cell_frac = row_frac = 0.0
        else:
            missing_rows = missing.any(dim=-1)
            cells, rows_with_nan = torch.stack(
                (missing.sum(), missing_rows.sum())
            ).tolist()
            cell_frac = cells / missing.numel()
            row_frac = rows_with_nan / missing_rows.numel()
        self._route = (
            "sparse"
            if rows >= self.min_rows
            and row_frac >= self.min_row_frac
            and cell_frac >= self.min_cell_frac
            else "dense"
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._resolve(ensemble_table)
        if self._route in self.processors:
            self.processors[self._route].fit_ensemble(
                ensemble_table,
                generator=generator,
            )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self._resolve(ensemble_table)
        if self._route not in self.processors:
            return ensemble_table
        return self.processors[self._route].fit_transform_ensemble(
            ensemble_table,
            generator=generator,
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self._route not in self.processors:
            return ensemble_table
        return self.processors[self._route].transform_ensemble(ensemble_table)

    def get_extra_state(self) -> str | None:
        r""":meta private:"""  # noqa: D415
        return self._route

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._route = cast(Literal["sparse", "dense"] | None, state)

    def __repr__(self, *, indent: int = 0) -> str:
        reprs = []
        for route, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{route}={processor_repr}")
        for name, value, default in (
            ("min_rows", self.min_rows, 0),
            ("min_row_frac", self.min_row_frac, 0.0),
            ("min_cell_frac", self.min_cell_frac, 0.0),
        ):
            if value != default:
                reprs.append(f"{' ' * (indent + 2)}{name}={value}")
        if len(reprs) == 0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
