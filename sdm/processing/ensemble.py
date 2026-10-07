# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor

from sdm import EnsembleTable, TableTensor
from sdm.processing import InvertibleMixin, Processor


class EnsembleProcessor(Processor):
    r"""Base processor for ensemble-aware table transformations.

    An :class:`EnsembleProcessor` defines a reusable transformation on
    :class:`~sdm.EnsembleTable` for feature, target and output
    pre/post-processing across ensemble members.
    An :class:`EnsembleProcessor` learns any required state via
    :meth:`fit_ensemble` and applies the transformation via
    :meth:`transform_ensemble`. :meth:`fit_ensemble`,
    :meth:`transform_ensemble`, and :meth:`fit_transform_ensemble`
    are no-ops for stypes outside of
    :attr:`~sdm.processing.base.Processor.handles_stypes`.

    As a :class:`~sdm.processing.base.Processor`, it also accepts a
    :class:`~sdm.tensor.TableTensor` and processes it as an ensemble
    with one member.
    """

    @staticmethod
    def as_processor(processor: object) -> EnsembleProcessor:
        r"""Normalize a processor-like object to a :class:`EnsembleProcessor`.

        Args:
            processor: A processor-like object. An :class:`EnsembleProcessor`
                is returned as-is, and any ordinary processor-like object is
                adapted to ensemble processing.
        """
        from sdm.processing import EnsembleProcessorAdapter  # noqa: PLC0415

        processor = Processor.as_processor(processor)
        if isinstance(processor, EnsembleProcessor):
            return processor
        return EnsembleProcessorAdapter(processor)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_ensemble(
            EnsembleTable.from_table(table, num_members=1),
            generator=generator,
        )

    def _fit_with_positions(
        self,
        table: TableTensor,
        row_positions: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_ensemble_with_positions(
            EnsembleTable.from_table(table, num_members=1),
            row_positions,
            generator=generator,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        output = self._transform_ensemble(
            EnsembleTable.from_table(table, num_members=1)
        )
        return output[0]

    def _transform_with_positions(
        self, table: TableTensor, row_positions: Tensor
    ) -> TableTensor:
        output = self._transform_ensemble_with_positions(
            EnsembleTable.from_table(table, num_members=1), row_positions
        )
        return output[0]

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        if not self.requires_fit:
            return self._transform(table)
        output = self._fit_transform_ensemble(
            EnsembleTable.from_table(table, num_members=1),
            generator=generator,
        )
        return output[0]

    def _fit_transform_with_positions(
        self,
        table: TableTensor,
        row_positions: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        if not self.requires_fit:
            return self._transform_with_positions(table, row_positions)
        output = self._fit_transform_ensemble_with_positions(
            EnsembleTable.from_table(table, num_members=1),
            row_positions,
            generator=generator,
        )
        return output[0]

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError

    def _fit_ensemble_with_positions(
        self,
        ensemble_table: EnsembleTable,
        row_positions: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_ensemble(ensemble_table, generator=generator)

    @abc.abstractmethod
    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        pass

    def _transform_ensemble_with_positions(
        self, ensemble_table: EnsembleTable, row_positions: Tensor
    ) -> EnsembleTable:
        return self._transform_ensemble(ensemble_table)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if self.requires_fit:
            self._fit_ensemble(ensemble_table, generator=generator)
        return self._transform_ensemble(ensemble_table)

    def _fit_transform_ensemble_with_positions(
        self,
        ensemble_table: EnsembleTable,
        row_positions: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if (
            not self.requires_row_positions
            and type(self)._fit_ensemble_with_positions
            is EnsembleProcessor._fit_ensemble_with_positions
            and type(self)._transform_ensemble_with_positions
            is EnsembleProcessor._transform_ensemble_with_positions
        ):
            return self._fit_transform_ensemble(
                ensemble_table, generator=generator
            )
        if self.requires_fit:
            self._fit_ensemble_with_positions(
                ensemble_table, row_positions, generator=generator
            )
        return self._transform_ensemble_with_positions(
            ensemble_table, row_positions
        )

    def _check_ensemble_row_positions(
        self, ensemble_table: EnsembleTable, row_positions: Tensor | None
    ) -> None:
        if row_positions is None:
            if self.requires_row_positions:
                raise ValueError(
                    f"{self.__class__.__name__} requires row_positions"
                )
            return
        if any(
            row_positions.shape != (group.size(-2),)
            for group in ensemble_table._iter_groups()
        ):
            raise ValueError(
                "row_positions must have shape [R] for every group"
            )

    def fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        row_positions: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Self:
        """Fit the processor on an ensemble table.

        Args:
            ensemble_table: Ensemble table used to compute the processor state.
            row_positions: Shared row coordinates with shape ``[R]``.
            generator: Pseudorandom number generator used for sampling.
        """
        if not any(
            group.active_stypes & self.handles_stypes
            for group in ensemble_table._iter_groups()
        ):
            return self
        if self.requires_fit:
            self._check_ensemble_row_positions(ensemble_table, row_positions)
            if row_positions is None:
                self._fit_ensemble(ensemble_table, generator=generator)
            else:
                self._fit_ensemble_with_positions(
                    ensemble_table, row_positions, generator=generator
                )
            self._set_fitted(ensemble_table[0].device)
        return self

    def transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        row_positions: Tensor | None = None,
    ) -> EnsembleTable:
        """Transform an ensemble table with fitted state.

        Args:
            ensemble_table: Ensemble table to transform.
            row_positions: Shared row coordinates with shape ``[R]``.

        Returns:
            The transformed ensemble table.
        """
        if not any(
            group.active_stypes & self.handles_stypes
            for group in ensemble_table._iter_groups()
        ):
            return ensemble_table
        self._check_is_fitted()
        self._check_ensemble_row_positions(ensemble_table, row_positions)
        if row_positions is None:
            return self._transform_ensemble(ensemble_table)
        return self._transform_ensemble_with_positions(
            ensemble_table, row_positions
        )

    def fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        row_positions: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        """Fit the processor and transform an ensemble table.

        Args:
            ensemble_table: Ensemble table to fit on and transform.
            row_positions: Shared row coordinates with shape ``[R]``.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            The transformed ensemble table.
        """
        if not any(
            group.active_stypes & self.handles_stypes
            for group in ensemble_table._iter_groups()
        ):
            return ensemble_table
        self._check_ensemble_row_positions(ensemble_table, row_positions)
        if row_positions is None:
            output = self._fit_transform_ensemble(
                ensemble_table, generator=generator
            )
        else:
            output = self._fit_transform_ensemble_with_positions(
                ensemble_table, row_positions, generator=generator
            )
        if self.requires_fit:
            self._set_fitted(ensemble_table[0].device)
        return output


class EnsembleInvertibleMixin(InvertibleMixin):
    r"""Extend a :class:`EnsembleProcessor` by an inverse transformation."""

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        output = self._inverse_transform_ensemble(
            EnsembleTable.from_table(table, num_members=1)
        )
        return output[0]

    def _inverse_transform_with_positions(
        self, table: TableTensor, row_positions: Tensor
    ) -> TableTensor:
        output = self._inverse_transform_ensemble_with_positions(
            EnsembleTable.from_table(table, num_members=1), row_positions
        )
        return output[0]

    @abc.abstractmethod
    def _inverse_transform_ensemble(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable: ...

    def _inverse_transform_ensemble_with_positions(
        self, ensemble_table: EnsembleTable, row_positions: Tensor
    ) -> EnsembleTable:
        return self._inverse_transform_ensemble(ensemble_table)

    def inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        row_positions: Tensor | None = None,
    ) -> EnsembleTable:
        r"""Apply the inverse transformation to ``ensemble_table``.

        Args:
            ensemble_table: The ensemble table in transformed representation.
            row_positions: Shared row coordinates with shape ``[R]``.

        Returns:
            The table restored to the representation before
            :meth:`~EnsembleProcessor.transform_ensemble`.
        """
        self._check_is_fitted()
        self._check_ensemble_row_positions(ensemble_table, row_positions)
        if row_positions is None:
            return self._inverse_transform_ensemble(ensemble_table)
        return self._inverse_transform_ensemble_with_positions(
            ensemble_table, row_positions
        )

    if TYPE_CHECKING:

        def _check_ensemble_row_positions(
            self, ensemble_table: EnsembleTable, row_positions: Tensor | None
        ) -> None: ...
