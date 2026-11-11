# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import torch
from torch import Tensor

from sdm import EnsembleData, Stype, TableLike
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _ndtri

BOUNDS_THRESH = 1e-7
_MAX_NUM_COLS = 32


def _batched_interp(
    values: Tensor,
    boundaries: Tensor,
    references: Tensor,
) -> Tensor:
    # Batched 1D linear interpolation over independent feature grids:
    # values ``[F, N]`` with boundaries/references as ``[Q]`` or ``[F, Q]``.
    n = boundaries.shape[-1]
    if n == 1:
        if references.dim() == 1:
            return references.select(0, 0).expand_as(values)
        return references[:, :1].expand_as(values)

    if boundaries.dim() == 2 and boundaries.stride(0) == 0:
        boundaries = boundaries[0]
    if references.dim() == 2 and references.stride(0) == 0:
        references = references[0]
    idx = torch.searchsorted(boundaries, values, right=True).clamp(1, n - 1)

    if boundaries.dim() == 1:
        x0 = boundaries.take(idx - 1)
        x1 = boundaries.take(idx)
        lower_boundary = boundaries.select(0, 0)
        upper_boundary = boundaries.select(0, -1)
    else:
        x0 = boundaries.gather(1, idx - 1)
        x1 = boundaries.gather(1, idx)
        lower_boundary = boundaries[:, :1]
        upper_boundary = boundaries[:, -1:]

    if references.dim() == 1:
        y0 = references.take(idx - 1)
        y1 = references.take(idx)
        lower = references.select(0, 0)
        upper = references.select(0, -1)
    else:
        y0 = references.gather(1, idx - 1)
        y1 = references.gather(1, idx)
        lower = references[:, :1]
        upper = references[:, -1:]

    denom = x1 - x0
    weight = torch.where(denom != 0, (values - x0) / denom, 0.0)
    result = torch.lerp(y0, y1, weight)

    result = torch.where(values <= lower_boundary, lower, result)
    return torch.where(values >= upper_boundary, upper, result)


class QuantileTransform(Processor, InvertibleMixin):
    """Map numerical columns through their empirical quantiles.

    QuantileTransform grids are capped by the number of fitted rows and, when
    ``subsample`` is set, by ``20%`` of the subsample size to keep dense grids
    tractable.

    The subsample rows are drawn from the ``generator`` passed to ``fit()``;
    without one, they are drawn from the data device's global generator.

    Args:
        n_quantiles: Maximum number of quantiles to compute.
        subsample: Maximum number of rows to use for quantile computation.
        output_distribution: Distribution to map the empirical quantiles to.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _quantiles: Tensor
    _references: Tensor

    def __init__(
        self,
        *,
        n_quantiles: int = 1000,
        subsample: int | None = 10_000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
    ) -> None:
        super().__init__()
        if n_quantiles <= 0:
            raise ValueError("n_quantiles must be positive.")
        if subsample is not None and subsample <= 0:
            raise ValueError("subsample must be positive or None.")
        self._n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.register_buffer(
            "_quantiles",
            torch.empty(0),
        )
        self.register_buffer(
            "_references",
            torch.empty(0),
        )

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        quantiles, references = EnsembleData.execute(
            self._grid, table.numerical, generator=generator, batched=True
        )
        self.register_buffer("_quantiles", quantiles)
        self.register_buffer("_references", references)

    def _grid(
        self, numerical: Tensor, *, generator: torch.Generator | None
    ) -> tuple[Tensor, Tensor]:
        if numerical.size(-1) == 0:
            return (
                numerical.new_empty((*numerical.shape[:-2], 1, 0)),
                numerical.new_zeros((*numerical.shape[:-2], 1)),
            )
        n_samples = numerical.size(-2)
        quantile_limit = n_samples
        if self.subsample is not None:
            # Keep quantiles well below the subsample size; very dense
            # percentile grids are slow to fit and add little resolution.
            quantile_limit = min(
                quantile_limit,
                int(self.subsample * 0.2),
            )
        n_quantiles = max(
            1,
            min(self._n_quantiles, quantile_limit),
        )
        references = torch.linspace(
            0,
            1,
            n_quantiles,
            device=numerical.device,
            dtype=numerical.dtype,
        )

        if self.subsample is not None and self.subsample < n_samples:
            indices = torch.randperm(
                n_samples, generator=generator, device=numerical.device
            )[: self.subsample]
            input_sample = numerical.index_select(-2, indices)
        else:
            input_sample = numerical

        sample_size = input_sample.size(-2)
        sample_columns = (
            input_sample.movedim(-1, -2).reshape(-1, sample_size).contiguous()
        )
        quantiles = torch.quantile(
            sample_columns,
            references,
            dim=-1,
        )
        quantiles = (
            quantiles.movedim(0, -1)
            .reshape(
                *numerical.shape[:-2],
                numerical.size(-1),
                n_quantiles,
            )
            .movedim(-1, -2)
        )
        return quantiles, references.expand(*numerical.shape[:-2], n_quantiles)

    def _transform(self, table: TableLike) -> TableLike:
        numerical = EnsembleData.execute(
            self._transform_values,
            table.numerical,
            self._quantiles,
            self._references,
            batched=True,
        )
        return table.replace_blocks(numerical=numerical)

    def _transform_values(
        self, numerical: Tensor, quantiles: Tensor, references: Tensor
    ) -> Tensor:
        n_samples = numerical.size(-2)
        n_features = numerical.size(-1)
        input_columns = (
            numerical.movedim(-1, -2).reshape(-1, n_samples).contiguous()
        )
        quantile_columns = (
            quantiles.movedim(-1, -2)
            .reshape(-1, references.size(-1))
            .contiguous()
        )
        reference_columns = (
            references.unsqueeze(-2)
            .expand(*quantiles.shape[:-2], n_features, references.size(-1))
            .reshape(-1, references.size(-1))
        )
        transformed_columns = torch.empty_like(input_columns)

        for start in range(0, input_columns.size(0), _MAX_NUM_COLS):
            end = min(start + _MAX_NUM_COLS, input_columns.size(0))
            input_chunk = input_columns.narrow(0, start, end - start)
            quantile_chunk = quantile_columns.narrow(0, start, end - start)
            reference_chunk = reference_columns.narrow(0, start, end - start)
            lower_bound_x = quantile_chunk[:, :1]
            upper_bound_x = quantile_chunk[:, -1:]
            if self.output_distribution == "normal":
                bounds_thresh = input_chunk.new_tensor(BOUNDS_THRESH)
                lower_bounds_idx = input_chunk - bounds_thresh < lower_bound_x
                upper_bounds_idx = input_chunk + bounds_thresh > upper_bound_x
            elif self.output_distribution == "uniform":
                lower_bounds_idx = input_chunk == lower_bound_x
                upper_bounds_idx = input_chunk == upper_bound_x
            else:
                raise ValueError(
                    "output_distribution must be 'uniform' or 'normal'."
                )

            forward = _batched_interp(
                input_chunk,
                quantile_chunk,
                reference_chunk,
            )
            backward = _batched_interp(
                -input_chunk,
                -quantile_chunk.flip(1),
                -reference_chunk.flip(-1),
            )
            output = 0.5 * (forward - backward)

            output = torch.where(upper_bounds_idx, 1.0, output)
            output = torch.where(lower_bounds_idx, 0.0, output)

            if self.output_distribution == "normal":
                eps = input_chunk.new_tensor(
                    BOUNDS_THRESH - torch.finfo(torch.float64).eps
                )
                output = _ndtri(output)
                clip_min = _ndtri(eps)
                clip_max = _ndtri(1.0 - eps)
                output = output.clamp(clip_min, clip_max)

            transformed_columns[start:end] = output

        return transformed_columns.reshape(
            *numerical.shape[:-2],
            n_features,
            n_samples,
        ).movedim(-1, -2)

    def _inverse_transform(self, table: TableLike) -> TableLike:
        numerical = EnsembleData.execute(
            self._inverse_transform_values,
            table.numerical,
            self._quantiles,
            self._references,
            batched=True,
        )
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform_values(
        self, numerical: Tensor, quantiles: Tensor, references: Tensor
    ) -> Tensor:
        n_samples = numerical.size(-2)
        n_features = numerical.size(-1)
        input_columns = (
            numerical.movedim(-1, -2).reshape(-1, n_samples).contiguous()
        )
        quantile_columns = (
            quantiles.movedim(-1, -2)
            .reshape(-1, references.size(-1))
            .contiguous()
        )
        reference_columns = (
            references.unsqueeze(-2)
            .expand(*quantiles.shape[:-2], n_features, references.size(-1))
            .reshape(-1, references.size(-1))
        )
        inverse_columns = torch.empty_like(input_columns)

        for start in range(0, input_columns.size(0), _MAX_NUM_COLS):
            end = min(start + _MAX_NUM_COLS, input_columns.size(0))
            input_chunk = input_columns.narrow(0, start, end - start)
            quantile_chunk = quantile_columns.narrow(0, start, end - start)
            reference_chunk = reference_columns.narrow(0, start, end - start)
            lower_bound_y = quantile_chunk[:, :1]
            upper_bound_y = quantile_chunk[:, -1:]
            if self.output_distribution == "normal":
                input_chunk = torch.special.ndtr(input_chunk)

            if self.output_distribution == "normal":
                bounds_thresh = input_chunk.new_tensor(BOUNDS_THRESH)
                lower_bounds_idx = input_chunk - bounds_thresh < 0.0
                upper_bounds_idx = input_chunk + bounds_thresh > 1.0
            elif self.output_distribution == "uniform":
                lower_bounds_idx = input_chunk == 0.0
                upper_bounds_idx = input_chunk == 1.0
            else:
                raise ValueError(
                    "output_distribution must be 'uniform' or 'normal'."
                )

            output = _batched_interp(
                input_chunk,
                reference_chunk,
                quantile_chunk,
            )
            output = torch.where(upper_bounds_idx, upper_bound_y, output)
            output = torch.where(lower_bounds_idx, lower_bound_y, output)
            inverse_columns[start:end] = output

        return inverse_columns.reshape(
            *numerical.shape[:-2],
            n_features,
            n_samples,
        ).movedim(-1, -2)

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = []
        if self._n_quantiles != 1000:
            arguments.append(f"n_quantiles={self._n_quantiles!r}")
        if self.subsample != 10_000:
            arguments.append(f"subsample={self.subsample!r}")
        if self.output_distribution != "uniform":
            arguments.append(
                f"output_distribution={self.output_distribution!r}"
            )
        if not arguments:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )
