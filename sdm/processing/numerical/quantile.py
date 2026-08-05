from typing import Literal

import torch
from torch import Tensor

from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor

BOUNDS_THRESH = 1e-7
_MAX_NUM_COLS = 32


def _torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    n = xp.numel()
    if n == 1:
        return fp[0].expand_as(x)

    idx = torch.searchsorted(xp, x, right=True).clamp(1, n - 1)

    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]

    denom = x1 - x0
    weight = torch.where(denom != 0, (x - x0) / denom, 0.0)
    result = torch.lerp(y0, y1, weight)

    result = torch.where(x <= xp[0], fp[0], result)
    return torch.where(x >= xp[-1], fp[-1], result)


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
            return references[0].expand_as(values)
        return references[:, :1].expand_as(values)

    idx = torch.searchsorted(boundaries, values, right=True).clamp(1, n - 1)

    if boundaries.dim() == 1:
        x0 = boundaries[idx - 1]
        x1 = boundaries[idx]
        lower_boundary = boundaries[0]
        upper_boundary = boundaries[-1]
    else:
        x0 = boundaries.gather(1, idx - 1)
        x1 = boundaries.gather(1, idx)
        lower_boundary = boundaries[:, :1]
        upper_boundary = boundaries[:, -1:]

    if references.dim() == 1:
        y0 = references[idx - 1]
        y1 = references[idx]
        lower = references[0]
        upper = references[-1]
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
    """Map feature columns through their empirical quantiles.

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

    supported_stypes = frozenset({Stype.numerical})

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
        if output_distribution not in {"uniform", "normal"}:
            raise ValueError(
                "output_distribution must be 'uniform' or 'normal'."
            )
        self._n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.n_quantiles = 0

        self.register_buffer("quantiles", torch.empty(0))
        self.register_buffer("references", torch.empty(0))

    def _subsample_indices(
        self,
        inp: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        return torch.randperm(
            inp.shape[0],
            generator=generator,
            device=inp.device,
        )[: self.subsample]

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        n_samples = numerical.shape[0]
        quantile_limit = n_samples
        if self.subsample is not None:
            # Keep quantiles well below the subsample size; very dense
            # percentile grids are slow to fit and add little resolution.
            quantile_limit = min(quantile_limit, int(self.subsample * 0.2))
        self.n_quantiles = max(1, min(self._n_quantiles, quantile_limit))

        self.references = torch.linspace(
            0,
            1,
            self.n_quantiles,
            device=numerical.device,
            dtype=numerical.dtype,
        )

        if self.subsample is not None and self.subsample < n_samples:
            indices = self._subsample_indices(numerical, generator)
            input_sample = numerical[indices]
        else:
            input_sample = numerical

        self.quantiles = torch.quantile(
            input_sample,
            self.references,
            dim=0,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` into the configured output distribution."""
        numerical = table.numerical
        transformed = torch.empty_like(numerical)
        for start in range(0, numerical.shape[1], _MAX_NUM_COLS):
            end = min(start + _MAX_NUM_COLS, numerical.shape[1])
            # Searchsorted works over the innermost dimension, so columns
            # become independent rows: input ``[N, F]`` -> ``[F, N]``.
            input_columns = numerical[:, start:end].T.contiguous()
            quantile_columns = self.quantiles[:, start:end].T.contiguous()
            lower_bound_x = quantile_columns[:, :1]
            upper_bound_x = quantile_columns[:, -1:]
            if self.output_distribution == "normal":
                bounds_thresh = input_columns.new_tensor(BOUNDS_THRESH)
                lower_bounds_idx = (
                    input_columns - bounds_thresh < lower_bound_x
                )
                upper_bounds_idx = (
                    input_columns + bounds_thresh > upper_bound_x
                )
            else:
                lower_bounds_idx = input_columns == lower_bound_x
                upper_bounds_idx = input_columns == upper_bound_x

            finite = input_columns.isfinite()
            forward = _batched_interp(
                input_columns,
                quantile_columns,
                self.references,
            )
            backward = _batched_interp(
                -input_columns,
                -quantile_columns.flip(1),
                -self.references.flip(0),
            )
            output = 0.5 * (forward - backward)

            output = torch.where(finite, output, input_columns)
            output = torch.where(upper_bounds_idx, 1.0, output)
            output = torch.where(lower_bounds_idx, 0.0, output)

            if self.output_distribution == "normal":
                eps = input_columns.new_tensor(
                    BOUNDS_THRESH - torch.finfo(torch.float64).eps
                )
                output = torch.special.ndtri(output)
                clip_min = torch.special.ndtri(eps)
                clip_max = torch.special.ndtri(1.0 - eps)
                output = output.clamp(clip_min, clip_max)

            transformed[:, start:end] = output.T.contiguous()
        return table.replace_blocks(numerical=transformed)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        inverse = torch.empty_like(numerical)
        for start in range(0, numerical.shape[1], _MAX_NUM_COLS):
            end = min(start + _MAX_NUM_COLS, numerical.shape[1])
            # Searchsorted works over the innermost dimension, so columns
            # become independent rows: input ``[N, F]`` -> ``[F, N]``.
            input_columns = numerical[:, start:end].T.contiguous()
            quantile_columns = self.quantiles[:, start:end].T.contiguous()
            lower_bound_y = quantile_columns[:, :1]
            upper_bound_y = quantile_columns[:, -1:]
            if self.output_distribution == "normal":
                input_columns = torch.special.ndtr(input_columns)

            if self.output_distribution == "normal":
                bounds_thresh = input_columns.new_tensor(BOUNDS_THRESH)
                lower_bounds_idx = input_columns - bounds_thresh < 0.0
                upper_bounds_idx = input_columns + bounds_thresh > 1.0
            else:
                lower_bounds_idx = input_columns == 0.0
                upper_bounds_idx = input_columns == 1.0

            finite = input_columns.isfinite()
            output = _batched_interp(
                input_columns,
                self.references,
                quantile_columns,
            )
            output = torch.where(finite, output, input_columns)
            output = torch.where(upper_bounds_idx, upper_bound_y, output)
            output = torch.where(lower_bounds_idx, lower_bound_y, output)
            inverse[:, start:end] = output.T.contiguous()
        return table.replace_blocks(numerical=inverse)
