import math
from typing import Literal

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor

BOUNDS_THRESH = 1e-7
_MAX_NUM_COLS = 32


def _column_chunk(
    columns: Tensor, start: int, end: int
) -> tuple[Tensor, bool]:
    if columns.dim() == 2:
        return columns[start:end], False
    n_features = columns.size(-2)
    batch_index, first = divmod(start, n_features)
    last_batch, last = divmod(end - 1, n_features)
    if batch_index == last_batch:
        coordinates = []
        for size in reversed(columns.shape[:-2]):
            batch_index, coordinate = divmod(batch_index, size)
            coordinates.append(coordinate)
        return columns[(*reversed(coordinates), slice(first, last + 1))], False
    # Only chunks spanning batches need gathering; flattening all batch and
    # feature axes would copy the entire input before processing any chunk.
    indices = torch.unravel_index(
        torch.arange(start, end, device=columns.device), columns.shape[:-1]
    )
    return columns[indices], True


def _batched_interp(
    values: Tensor,
    boundaries: Tensor,
    references: Tensor,
    *,
    right: bool = True,
    out: Tensor | None = None,
) -> Tensor:
    # Batched 1D linear interpolation over independent feature grids:
    # values ``[F, N]`` with boundaries/references as ``[Q]`` or ``[F, Q]``.
    n = boundaries.shape[-1]
    if out is None:
        out = references.new_empty(values.shape)
    if n == 1:
        return out.copy_(references[..., :1])

    idx = torch.searchsorted(boundaries, values, right=right).clamp_(1, n - 1)
    boundaries = boundaries.expand(values.size(0), -1)
    references = references.expand(values.size(0), -1)
    denom = boundaries.gather(1, idx)
    idx.sub_(1)
    weight = boundaries.gather(1, idx)
    if right:
        denom.sub_(weight)
        if values.dtype == weight.dtype:
            torch.sub(values, weight, out=weight)
        else:
            weight = values - weight
    else:
        # Interpolate from the upper endpoint, matching the mirrored grid's
        # arithmetic without materializing negated values and flipped grids.
        denom, weight = weight, denom
        torch.sub(weight, denom, out=denom)
        if values.dtype == weight.dtype:
            weight.sub_(values)
        else:
            weight = weight - values
    weight.div_(denom).masked_fill_(denom == 0, 0.0)
    del denom

    if right:
        torch.gather(references, 1, idx, out=out)
        idx.add_(1)
        upper = references.gather(1, idx)
    else:
        upper = references.gather(1, idx)
        idx.add_(1)
        torch.gather(references, 1, idx, out=out)
    del idx
    out.lerp_(upper, weight)
    del upper, weight

    torch.where(values <= boundaries[:, :1], references[:, :1], out, out=out)
    return torch.where(
        values >= boundaries[:, -1:], references[:, -1:], out, out=out
    )


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

    handles_stypes = frozenset({Stype.numerical})
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

    def _subsample_indices(
        self,
        inp: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        return torch.randperm(
            inp.size(-2),
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
            input_sample = numerical.index_select(
                -2, self._subsample_indices(numerical, generator)
            )
        else:
            input_sample = numerical

        sample_size = input_sample.size(-2)
        # Contiguous feature grids keep quantile's CUDA sort workspace small.
        sample_columns = (
            input_sample.movedim(-1, -2).reshape(-1, sample_size).contiguous()
        )
        del input_sample
        quantiles = torch.quantile(sample_columns, references, dim=-1)
        # Store contiguous feature grids once for subsequent searches.
        self._quantiles = (
            quantiles.movedim(0, -1)
            .contiguous()
            .reshape(*numerical.shape[:-2], numerical.size(-1), n_quantiles)
            .movedim(-1, -2)
        )
        self._references = references

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        n_samples = numerical.size(-2)
        n_features = numerical.size(-1)
        input_columns = numerical.movedim(-1, -2)
        num_columns = math.prod(input_columns.shape[:-1])
        quantile_columns = (
            self._quantiles.movedim(-1, -2)
            .reshape(-1, self._references.numel())
            .contiguous()
        )
        transformed_columns = numerical.new_empty((num_columns, n_samples))

        for start in range(0, num_columns, _MAX_NUM_COLS):
            end = min(start + _MAX_NUM_COLS, num_columns)
            input_chunk, _ = _column_chunk(input_columns, start, end)
            input_chunk = input_chunk.contiguous()
            quantile_chunk = quantile_columns[start:end]
            output = _batched_interp(
                input_chunk,
                quantile_chunk,
                self._references,
                out=(
                    transformed_columns[start:end]
                    if numerical.dtype == self._references.dtype
                    else None
                ),
            )
            backward = _batched_interp(
                input_chunk,
                quantile_chunk,
                self._references,
                right=False,
            )
            output.add_(backward).mul_(0.5)
            del backward

            if self.output_distribution == "normal":
                output.masked_fill_(
                    input_chunk + BOUNDS_THRESH > quantile_chunk[:, -1:], 1.0
                )
                output.masked_fill_(
                    input_chunk - BOUNDS_THRESH < quantile_chunk[:, :1], 0.0
                )
            elif self.output_distribution == "uniform":
                output.masked_fill_(input_chunk == quantile_chunk[:, -1:], 1.0)
                output.masked_fill_(input_chunk == quantile_chunk[:, :1], 0.0)
            else:
                raise ValueError(
                    "output_distribution must be 'uniform' or 'normal'."
                )

            if self.output_distribution == "normal":
                eps = input_chunk.new_tensor(
                    BOUNDS_THRESH - torch.finfo(torch.float64).eps
                )
                torch.special.ndtri(output, out=output)
                clip_min = torch.special.ndtri(eps)
                clip_max = torch.special.ndtri(1.0 - eps)
                output.clamp_(clip_min, clip_max)

            if numerical.dtype != self._references.dtype:
                transformed_columns[start:end] = output
            del input_chunk, output

        transformed = transformed_columns.reshape(
            *numerical.shape[:-2],
            n_features,
            n_samples,
        ).movedim(-1, -2)
        return table.replace_blocks(numerical=transformed)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        n_samples = numerical.size(-2)
        n_features = numerical.size(-1)
        input_columns = numerical.movedim(-1, -2)
        num_columns = math.prod(input_columns.shape[:-1])
        quantile_columns = (
            self._quantiles.movedim(-1, -2)
            .reshape(-1, self._references.numel())
            .contiguous()
        )
        inverse_columns = numerical.new_empty((num_columns, n_samples))

        for start in range(0, num_columns, _MAX_NUM_COLS):
            end = min(start + _MAX_NUM_COLS, num_columns)
            input_chunk, owns_chunk = _column_chunk(input_columns, start, end)
            quantile_chunk = quantile_columns[start:end]
            if self.output_distribution == "normal":
                input_chunk = torch.special.ndtr(
                    input_chunk,
                    out=(
                        input_chunk
                        if owns_chunk
                        else numerical.new_empty(input_chunk.shape)
                    ),
                )
            else:
                input_chunk = input_chunk.contiguous()

            output = _batched_interp(
                input_chunk,
                self._references,
                quantile_chunk,
                out=(
                    inverse_columns[start:end]
                    if numerical.dtype == quantile_chunk.dtype
                    else None
                ),
            )
            if self.output_distribution == "normal":
                upper_bounds_idx = input_chunk + BOUNDS_THRESH > 1.0
                lower_bounds_idx = input_chunk - BOUNDS_THRESH < 0.0
            elif self.output_distribution == "uniform":
                upper_bounds_idx = input_chunk == 1.0
                lower_bounds_idx = input_chunk == 0.0
            else:
                raise ValueError(
                    "output_distribution must be 'uniform' or 'normal'."
                )
            torch.where(
                upper_bounds_idx, quantile_chunk[:, -1:], output, out=output
            )
            torch.where(
                lower_bounds_idx, quantile_chunk[:, :1], output, out=output
            )
            if numerical.dtype != quantile_chunk.dtype:
                inverse_columns[start:end] = output
            del input_chunk, output, upper_bounds_idx, lower_bounds_idx

        inverse = inverse_columns.reshape(
            *numerical.shape[:-2],
            n_features,
            n_samples,
        ).movedim(-1, -2)
        return table.replace_blocks(numerical=inverse)
