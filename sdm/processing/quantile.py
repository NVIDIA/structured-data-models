from typing import Literal

import torch
from torch import Tensor

from sdm.processing._utils import _as_float
from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor

BOUNDS_THRESH = 1e-7


def _torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    n = xp.numel()
    if n == 1:
        return fp[0].expand_as(x)

    idx = torch.searchsorted(xp, x, right=True).clamp(1, n - 1)

    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]

    denom = x1 - x0
    weight = torch.where(denom != 0, (x - x0) / denom, torch.zeros_like(x))
    result = torch.lerp(y0, y1, weight)

    result = torch.where(x <= xp[0], fp[0], result)
    return torch.where(x >= xp[-1], fp[-1], result)


class Quantile(Processor, InvertibleMixin):
    """Map feature columns through their empirical quantiles.

    Quantile grids are capped by the number of fitted rows and, when
    ``subsample`` is set, by ``20%`` of the subsample size to keep dense grids
    tractable.

    Args:
        n_quantiles: Maximum number of quantiles to compute.
        subsample: Maximum number of rows to use for quantile computation.
        output_distribution: Distribution to map the empirical quantiles to.
        random_state: Seed for deterministic subsampling. If ``None``, use the
            global PyTorch generator.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        *,
        n_quantiles: int = 1000,
        subsample: int | None = 10_000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
        random_state: int | None = 0,
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
        self.random_state = random_state
        self.n_quantiles = 0

        self.register_buffer("quantiles", torch.empty(0))
        self.register_buffer("references", torch.empty(0))

    def _subsample_indices(self, input: Tensor) -> Tensor:
        generator = None
        if self.random_state is not None:
            generator = torch.Generator(device=input.device)
            generator.manual_seed(self.random_state)
        return torch.randperm(
            input.shape[0],
            device=input.device,
            generator=generator,
        )[: self.subsample]

    def _fit(self, input: TableTensor) -> None:
        numerical = _as_float(input.numerical)
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
            input_sample = numerical[self._subsample_indices(numerical)]
        else:
            input_sample = numerical

        self.quantiles = torch.nanquantile(
            input_sample,
            self.references,
            dim=0,
        )

    def _transform_col(
        self,
        input: Tensor,
        quantiles: Tensor,
        *,
        inverse: bool = False,
    ) -> Tensor:
        input_col = input.clone()
        zero = input_col.new_zeros(())
        one = input_col.new_ones(())

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = zero
            upper_bound_y = one
        else:
            lower_bound_x = zero
            upper_bound_x = one
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            if self.output_distribution == "normal":
                input_col = torch.special.ndtr(input_col)

        if self.output_distribution == "normal":
            bounds_thresh = input_col.new_tensor(BOUNDS_THRESH)
            lower_bounds_idx = input_col - bounds_thresh < lower_bound_x
            upper_bounds_idx = input_col + bounds_thresh > upper_bound_x
        else:
            lower_bounds_idx = input_col == lower_bound_x
            upper_bounds_idx = input_col == upper_bound_x

        isfinite_mask = input_col.isfinite()
        input_col_finite = input_col[isfinite_mask]
        if not inverse:
            forward = _torch_interp(
                input_col_finite,
                quantiles,
                self.references,
            )
            backward = _torch_interp(
                -input_col_finite,
                -quantiles.flip(0),
                -self.references.flip(0),
            )
            input_col[isfinite_mask] = 0.5 * (forward - backward)
        else:
            input_col[isfinite_mask] = _torch_interp(
                input_col_finite,
                self.references,
                quantiles,
            )

        input_col[upper_bounds_idx] = upper_bound_y
        input_col[lower_bounds_idx] = lower_bound_y
        if not inverse and self.output_distribution == "normal":
            eps = input_col.new_tensor(
                BOUNDS_THRESH - torch.finfo(torch.float64).eps
            )
            input_col = torch.special.ndtri(input_col)
            clip_min = torch.special.ndtri(eps)
            clip_max = torch.special.ndtri(one - eps)
            input_col = input_col.clamp(clip_min, clip_max)

        return input_col

    def _transform(self, input: TableTensor) -> TableTensor:
        """Transform ``input`` into the configured output distribution."""
        numerical = _as_float(input.numerical)
        transformed = torch.empty_like(numerical)
        for i in range(numerical.shape[1]):
            transformed[:, i] = self._transform_col(
                numerical[:, i],
                self.quantiles[:, i],
                inverse=False,
            )
        return input.replace_blocks(numerical=transformed)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        numerical = _as_float(input.numerical)
        inverse = numerical.clone()
        for i in range(numerical.shape[1]):
            inverse[:, i] = self._transform_col(
                numerical[:, i],
                self.quantiles[:, i],
                inverse=True,
            )
        return input.replace_blocks(numerical=inverse)
