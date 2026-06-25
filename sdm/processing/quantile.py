from typing import Literal

import torch
from torch import Tensor

from sdm.processing.base import InvertibleMixin, Processor

BOUNDS_THRESH = 1e-7


def _torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    n = xp.numel()
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

    Args:
        n_quantiles: The number of quantiles to compute.
        subsample: The number of samples to use for quantile computation.
        output_distribution: The distribution to map the data to.
        random_state: Seed for deterministic subsampling. If ``None``, use the
            global PyTorch generator.
    """

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

        if self.output_distribution == "normal":
            self._distribution = torch.distributions.Normal(
                loc=0.0,
                scale=1.0,
                validate_args=False,
            )

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

    def _fit(self, input: Tensor) -> None:
        n_samples = input.shape[0]
        quantile_limit = n_samples
        if self.subsample is not None:
            quantile_limit = min(quantile_limit, int(self.subsample * 0.2))
        self.n_quantiles = max(1, min(self._n_quantiles, quantile_limit))

        self.references = torch.linspace(
            0,
            1,
            self.n_quantiles,
            device=input.device,
            dtype=input.dtype,
        )

        if self.subsample is not None and self.subsample < n_samples:
            input_sample = input[self._subsample_indices(input)]
        else:
            input_sample = input

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
                input_col = self._distribution.cdf(input_col)

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
            input_col = self._distribution.icdf(input_col)
            eps = input_col.new_tensor(
                BOUNDS_THRESH - torch.finfo(torch.float64).eps
            )
            clip_min = self._distribution.icdf(eps)
            clip_max = self._distribution.icdf(one - eps)
            input_col = input_col.clamp(clip_min, clip_max)

        return input_col

    def forward(self, input: Tensor) -> Tensor:
        """Transform ``input`` into the configured output distribution."""
        transformed = torch.empty_like(input)
        for i in range(input.shape[1]):
            transformed[:, i] = self._transform_col(
                input[:, i],
                self.quantiles[:, i],
                inverse=False,
            )
        return transformed

    def _inverse_transform(self, input: Tensor) -> Tensor:
        inverse = input.clone()
        for i in range(input.shape[1]):
            inverse[:, i] = self._transform_col(
                input[:, i],
                self.quantiles[:, i],
                inverse=True,
            )
        return inverse
