from typing import Literal

import torch
from torch import Tensor

from sdm.processing.base import InvertibleMixin, Processor

BOUNDS_THRESH = 1e-7


def _torch_interp_1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
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


def _torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Interpolate ``x`` along sorted knot positions ``xp`` into ``fp``.

    Supports a single column (all arguments 1D) and batched columns where
    ``x`` has shape ``(n_samples, n_features)`` and ``xp`` / ``fp`` provide
    per-column knots.
    """
    if x.ndim == 1:
        return _torch_interp_1d(x, xp, fp)

    if x.shape[1] == 1 and (xp.ndim == 1 or xp.shape[1] == 1):
        xp_1d = xp.squeeze(-1) if xp.ndim > 1 else xp
        fp_1d = fp.squeeze(-1) if fp.ndim > 1 else fp
        return _torch_interp_1d(x.squeeze(1), xp_1d, fp_1d).unsqueeze(1)

    if xp.ndim == 1:
        xp = xp.unsqueeze(1).expand(-1, x.shape[1])
    if fp.ndim == 1:
        fp = fp.unsqueeze(1).expand(-1, x.shape[1])

    n = xp.shape[0]
    if n == 1:
        return fp[0].expand_as(x)

    x_t = x.transpose(0, 1).contiguous()
    xp_t = xp.transpose(0, 1).contiguous()
    fp_t = fp.transpose(0, 1)

    idx = torch.searchsorted(xp_t, x_t, right=True).clamp(1, n - 1)
    idx_m1 = idx - 1

    x0 = torch.gather(xp_t, 1, idx_m1)
    x1 = torch.gather(xp_t, 1, idx)
    y0 = torch.gather(fp_t, 1, idx_m1)
    y1 = torch.gather(fp_t, 1, idx)

    denom = x1 - x0
    weight = torch.where(denom != 0, (x_t - x0) / denom, torch.zeros_like(x_t))
    result = torch.lerp(y0, y1, weight)

    result = torch.where(x_t <= xp_t[:, :1], fp_t[:, :1], result)
    result = torch.where(x_t >= xp_t[:, -1:], fp_t[:, -1:], result)
    return result.transpose(0, 1)


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

    def _transform(self, input: Tensor, *, inverse: bool = False) -> Tensor:
        quantiles = self.quantiles
        output = input.clone()
        zero = output.new_zeros(())
        one = output.new_ones(())

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
                output = torch.special.ndtr(output)

        if self.output_distribution == "normal":
            bounds_thresh = output.new_tensor(BOUNDS_THRESH)
            lower_bounds_idx = output - bounds_thresh < lower_bound_x
            upper_bounds_idx = output + bounds_thresh > upper_bound_x
        else:
            lower_bounds_idx = output == lower_bound_x
            upper_bounds_idx = output == upper_bound_x

        finite = output.isfinite()
        values = torch.where(finite, output, torch.zeros_like(output))

        if not inverse:
            forward = _torch_interp(values, quantiles, self.references)
            backward = _torch_interp(
                -values,
                -quantiles.flip(0),
                -self.references.flip(0),
            )
            interpolated = 0.5 * (forward - backward)
            output = torch.where(finite, interpolated, output)
        else:
            interpolated = _torch_interp(values, self.references, quantiles)
            output = torch.where(finite, interpolated, output)

        output = torch.where(upper_bounds_idx, upper_bound_y, output)
        output = torch.where(lower_bounds_idx, lower_bound_y, output)

        if not inverse and self.output_distribution == "normal":
            eps = output.new_tensor(
                BOUNDS_THRESH - torch.finfo(torch.float64).eps
            )
            output = torch.special.ndtri(output)
            clip_min = torch.special.ndtri(eps)
            clip_max = torch.special.ndtri(one - eps)
            output = output.clamp(clip_min, clip_max)

        return output

    def forward(self, input: Tensor) -> Tensor:
        """Transform ``input`` into the configured output distribution."""
        return self._transform(input, inverse=False)

    def _inverse_transform(self, input: Tensor) -> Tensor:
        return self._transform(input, inverse=True)
