# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


class CausalPatchNorm(torch.nn.Module):
    """Normalize patches using running observed means and sample deviations.

    Each patch uses observations through its last step. Statistics accumulate
    in float64 and return in the input dtype. Unobserved values become zero.

    Args:
        patch_size: Number of time steps per patch.
        minimum_scale: Floor for the running standard deviation.
        use_asinh: Apply an inverse hyperbolic sine after normalization.
    """

    def __init__(
        self,
        patch_size: int,
        minimum_scale: float = 1e-6,
        use_asinh: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.minimum_scale = minimum_scale
        self.use_asinh = use_asinh

    @torch.no_grad()
    def stats(self, x: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
        """Return running patch means and scales, broadcast to time steps.

        Args:
            x: Values ``[..., T]``, with ``T`` divisible by the patch size.
            observed: Boolean observation mask with the same shape as ``x``.
        """
        values = x.double().masked_fill(~observed, 0)
        weights = observed.double()
        count = weights.cumsum(-1).clamp_min(1)
        mean = values.cumsum(-1) / count
        previous = torch.nn.functional.pad(mean[..., :-1], (1, 0))
        # Welford increments preserve accuracy for large offsets.
        increments = (values - previous) * (values - mean) * weights
        variance = increments.cumsum(-1) / (count - 1).clamp_min(1)
        scale = variance.sqrt().clamp_min(self.minimum_scale)
        mean = mean.to(x.dtype).unflatten(-1, (-1, self.patch_size))[..., -1]
        scale = scale.to(x.dtype).unflatten(-1, (-1, self.patch_size))[..., -1]
        return (
            mean.repeat_interleave(self.patch_size, dim=-1),
            scale.repeat_interleave(self.patch_size, dim=-1),
        )

    def forward(
        self, x: Tensor, observed: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return normalized values, patch means, and patch scales.

        Args:
            x: Values ``[..., T]``, with ``T`` divisible by the patch size.
            observed: Boolean observation mask with the same shape as ``x``.
        """
        mean, scale = self.stats(x, observed)
        dtype = torch.promote_types(x.dtype, torch.float32)
        out = (x.to(dtype) - mean.to(dtype)) / scale.to(dtype).clamp_min(
            torch.finfo(dtype).eps
        )
        out = out.masked_fill(~observed, 0)
        if self.use_asinh:
            out = out.asinh()
        return out.to(x.dtype), mean, scale
