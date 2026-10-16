# Portions derived from DPM-Solver:
# Copyright (c) 2022 Cheng Lu
# Licensed under the MIT License; see third_party/dpm-solver/LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


class NoiseScheduleVP(torch.nn.Module):
    """Continuous-time VP coefficients for Kumo-Anomaly's DPM-Solver++.

    Interpolate ``log(alpha_t) = 0.5 * cumsum(log(1 - beta))`` between
    discrete steps. As in Kumo-TS, discard the tail where half-logSNR is
    below ``clipped_lambda`` (default -5.1) and assign the retained steps
    times ``(i + 1) / total_N``.
    ``total_N`` is the retained step count and ``T = 1`` is the final time.
    Unlike :class:`DiffusionSchedule`, cumulative alphas are not clamped.

    Methods accept scalar or batched tensors on the module's device and
    return flattened, one-dimensional tensors. Outside the retained time
    interval ``[1 / total_N, T]``, interpolation uses the boundary segment
    for linear extrapolation. The derived tables are nonpersistent buffers.

    Args:
        betas: One-dimensional diffusion variances strictly between zero
            and one, with at least two steps remaining after truncation.
            Coefficients retain the input device and dtype (float32 or
            float64). Pass ``DiffusionSchedule.beta.float()`` to match
            Kumo-TS inference.
        clipped_lambda: Minimum retained half-logSNR. Defaults to -5.1,
            matching Kumo-TS's numerical-stability cutoff.
    """

    _log_alpha: Tensor
    _time: Tensor

    def __init__(self, betas: Tensor, *, clipped_lambda: float = -5.1) -> None:
        super().__init__()
        self.schedule = "discrete"
        self.T = 1.0
        log_alpha = 0.5 * (1 - betas.detach()).log().cumsum(dim=0)
        log_sigma = 0.5 * (1 - (2 * log_alpha).exp()).log()
        # A single host synchronization at construction determines table size.
        num_clipped = int(
            torch.searchsorted((log_alpha - log_sigma).flip(0), clipped_lambda)
        )
        self.total_N = log_alpha.numel() - num_clipped
        if self.total_N < 2:
            raise ValueError(
                "At least two diffusion steps must remain after truncation"
            )
        self.register_buffer(
            name="_log_alpha",
            tensor=log_alpha[: self.total_N],
            persistent=False,
        )
        # Match upstream's float32 CPU grid before dtype/device conversion.
        time = torch.linspace(
            start=0,
            end=self.T,
            steps=self.total_N + 1,
            device="cpu",
            dtype=torch.float32,
        )[1:].to(betas)
        self.register_buffer("_time", time, persistent=False)

    def marginal_log_mean_coeff(self, t: Tensor) -> Tensor:
        """Return interpolated log signal coefficients at times ``t``."""
        return _interpolate(t.reshape(-1), self._time, self._log_alpha)

    def marginal_alpha(self, t: Tensor) -> Tensor:
        """Return signal coefficients ``alpha_t`` at continuous times ``t``."""
        return self.marginal_log_mean_coeff(t).exp()

    def marginal_std(self, t: Tensor) -> Tensor:
        """Return noise standard deviations ``sqrt(1 - alpha_t**2)``."""
        return (1 - (2 * self.marginal_log_mean_coeff(t)).exp()).sqrt()

    def marginal_lambda(self, t: Tensor) -> Tensor:
        """Return half-logSNR ``log(alpha_t) - log(sigma_t)``."""
        log_alpha = self.marginal_log_mean_coeff(t)
        return log_alpha - 0.5 * (1 - (2 * log_alpha).exp()).log()

    def inverse_lambda(self, lamb: Tensor) -> Tensor:
        """Return continuous times for the supplied half-logSNR values."""
        log_alpha = -0.5 * torch.logaddexp(torch.zeros_like(lamb), -2 * lamb)
        return _interpolate(
            x=log_alpha.reshape(-1),
            xp=self._log_alpha.flip(0),
            yp=self._time.flip(0),
        )


def _interpolate(x: Tensor, xp: Tensor, yp: Tensor) -> Tensor:
    """Interpolate sorted 1D knots, extrapolating at either boundary."""
    right = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.numel() - 1)
    left = right - 1
    return yp[left] + (x - xp[left]) * (yp[right] - yp[left]) / (
        xp[right] - xp[left]
    )
