# Portions derived from DPM-Solver:
# Copyright (c) 2022 Cheng Lu
# Licensed under the MIT License; see third_party/dpm-solver/LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.noise_schedule import NoiseScheduleVP


def dpm_solver_update(
    x: Tensor,
    data_start: Tensor,
    *,
    alpha_end: Tensor,
    sigma_start: Tensor,
    sigma_end: Tensor,
    lambda_delta: Tensor,
    data_mid: Tensor | None = None,
    r1: Tensor | float = 0.5,
) -> Tensor:
    """Apply a first- or second-order single-step DPM-Solver++ update.

    This implements Kumo-TS's ``algorithm_type="dpmsolver++"`` and
    ``solver_type="dpmsolver"`` formulas. Predictions are clean data, not
    noise: convert predicted noise ``eps`` using ``(x - sigma * eps) / alpha``
    at the corresponding time before calling this function.

    Without ``data_mid``, take a first-order step. For second order, first
    use this function to advance from the start to the intermediate time,
    then evaluate the model there and pass its clean-data prediction as
    ``data_mid`` together with the original ``x`` and full-interval
    coefficients. Model calls, time selection, and random draws are owned
    by the caller; this function is deterministic and differentiable.

    Args:
        x: Noisy sample at the start time, for example shape ``[B, K, L]``.
        data_start: Clean-data prediction at the start, with ``x``'s shape.
        alpha_end: Signal amplitude at the end time, not cumulative alpha
            squared as stored by ``DiffusionSchedule.alpha``.
        sigma_start: Positive noise standard deviation at the start time.
        sigma_end: Noise standard deviation at the end time.
        lambda_delta: End minus start half-logSNR, where
            ``lambda = log(alpha) - log(sigma)``.
        data_mid: Clean-data prediction at the intermediate sample/time,
            with ``x``'s shape. Omit for first order.
        r1: Intermediate fraction of the half-logSNR interval, strictly
            between zero and one. Ignored for first order. Uniform time
            spacing generally does not imply ``r1 = 0.5``; the caller must
            compute ``(lambda_mid - lambda_start) / lambda_delta``.

    Returns:
        Updated sample with ``x``'s shape. All tensors must be on the same
        device with compatible dtypes; standard PyTorch promotion applies.
        Coefficients and tensor ``r1`` must broadcast to ``x``: use scalars
        for shared times or ``[B, 1, 1]`` for per-window coefficients.
    """
    factor = alpha_end * (-lambda_delta).expm1()
    out = (sigma_end / sigma_start) * x - factor * data_start
    if data_mid is not None:
        out = out - (0.5 / r1) * factor * (data_mid - data_start)
    return out


class DPMSolver(torch.nn.Module):
    """Second-order, single-step DPM-Solver++ on a uniform time grid.

    The denoiser receives integer training indices, not the 1000-scaled
    coordinates used by the original DPM-Solver model wrapper. Continuous
    times map to the nearest trained noise level ``round(t * total_N - 1)``.
    This corrects Kumo-TS's double timestep conversion. No final denoising
    or clipping of clean-data predictions is applied.

    Args:
        betas: Discrete training variances on CPU, in training order.
        num_steps: Number of denoiser evaluations, at least one. Odd counts
            end with a first-order update.
    """

    alpha: Tensor
    sigma: Tensor
    delta: Tensor
    fraction: Tensor
    steps: Tensor

    def __init__(self, betas: Tensor, num_steps: int = 20) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("'num_steps' must be at least one")
        schedule = NoiseScheduleVP(betas)
        self.orders = (2,) * (num_steps // 2) + (1,) * (num_steps % 2)
        time = torch.linspace(
            start=schedule.T,
            end=1 / schedule.total_N,
            steps=num_steps + 1,
            device="cpu",
            dtype=torch.float32,
        )
        intervals = []
        deltas = []
        fractions = []
        offset = 0
        for order in self.orders:
            start, end = time[offset], time[offset + order]
            # Match the reference solver's inner float32 grid, including
            # its round trip through half-logSNR for the intermediate time.
            inner = torch.linspace(
                start=float(start),
                end=float(end),
                steps=3,
                device="cpu",
                dtype=torch.float32,
            )
            lamb = schedule.marginal_lambda(inner)
            delta = lamb[2] - lamb[0]
            fraction = (lamb[1] - lamb[0]) / delta
            mid = schedule.inverse_lambda(lamb[0] + fraction * delta)[0]
            intervals.append(torch.stack([start, mid, end]))
            deltas.append(delta)
            fractions.append(fraction)
            offset += order
        times = torch.stack(intervals)
        for name, value in (
            ("alpha", schedule.marginal_alpha(times).reshape(-1, 3)),
            ("sigma", schedule.marginal_std(times).reshape(-1, 3)),
            ("delta", torch.stack(deltas)),
            ("fraction", torch.stack(fractions)),
            (
                "steps",
                (times * schedule.total_N - 1)
                .round()
                .long()
                .clamp(0, schedule.total_N - 1),
            ),
        ):
            self.register_buffer(name=name, tensor=value, persistent=False)

    def forward(
        self,
        x: Tensor,
        predict_noise: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Integrate initial noise using a noise-predicting denoiser.

        Args:
            x: Initial noise, for example ``[batch, features, time]``.
            predict_noise: Callable accepting ``x`` and a one-element
                integer timestep tensor, returning noise with ``x``'s shape.
                It owns conditioning and must not rescale the timestep.

        Returns:
            Reconstructed data with ``x``'s shape. Move this module to
            ``x.device`` before calling; integration itself draws no noise.
        """
        for i, order in enumerate(self.orders):
            alpha, sigma = self.alpha[i], self.sigma[i]
            delta, fraction = self.delta[i], self.fraction[i]
            eps = predict_noise(x, self.steps[i, :1])
            data = (x - sigma[0] * eps) / alpha[0]
            data_mid = None
            if order == 2:
                mid = dpm_solver_update(
                    x=x,
                    data_start=data,
                    alpha_end=alpha[1],
                    sigma_start=sigma[0],
                    sigma_end=sigma[1],
                    lambda_delta=fraction * delta,
                )
                eps_mid = predict_noise(mid, self.steps[i, 1:2])
                data_mid = (mid - sigma[1] * eps_mid) / alpha[1]
            x = dpm_solver_update(
                x=x,
                data_start=data,
                alpha_end=alpha[2],
                sigma_start=sigma[0],
                sigma_end=sigma[2],
                lambda_delta=delta,
                data_mid=data_mid,
                r1=fraction,
            )
        return x
