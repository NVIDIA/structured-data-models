# Portions derived from DPM-Solver:
# Copyright (c) 2022 Cheng Lu
# Licensed under the MIT License; see third_party/dpm-solver/LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from collections.abc import Callable

import torch
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.noise_schedule import NoiseScheduleVP


class DPMSolver(torch.nn.Module):
    r"""The second-order, single-step sampler from the `"DPM-Solver++: Fast
    Solver for Guided Sampling of Diffusion Probabilistic Models"
    <https://arxiv.org/abs/2211.01095>`__ paper.

    Time runs uniformly from ``T = 1`` to ``1 / total_N`` of
    :class:`NoiseScheduleVP`, and the solver integrates clean-data
    predictions without final denoising or clipping.

    The denoiser receives training step indices: each solver time ``t``
    maps to ``round(t * total_N - 1)``, clamped to ``[0, total_N - 1]``.

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
            # Place the intermediate time at the midpoint of the interval,
            # expressed as a fraction of its half-logSNR step:
            inner = torch.linspace(
                start=start,
                end=end,
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
        times = torch.stack(intervals)  # [num_intervals, 3]
        steps = (times * schedule.total_N - 1).round().long()
        steps = steps.clamp(0, schedule.total_N - 1)
        for name, value in (
            ("alpha", schedule.marginal_alpha(times).reshape(-1, 3)),
            ("sigma", schedule.marginal_std(times).reshape(-1, 3)),
            ("delta", torch.stack(deltas)),
            ("fraction", torch.stack(fractions)),
            ("steps", steps),
        ):
            self.register_buffer(name=name, tensor=value, persistent=False)

    def forward(
        self,
        x: Tensor,
        predict_noise: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Integrate initial noise using a noise-predicting denoiser.

        Args:
            x: Initial noise, for example ``[batch, features, time]``, on the
                module's device.
            predict_noise: Callable accepting the current sample and a
                one-element training step index tensor, returning predicted
                noise with the sample's shape.

        Returns:
            Reconstructed data with ``x``'s shape.
        """
        for i, order in enumerate(self.orders):
            alpha, sigma = self.alpha[i], self.sigma[i]
            delta, fraction = self.delta[i], self.fraction[i]
            eps = predict_noise(x, self.steps[i, :1])
            data = (x - sigma[0] * eps) / alpha[0]
            data_mid = None
            if order == 2:
                mid = _update(
                    x=x,
                    data=data,
                    alpha_end=alpha[1],
                    sigma_start=sigma[0],
                    sigma_end=sigma[1],
                    delta=fraction * delta,
                )
                eps_mid = predict_noise(mid, self.steps[i, 1:2])
                data_mid = (mid - sigma[1] * eps_mid) / alpha[1]
            x = _update(
                x=x,
                data=data,
                alpha_end=alpha[2],
                sigma_start=sigma[0],
                sigma_end=sigma[2],
                delta=delta,
                data_mid=data_mid,
                fraction=fraction,
            )
        return x


def _update(
    x: Tensor,
    data: Tensor,
    alpha_end: Tensor,
    sigma_start: Tensor,
    sigma_end: Tensor,
    delta: Tensor,
    data_mid: Tensor | None = None,
    fraction: Tensor | float = 0.5,
) -> Tensor:
    factor = alpha_end * (-delta).expm1()
    out = (sigma_end / sigma_start) * x - factor * data
    if data_mid is not None:
        out = out - (0.5 / fraction) * factor * (data_mid - data)
    return out
