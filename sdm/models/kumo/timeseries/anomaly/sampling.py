# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule


class DDPMStep(torch.nn.Module):
    """Apply one Kumo-Anomaly DDPM reverse-diffusion update.

    This component consumes predicted noise, not a denoiser. Callers control
    the reverse-step loop, conditioning, and random draws. No normalization,
    observation restoration, or anomaly scoring is applied.

    Args:
        schedule: Diffusion coefficients used to derive the mean and noise
            scales. Coefficients retain the schedule's device and dtype and
            are excluded from the state dict. Construct a new step module
            when changing the schedule.
    """

    mean_scale: Tensor
    noise_scale: Tensor
    std: Tensor

    def __init__(self, schedule: DiffusionSchedule) -> None:
        super().__init__()
        alpha = schedule.alpha
        previous_alpha = torch.cat([alpha.new_ones(1), alpha[:-1]])
        for name, value in (
            ("mean_scale", schedule.alpha_hat.rsqrt()),
            ("noise_scale", (1 - schedule.alpha_hat) / (1 - alpha).sqrt()),
            (
                "std",
                ((1 - previous_alpha) / (1 - alpha) * schedule.beta).sqrt(),
            ),
        ):
            self.register_buffer(name=name, tensor=value, persistent=False)

    def forward(
        self,
        x: Tensor,
        predicted_noise: Tensor,
        step: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Take a reverse step using explicitly supplied Gaussian noise.

        Args:
            x: Current normalized sample with shape ``[B, K, L]``.
            predicted_noise: Denoiser prediction with the shape and dtype of
                ``x``.
            step: Integer schedule indices with shape ``[B]`` or ``[1]``;
                a scalar also broadcasts across the batch. Zero is the final
                reverse step. All inputs must be on the coefficient device.
            noise: Standard-normal noise with the same shape and dtype as
                ``x``. The noise contribution is zero at step zero. Random
                draws and their generator are controlled by the caller.

        Returns:
            Updated samples with the shape and dtype of ``x``. Coefficients
            are cast to the sample dtype before the update.
        """
        mean_scale = self.mean_scale[step].to(x.dtype)[..., None, None]
        noise_scale = self.noise_scale[step].to(x.dtype)[..., None, None]
        std = self.std[step].to(x.dtype)[..., None, None]
        return mean_scale * (x - noise_scale * predicted_noise) + std * noise
