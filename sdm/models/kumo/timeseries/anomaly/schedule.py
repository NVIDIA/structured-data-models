# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import torch
from torch import Tensor


class DiffusionSchedule(torch.nn.Module):
    """Discrete noise coefficients for Kumo-Anomaly diffusion.

    The one-dimensional buffers ``beta``, ``alpha_hat = 1 - beta``, and
    ``alpha = cumprod(alpha_hat)`` have length ``num_steps``. Only ``alpha``
    is clipped to ``[1e-6, 1 - 1e-6]``, matching Kumo-TS. Tables are computed
    in float64 on CPU before conversion to the requested device and dtype.
    They are derived from configuration and excluded from the state dict.

    Args:
        num_steps: Number of diffusion steps, at least one.
        beta_start: Initial noise variance, between zero and one.
        beta_end: Final noise variance, between ``beta_start`` and one.
        schedule: ``"quad"`` interpolates square roots of the noise variances
            before squaring; ``"linear"`` interpolates the variances directly.
        device: The device.
        dtype: Coefficient dtype. Defaults to float64, as in the upstream
            NumPy schedule. Use float32 or float64 to retain the alpha bounds.
    """

    beta: Tensor
    alpha_hat: Tensor
    alpha: Tensor

    def __init__(
        self,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        schedule: Literal["quad", "linear"] = "quad",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        device = torch.get_default_device() if device is None else device
        self.num_steps = num_steps
        if schedule == "quad":
            beta_start = beta_start**0.5
            beta_end = beta_end**0.5
        else:
            assert schedule == "linear"
        beta = torch.linspace(
            start=beta_start,
            end=beta_end,
            steps=num_steps,
            device="cpu",
            dtype=torch.float64,
        )
        if schedule == "quad":
            beta = beta.square()
        alpha_hat = 1 - beta
        alpha = alpha_hat.cumprod(dim=0).clamp(min=1e-6, max=1 - 1e-6)
        for name, value in (
            ("beta", beta),
            ("alpha_hat", alpha_hat),
            ("alpha", alpha),
        ):
            self.register_buffer(
                name=name,
                tensor=value.to(device=device, dtype=dtype),
                persistent=False,
            )

    def forward(self, step: Tensor) -> Tensor:
        """Look up clipped cumulative signal coefficients.

        Args:
            step: Integer diffusion indices in ``[0, num_steps)`` with any
                shape, on the module's device.

        Returns:
            Cumulative ``alpha`` values with the same shape as ``step`` and
            the coefficient dtype. Cast to the data dtype and broadcast over
            feature/time dimensions when constructing noisy observations.
        """
        return self.alpha[step]
