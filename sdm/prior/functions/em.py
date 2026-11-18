# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP, Correlated, LogUniformInt, Uniform
from sdm.prior.functions import RandomFunction, RandomLinear


class RandomEMAssignment(RandomFunction):
    """Compute random soft cluster assignments and project them.

    Args:
        p: Minkowski distance exponent.
        q: Exponent applied to the normalized distance.
        bandwidth: Log-scale spread of cluster widths; defaults to a
            correlated uniform draw between zero and one.
        m: Number of cluster centers. Defaults to
            ``LogUniformInt(2, max(16, 2 * out_channels))``.
        linear: Random linear function applied after softmax. Defaults to
            :class:`RandomLinear`.
    """

    def __init__(
        self,
        p: HP[float] | float,
        q: HP[float] | float,
        bandwidth: HP[float] | float | None = None,
        m: HP[int] | int | None = None,
        linear: RandomLinear | None = None,
    ) -> None:
        self.p = HP.from_value(p)
        self.q = HP.from_value(q)
        self.bandwidth = HP.from_value(
            bandwidth
            if bandwidth is not None
            else Correlated(Uniform(0.0, 1.0))
        )
        self.m = HP.from_value(m) if m is not None else None
        self.linear = linear or RandomLinear()

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        x = x.unsqueeze(0)
        batch_size, num_rows, in_channels = x.size()

        if self.m is not None:
            m = self.m.sample(generator=generator)
        else:
            high = max(16, 2 * out_channels)
            m = LogUniformInt(2, high).sample(generator=generator)

        # Centers: randomly sampled input rows plus standard normal noise.
        index = torch.randint(
            low=0,
            high=num_rows,
            size=(batch_size, m),
            device=x.device,
            generator=generator,
        )
        center = x[torch.arange(batch_size, device=x.device)[:, None], index]
        center = center + torch.randn(
            (batch_size, m, in_channels),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )

        bandwidth = self.bandwidth.sample_batch(
            size=(batch_size, 1, 1),
            generator=generator,
        ).to(x.dtype)
        sigma = torch.randn(
            (batch_size, 1, m),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        sigma = (bandwidth * sigma).exp()

        p = self.p.sample(generator=generator)
        q = self.q.sample_batch(
            size=(batch_size, 1, 1),
            generator=generator,
        ).to(x.dtype)

        dist = torch.cdist(x, center, p=p)
        log_term = -0.5 * (2 * torch.pi * sigma**2).log()
        norm_term = (dist / sigma) ** q
        logits = log_term - norm_term
        prob = logits.softmax(dim=-1)

        return self.linear.forward(
            x=prob.squeeze(0), out_channels=out_channels, generator=generator
        )
