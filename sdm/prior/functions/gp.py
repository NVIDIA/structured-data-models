# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP, Mixture
from sdm.prior.functions import RandomFunction
from sdm.prior.util import random_weight


class RandomGP(RandomFunction):
    """Approximate a Gaussian process with random Fourier features.

    Randomly choose radial frequencies with a scaled input projection or
    independent-coordinate frequencies without the projection.

    Args:
        a: Shape parameter controlling the frequency distribution's tail.
        alpha: Scale of the input projection for radial frequencies.
        num_features: Number of random Fourier features.
    """

    def __init__(
        self,
        a: HP[float] | float,
        alpha: HP[float] | float,
        num_features: int = 256,
    ) -> None:
        self.a = HP.from_value(a)
        self.alpha = HP.from_value(alpha)
        self.num_features = num_features
        self.product_kernel = Mixture(
            [False, True],
            correlated=True,
        )

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        x = x.unsqueeze(0)
        batch_size, _, in_channels = x.size()
        eps = torch.finfo(x.dtype).eps

        a = self.a.sample_batch(
            size=(batch_size, 1),
            generator=generator,
        ).to(x.dtype)

        if self.product_kernel.pick(generator=generator):
            # Sample each frequency coordinate from a one-sided power law.
            u = torch.rand(
                size=(batch_size, self.num_features, in_channels),
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
            r = (1.0 - u).clamp(min=eps)
            r = r.pow(1.0 / (1.0 - a.view(-1, 1, 1))) - 1.0
            W = r
        else:
            # Sample a power-law magnitude and a uniform unit-sphere direction.
            u = torch.rand(
                size=(batch_size, self.num_features, 1),
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
            r = (1.0 - u).clamp(min=eps)
            r = r.pow(1.0 / (1.0 - a.view(-1, 1, 1))) - 1.0
            z = torch.randn(
                size=(batch_size, self.num_features, in_channels),
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
            W = r * z / z.norm(dim=-1, keepdim=True).clamp(min=eps)

            # M_ij = alpha * w_i * A_ij
            alpha = self.alpha.sample_batch(
                size=(batch_size, 1, 1),
                generator=generator,
            ).to(x.dtype)
            w = random_weight(
                size=(batch_size, in_channels),
                dtype=x.dtype,
                generator=generator,
            ).view(batch_size, in_channels, 1)
            A = torch.randn(
                (batch_size, in_channels, in_channels),
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
            A = A / A.norm(dim=-1, keepdim=True).clamp(min=eps)
            M = alpha * w * A  # [batch_size, in_channels, in_channels]
            x = x @ M.transpose(-1, -2)

        b = (2 * torch.pi) * torch.rand(
            size=(batch_size, 1, self.num_features),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )

        features = (x @ W.transpose(-1, -2) + b).cos()

        Z = torch.randn(
            size=(batch_size, out_channels, self.num_features),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )

        out = features @ Z.transpose(1, 2)
        return (self.num_features**-0.5) * out.squeeze(0)
