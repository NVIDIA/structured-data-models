# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from sdm.prior import HP
from sdm.prior.matrix_samplers import MatrixSampler
from sdm.prior.matrix_samplers._normalize import normalize
from sdm.prior.util import random_weight


class KernelMatrixSampler(MatrixSampler):
    """Sample Laplace-kernel matrices between Gaussian-mixture anchor points.

    Uses entries ``exp(-gamma * distance)`` with random signs.

    Args:
        num_centers: Number of Gaussian-mixture centers.
        gamma: Inverse distance scale.
        channels: Dimensionality of the anchor points.
        normalize: Add small Gaussian noise and L2-normalize rows.
    """

    def __init__(
        self,
        num_centers: HP[int] | int,
        gamma: HP[float] | float,
        channels: int = 3,
        normalize: bool = True,
    ) -> None:
        self.num_centers = HP.from_value(num_centers)
        self.gamma = HP.from_value(gamma)
        self.channels = channels
        self.normalize = normalize

    def sample(  # noqa: D102
        self,
        size: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        *batch, m, k = size
        dataset_batch = batch[0] if batch else 1
        matrix_batch = math.prod(batch[1:])
        channels = self.channels
        num_points = m + k
        num_centers = self.num_centers.sample(generator=generator)

        centers = torch.randn(
            (dataset_batch, num_centers, channels),
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
        cov = torch.randn(
            (dataset_batch, num_centers, channels, channels),
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
        importance = random_weight(
            size=(dataset_batch * num_centers, channels),
            dtype=dtype,
            generator=generator,
        ).view(dataset_batch, num_centers, channels)
        cov = cov * importance.unsqueeze(-2)
        log_regime = torch.randn(
            (dataset_batch, 2), dtype=dtype, device="cpu", generator=generator
        )
        component_scale = torch.randn(
            (dataset_batch, num_centers, 1, 1),
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
        cov = cov * torch.exp(
            log_regime[:, :1, None, None]
            + log_regime[:, 1:, None, None] * component_scale
        )

        component_weight = random_weight(
            size=(dataset_batch, num_centers),
            dtype=dtype,
            generator=generator,
        )
        component = torch.multinomial(
            component_weight,
            num_samples=matrix_batch * num_points,
            replacement=True,
            generator=generator,
        )
        batch_index = torch.arange(dataset_batch, device="cpu").unsqueeze(-1)
        noise = torch.randn(
            (dataset_batch, matrix_batch * num_points, channels),
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
        selected_cov = cov[batch_index, component]
        pts = centers[batch_index, component] + torch.einsum(
            "bnij,bnj->bni", selected_cov, noise
        )
        pts = pts.view(*batch, num_points, channels)
        left, right = pts[..., :m, :], pts[..., m:, :]
        dist = torch.cdist(left, right)

        gamma = (
            self.gamma.sample_batch(
                size=(dataset_batch, matrix_batch),
                generator=generator,
            )
            .to(dtype)
            .view(*batch, 1, 1)
        )

        W = (-gamma * dist).exp()
        sign = torch.randint(
            high=2,
            size=size,
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
        W *= 2 * sign - 1
        return normalize(W, generator) if self.normalize else W
