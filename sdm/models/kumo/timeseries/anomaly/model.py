# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

import copy
from functools import partial
from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import Recipe, Stype, TableTensor
from sdm._inference import inference_mode
from sdm.models import Model
from sdm.models.kumo.timeseries.anomaly.denoiser import DiffusionDenoiser
from sdm.models.kumo.timeseries.anomaly.embedding import SideEmbedding
from sdm.models.kumo.timeseries.anomaly.recipe import default_recipe
from sdm.models.kumo.timeseries.anomaly.sampling import DPMSolver
from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule


class KumoAnomaly(Model):
    r"""The time series anomaly detection foundation model from the
    `"NV-Tesseract-AD: Curriculum-Enhanced Conditional Diffusion for
    Multivariate Time-Series Anomaly Detection"
    <https://arxiv.org/abs/2610.12754>`__ paper.

    :class:`KumoAnomaly` is a conditional diffusion model that scores each
    time step of a multivariate series by its reconstruction error.
    The series is split into windows of 100 time steps, each made of four
    alternating blocks.
    Two complementary masks condition on one half of the blocks and
    reconstruct the other with DPM-Solver++, and the per-step median over
    ``num_samples`` reconstructions is compared to the input.

    Args:
        device: The device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )

    def __init__(self, device: torch.device | str | None = None) -> None:
        super().__init__()

        self.model = _KumoAnomaly(device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def forward(
        self,
        x: Tensor | TableTensor,  # [R, D]
        reference: Tensor | TableTensor | None = None,  # [R_reference, D]
        *,
        recipe: Recipe | None = None,
        num_samples: int = 15,
        num_steps: int = 20,
        generator: torch.Generator | None = None,
    ) -> TableTensor:  # [R, 1]
        r"""Score each time step of a series by its reconstruction error.

        Args:
            x: The series to score, with rows as ordered time steps.
            reference: The rows used to fit preprocessing.
                If set to :obj:`None`, :obj:`x` is used, *i.e.* the series is
                scored transductively.
            recipe: The custom recipe for pre-processing.
            num_samples: The number of reconstructions per mask.
            num_steps: The number of denoiser evaluations per reconstruction.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            A table with one numerical ``"score"`` column holding the mean
            absolute reconstruction error per row in preprocessed feature
            space, where higher scores indicate more anomalous rows.
        """
        device = next(self.parameters()).device
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if reference is None:
            reference = x
        elif not isinstance(reference, TableTensor):
            reference = TableTensor.from_tensor(reference)
        x = cast(TableTensor, x.to(device))
        reference = cast(TableTensor, reference.to(device))

        recipe = self.default_recipe() if recipe is None else recipe
        recipe = copy.deepcopy(recipe)
        with (
            torch.amp.autocast(device.type, enabled=False),
            inference_mode(),
        ):
            x = recipe.features.fit(reference).transform(x)

        if x.size(-1) > self.model.num_features:
            raise ValueError(
                f"{self.__class__.__name__!r} supports at most "
                f"{self.model.num_features} features (got {x.size(-1)}). "
                f"Wider inputs are not yet supported."
            )

        with inference_mode():
            out = self.model(
                x.numerical,
                num_samples=num_samples,
                num_steps=num_steps,
                generator=generator,
            )

        return TableTensor(
            columns={Stype.numerical: ["score"]},
            numerical=out.unsqueeze(-1),
        )


class _KumoAnomaly(torch.nn.Module):
    def __init__(
        self,
        num_features: int = 40,
        time_channels: int = 256,
        feature_channels: int = 64,
        channels: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        embedding_channels: int = 256,
        num_steps: int = 500,
        window_size: int = 100,
        num_blocks: int = 4,
        batch_size: int = 32,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_features = num_features
        self.num_steps = num_steps
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.batch_size = batch_size

        self.side_embedding = SideEmbedding(
            num_features=num_features,
            time_channels=time_channels,
            feature_channels=feature_channels,
            **factory_kwargs,
        )
        self.denoiser = DiffusionDenoiser(
            channels=channels,
            side_channels=time_channels + feature_channels,
            num_steps=num_steps,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_channels=embedding_channels,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        num_samples: int,
        num_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:  # [..., R]

        *size, length, features = x.size()
        x = x.reshape(-1, length, features)
        x = F.pad(x, (0, self.num_features - features))

        # Pad the last window by repeating the last row:
        padding = -length % self.window_size
        x = torch.cat([x, x[:, -1:].expand(-1, padding, -1)], dim=1)
        windows = x.reshape(-1, self.window_size, self.num_features)

        schedule = DiffusionSchedule(
            num_steps=self.num_steps,
            beta_start=0.0001,
            beta_end=0.01,
            schedule="linear",
            device="cpu",
        )
        sampler = DPMSolver(schedule.beta.float(), num_steps).to(x.device)

        out = torch.cat(
            [
                self.reconstruct(batch, sampler, num_samples, generator)
                for batch in windows.split(self.batch_size)
            ]
        )
        out = out.reshape(x.size())[:, :length, :features]
        out = (out - x[:, :length, :features]).abs().mean(dim=-1)
        return out.reshape(*size, length)

    def reconstruct(
        self,
        x: Tensor,  # [B, L, K]
        sampler: DPMSolver,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> Tensor:  # [B, L, K]

        x = x.transpose(1, 2)  # [B, K, L]
        batch, _, length = x.size()
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-5
        x = ((x - mean) / std).clamp(-10, 10)

        time = torch.arange(length, device=x.device).expand(batch, -1)
        blocks = time // (length // self.num_blocks)

        out = x.new_zeros((num_samples, *x.size()))
        for strategy in range(2):
            mask = ((blocks % 2) == strategy).to(x.dtype)
            mask = mask.unsqueeze(1).expand_as(x)  # [B, K, L]
            predict_noise = partial(
                self.predict_noise,
                cond=mask * x,
                target_mask=1 - mask,
                side_info=self.side_embedding(time, mask),
                strategy=torch.tensor([strategy], device=x.device),
            )
            for i in range(num_samples):
                # Draw noise in the memory layout of `x` to match the
                # reference sampling order for a given generator:
                noise = torch.empty_like(x).normal_(generator=generator)
                sample = sampler(noise, predict_noise) * std + mean
                out[i] += (1 - mask) * sample

        return out.median(dim=0).values.transpose(1, 2)

    def predict_noise(
        self,
        x: Tensor,  # [B, K, L]
        step: Tensor,  # [1]
        *,
        cond: Tensor,  # [B, K, L]
        target_mask: Tensor,  # [B, K, L]
        side_info: Tensor,  # [B, S, K, L]
        strategy: Tensor,  # [1]
    ) -> Tensor:  # [B, K, L]
        return self.denoiser(
            x=torch.stack([cond, target_mask * x], dim=1),
            side_info=side_info,
            step=step,
            strategy=strategy,
        )
