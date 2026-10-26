# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

import copy
from collections.abc import Sequence
from functools import partial
from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import CategoricalTensor, Recipe, Stype, TableTensor
from sdm._inference import inference_mode
from sdm.models import Callback, Model
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.anomaly.ckpt import remap_ckpt
from sdm.models.kumo.timeseries.anomaly.denoiser import DiffusionDenoiser
from sdm.models.kumo.timeseries.anomaly.embedding import SideEmbedding
from sdm.models.kumo.timeseries.anomaly.recipe import default_recipe
from sdm.models.kumo.timeseries.anomaly.sampling import DPMSolver
from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule
from sdm.models.kumo.timeseries.anomaly.threshold import (
    macs_threshold,
    scs_threshold,
)
from sdm.processing.execution import MemberQuery, RecipeExecution


class KumoAnomaly(Model):
    r"""The time series anomaly detection foundation model from the
    `"NV-Tesseract-AD: Curriculum-Enhanced Conditional Diffusion for
    Multivariate Time-Series Anomaly Detection"
    <https://arxiv.org/abs/2610.12754>`__ paper.

    :class:`KumoAnomaly` reconstructs multivariate time series with conditional
    diffusion and uses reconstruction errors to score anomalies. It combines
    sinusoidal time embeddings and learned feature embeddings with
    diffusion-step and masking-strategy embeddings. Its denoiser alternates
    attention over time and features with gated residuals and aggregated skip
    connections that predict noise at each time step and feature.

    At inference, two complementary masks hide alternating temporal blocks.
    Reverse diffusion reconstructs each hidden block from the visible context,
    and the sample median combines repeated reconstructions. Mean absolute
    reconstruction error across features gives a score per time step; larger
    scores indicate less accurately reconstructed observations.

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.model = _KumoAnomaly(device=device)

        if pretrained:
            self._load_from_pretrained(device=device)

        self.eval()

    def _load_from_pretrained(
        self,
        device: torch.device | str | None,
    ) -> KumoAnomaly:
        device = torch.get_default_device() if device is None else device

        path = download_checkpoint(
            repo_id="nvidia/Kumo-Anomaly",
            filename="final_model.pth",
            revision="226c5003b0582adc1cac433123f47324bfae39f2",
        )
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(remap_ckpt(ckpt["model"]))

        return self

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def forward(
        self,
        x: Tensor | TableTensor,  # [ ..., R, D]
        *,
        recipe: Recipe | None = None,
        num_samples: int = 30,
        num_steps: int = 20,
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
    ) -> TableTensor:  # [..., R, 3]
        r"""The forward pass.

        Args:
            x: The feature tensor with shape ``[..., R, D]`` with ``R`` rows
                and ``D`` columns.
            recipe: The custom recipe for pre- and post-processing.
            num_samples: The number of reconstruction samples per complementary
                masking strategy. Reconstructions are combined using the
                sample median.
            num_steps: The number of denoiser evaluations per sample and
                masking strategy.
            callbacks: Callbacks applied in sequence to this model call.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.

        Returns:
            A table with shape ``[..., R, 3]`` containing numerical ``mae``
            reconstruction scores and categorical boolean ``scs`` and ``macs``
            anomaly labels at each time step. SCS (Segmented Confidence
            Sequences) segments the preprocessed series and thresholds scores
            using expanding bounds within each segment. MACS (Multi-Scale
            Adaptive Confidence Segments) uses variance-weighted expanding
            score bounds.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = self.training
        requires_grad |= any(callback.requires_grad for callback in callbacks)

        recipe_execution = RecipeExecution(
            self.default_recipe() if recipe is None else copy.deepcopy(recipe)
        )
        with (
            torch.amp.autocast(x.device.type, enabled=False),
            inference_mode("no_grad" if requires_grad else "inference"),
        ):
            query = recipe_execution.fit_transform(
                x=x,
                y=None,
                related_tables=None,
                num_members=1,
                generator=generator,
            )[0]

        for callback in callbacks:
            query = MemberQuery(
                *callback.on_query_preprocessing_end(
                    model=self,
                    x=query.x,
                    related_tables=query.related_tables,
                ),
            )

        self._validate_feature_stype(query.x)

        with inference_mode("grad" if requires_grad else "inference"):
            out = self.model(
                x=query.x.numerical,
                num_samples=num_samples,
                num_steps=num_steps,
                generator=generator,
            )

            scs = scs_threshold(out, query.x.numerical).long()
            macs = macs_threshold(out).long()
            categories = torch.tensor([False, True], device=out.device)
            out = TableTensor(
                columns={
                    Stype.numerical: ("mae",),
                    Stype.categorical: ("scs", "macs"),
                },
                numerical=out.unsqueeze(-1),
                categorical=CategoricalTensor(
                    code=torch.stack([scs, macs], dim=-1),
                    categories=(categories, categories),
                ),
            )

            for callback in callbacks:
                out = callback.on_model_forward_end(self, out)

            out = cast(TableTensor, out.to(query.x.dtype))

        with (
            torch.amp.autocast(x.device.type, enabled=False),
            inference_mode("grad" if requires_grad else "inference"),
        ):
            return recipe_execution.transform_output([out])[0]


class _KumoAnomaly(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 40,
        time_channels: int = 256,
        feature_channels: int = 64,
        channels: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        embedding_channels: int = 256,
        num_steps: int = 500,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.in_channels = in_channels

        self.side_embedding = SideEmbedding(
            in_channels=in_channels,
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
        window_size: int = 100,
        num_blocks: int = 4,
        batch_size: int = 32,
        generator: torch.Generator | None = None,
    ) -> Tensor:  # [..., R]

        if x.size(-1) > self.in_channels:
            raise ValueError(
                f"'KumoAnomaly' supports at most {self.in_channels} columns "
                f"after pre-processing (got {x.size(-1)})"
            )

        *B, R, D = x.size()
        x = F.pad(x.reshape(-1, R, D), (0, self.in_channels - D))

        # Pad last window by repeating last row:
        x = torch.cat([x, x[:, -1:].expand(-1, -R % window_size, -1)], dim=-2)
        window = x.reshape(-1, window_size, self.in_channels)

        schedule = DiffusionSchedule(
            num_steps=self.denoiser.num_steps,
            beta_start=0.0001,
            beta_end=0.01,
            schedule="linear",
            device="cpu",
        )
        sampler = DPMSolver(schedule.beta.float(), num_steps).to(x.device)

        out = torch.cat(
            [
                self.reconstruct(
                    x=_x,
                    sampler=sampler,
                    num_samples=num_samples,
                    num_blocks=num_blocks,
                    generator=generator,
                )
                for _x in window.split(batch_size)
            ],
            dim=0,
        )

        out = out.view(x.size())[:, :R, :D].clamp(-1e8, 1e8)
        x = x[:, :R, :D].clamp(-1e8, 1e8)
        out = (out - x).abs().mean(dim=-1).clamp(max=1e8)
        return out.view(*B, R)

    def reconstruct(
        self,
        x: Tensor,  # [B, R, D]
        sampler: DPMSolver,
        num_samples: int,
        num_blocks: int,
        generator: torch.Generator | None,
    ) -> Tensor:  # [B, R, D]

        B, R, _ = x.size()
        x = x.transpose(1, 2)  # [B, D, R]
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-5
        x = ((x - mean) / std).clamp(-10, 10)

        time = torch.arange(R, device=x.device).expand(B, -1)
        blocks = time // (R // num_blocks)

        out = x.new_zeros((num_samples, *x.size()))
        for strategy in range(2):
            mask = ((blocks % 2) == strategy).to(x.dtype)
            mask = mask.unsqueeze(1).expand_as(x)  # [B, D, R]
            predict_noise = partial(
                self.predict_noise,
                cond=mask * x,
                target_mask=1 - mask,
                side_info=self.side_embedding(time, mask),
                strategy=torch.tensor([strategy], device=x.device),
            )
            for i in range(num_samples):
                noise = torch.empty_like(x).normal_(generator=generator)
                sample = sampler(noise, predict_noise) * std + mean
                out[i] += (1 - mask) * sample

        return out.median(dim=0).values.transpose(1, 2)

    def predict_noise(
        self,
        x: Tensor,  # [B, D, R]
        step: Tensor,  # [1]
        *,
        cond: Tensor,  # [B, D, R]
        target_mask: Tensor,  # [B, D, R]
        side_info: Tensor,  # [B, S, D, R]
        strategy: Tensor,  # [1]
    ) -> Tensor:  # [B, D, R]
        return self.denoiser(
            x=torch.stack([cond, target_mask * x], dim=1),
            side_info=side_info,
            step=step,
            strategy=strategy,
        )
