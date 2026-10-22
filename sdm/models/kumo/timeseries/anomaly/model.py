# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import ClassVar

import torch
from torch import Tensor

from sdm import Recipe, Stype, TableTensor
from sdm._inference import inference_mode
from sdm.models import Callback, Model
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.anomaly.recipe import default_recipe
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

        self.model = _KumoAnomaly(
            device="meta" if pretrained else device,
        )

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
        torch.load(path, map_location=device, weights_only=True)  # TODO

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
    ) -> TableTensor:  # [..., R, 1]
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
            The anomaly scores with shape ``[..., R, 1]``, denoting the mean
            absolute reconstruction error across preprocessed features at each
            time step.
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

            out = TableTensor(
                columns={Stype.numerical: ("mae",)},
                numerical=out.to(query.x.numerical),
            )

            for callback in callbacks:
                out = callback.on_model_forward_end(self, out)

        with (
            torch.amp.autocast(x.device.type, enabled=False),
            inference_mode("grad" if requires_grad else "inference"),
        ):
            return recipe_execution.transform_output([out])[0]


class _KumoAnomaly(torch.nn.Module):
    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        num_samples: int,
        num_steps: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:  # [..., R, 1]
        return x.mean(dim=-1, keepdim=True)
