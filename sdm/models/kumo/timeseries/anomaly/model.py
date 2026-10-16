# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import torch
from torch import Tensor

from sdm import Recipe, Stype, TableTensor
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.anomaly.ckpt import model_kwargs, remap_ckpt
from sdm.models.kumo.timeseries.anomaly.denoiser import DiffusionDenoiser
from sdm.models.kumo.timeseries.anomaly.embedding import SideEmbedding
from sdm.models.kumo.timeseries.anomaly.recipe import default_recipe
from sdm.models.kumo.timeseries.anomaly.sampling import DPMSolver
from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule


class AnomalyOutput(NamedTuple):
    """Per-row anomaly scores and model-space reconstruction.

    Attributes:
        scores: Mean absolute reconstruction error, shape ``[time]``.
        l2_scores: Euclidean reconstruction error, shape ``[time]``.
        reconstruction: Reconstructed series in min-max-scaled feature or
            PCA space, shape ``[time, min(input_features, model_features)]``.
        target: Preprocessed input in the same space as reconstruction.

    Padding features are excluded from every field. Higher scores indicate
    larger reconstruction errors, not calibrated anomaly probabilities.
    """

    scores: Tensor
    l2_scores: Tensor
    reconstruction: TableTensor
    target: TableTensor


class KumoAnomaly(torch.nn.Module):
    """Pretrained Kumo-Anomaly diffusion reconstruction and anomaly scoring.

    Unlike supervised ICL models, this model needs no labels or prediction
    task. Pass one ordered numerical series to :meth:`forward`. Alternating
    temporal blocks are reconstructed using two complementary conditioning
    masks and DPM-Solver++; the sample median supplies the reconstruction.

    The port corrects the original SDK's double timestep conversion. It
    can change predictions; SDK score or threshold equivalence is not
    claimed. Validate downstream thresholds again. The released conditional
    checkpoint is supported; training, DDPM/DDIM sampling, and the SDK's
    dataset-specific preprocessing and reporting are not included.

    Args:
        pretrained: Load the published checkpoint when true, initialize
            random weights when false, or load an upstream-format local
            checkpoint when given a path. Loading uses ``weights_only=True``.
        window_length: Non-overlapping reconstruction window length.
            Defaults to the SDK inference length, not training metadata.
        mask_blocks: Even number of alternating blocks per window. Must
            divide ``window_length``. The SDK inference default is four.
        device: Device for parameters and inference.
    """

    def __init__(
        self,
        pretrained: bool | str | Path = True,
        window_length: int = 100,
        mask_blocks: int = 4,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if (
            window_length < 1
            or mask_blocks < 2
            or mask_blocks % 2
            or window_length % mask_blocks
        ):
            raise ValueError(
                "'mask_blocks' must be even and divide a positive "
                "'window_length'"
            )
        self.window_length = window_length
        self.mask_blocks = mask_blocks
        self.scale_factor = 1.0
        if pretrained is not False:
            path = (
                download_checkpoint(
                    repo_id="nvidia/Kumo-Anomaly",
                    filename="final_model.pth",
                    revision="226c5003b0582adc1cac433123f47324bfae39f2",
                )
                if pretrained is True
                else pretrained
            )
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True
            )
            config = checkpoint["config"]
            self.model = _KumoAnomaly(**model_kwargs(config))
            self.model.load_state_dict(remap_ckpt(checkpoint["model"]))
            self.scale_factor = config["dataset"].get("scale_factor", 1.0)
        else:
            self.model = _KumoAnomaly()
        self.to(device=device)
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        """Return the published checkpoint's feature preprocessing recipe."""
        return default_recipe()

    @torch.no_grad()
    def forward(
        self,
        x: Tensor | TableTensor,
        *,
        reference: Tensor | TableTensor | None = None,
        num_samples: int = 30,
        num_steps: int = 20,
        batch_size: int = 8,
        generator: torch.Generator | None = None,
    ) -> AnomalyOutput:
        """Reconstruct and score every row of an ordered series.

        Args:
            x: Finite numerical observations, shape ``[time, features]``.
                Select measured features only, excluding timestamps, IDs,
                and labels. No sorting or resampling is performed.
            reference: Optional reference rows with the same ordered columns
                for fitting PCA and min-max ranges. Without reference,
                preprocessing fits on ``x`` (transductive SDK behavior).
                Even with reference, reconstruction uses the whole query
                window, including future rows; this is not a causal detector.
            num_samples: Reconstruction samples per masking strategy.
                Even counts use the lower median, matching Kumo-TS.
            num_steps: Denoiser evaluations per sample and strategy.
            batch_size: Number of windows reconstructed together. Changing
                batching changes the assignment of random draws to windows.
            generator: Random generator on the model device. Supplied
                generators leave the global random state unchanged.

        Returns:
            Scores and reconstructions in preprocessed model space, not raw
            input units. Short/final partial windows repeat their last row
            and are trimmed afterward; unlike the original SDK, no trailing
            observations are dropped. Padding can affect that window's
            normalization and reconstruction. No anomaly threshold is fitted.
        """
        if self.training:
            raise RuntimeError("Call 'eval()' before anomaly inference")
        if num_samples < 1:
            raise ValueError("'num_samples' must be at least one")
        x = self._as_table(x)
        reference = x if reference is None else self._as_table(reference)
        if reference.columns != x.columns:
            raise ValueError(
                "Reference and query must have the same ordered columns"
            )
        recipe = default_recipe(
            num_features=self.model.num_features,
            scale_factor=self.scale_factor,
        )
        recipe.features.fit(reference)
        processed = recipe.features.transform(x)
        values = processed.numerical
        length, features = values.shape
        padding = (-length) % self.window_length
        if padding:
            values = torch.cat([values, values[-1:].expand(padding, -1)])
        windows = values.reshape(-1, self.window_length, features)
        sampler = self.model.make_sampler(num_steps).to(values.device)
        reconstruction = torch.cat(
            [
                self.model(
                    windows=batch,
                    sampler=sampler,
                    num_samples=num_samples,
                    mask_blocks=self.mask_blocks,
                    generator=generator,
                )
                for batch in windows.split(batch_size)
            ],
            dim=0,
        ).reshape(-1, features)[:length]
        count = min(x.size(-1), features)
        target = processed.numerical[:, :count]
        reconstruction = reconstruction[:, :count]
        # The SDK bounds residuals in model space before MAE/L2 reduction.
        error = reconstruction.clamp(-1e8, 1e8).nan_to_num(
            nan=0.0
        ) - target.clamp(-1e8, 1e8).nan_to_num(nan=0.0)
        columns: dict[str, Sequence[str]] = {
            Stype.numerical: x.columns[Stype.numerical]
            if x.size(-1) <= features
            else [f"pca_{i}" for i in range(features)]
        }
        return AnomalyOutput(
            scores=error.abs().mean(dim=-1).clamp(0, 1e8),
            l2_scores=error.norm(dim=-1).clamp(0, 1e8),
            reconstruction=TableTensor(
                columns=columns, numerical=reconstruction
            ),
            target=TableTensor(columns=columns, numerical=target),
        )

    def _as_table(self, x: Tensor | TableTensor) -> TableTensor:
        """Normalize inputs and reject silently invalid features."""
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if x.ndim != 2 or x.size(-2) == 0 or x.size(-1) == 0:
            raise ValueError("Expected a nonempty [time, features] series")
        if x.numerical.size(-1) != x.size(-1):
            raise ValueError(
                "Select numerical measurement columns before inference"
            )
        weight = self.model.side_embedding.embed_layer.weight
        x = cast(TableTensor, x.to(device=weight.device, dtype=weight.dtype))
        if not x.numerical.isfinite().all():
            raise ValueError("Impute non-finite observations before inference")
        return x


class _KumoAnomaly(torch.nn.Module):
    """Compose the checkpoint-compatible conditional diffusion modules."""

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
        beta_start: float = 0.0001,
        beta_end: float = 0.01,
        schedule: Literal["linear", "quad"] = "linear",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.schedule_kwargs: dict[str, Any] = {
            "num_steps": num_steps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "schedule": schedule,
        }
        self.side_embedding = SideEmbedding(
            num_features=num_features,
            time_channels=time_channels,
            feature_channels=feature_channels,
        )
        self.denoiser = DiffusionDenoiser(
            channels=channels,
            side_channels=time_channels + feature_channels,
            num_steps=num_steps,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_channels=embedding_channels,
        )

    def make_sampler(self, num_steps: int) -> DPMSolver:
        """Build inference coefficients on CPU once per scored series."""
        schedule = DiffusionSchedule(**self.schedule_kwargs, device="cpu")
        return DPMSolver(betas=schedule.beta.float(), num_steps=num_steps)

    def forward(
        self,
        windows: Tensor,
        sampler: DPMSolver,
        num_samples: int = 30,
        mask_blocks: int = 4,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Reconstruct windows of shape ``[batch, time, features]``."""
        observed = windows.transpose(1, 2)  # [B, K, L]
        batch, _, length = observed.shape
        mean = observed.mean(dim=(1, 2), keepdim=True)
        std = observed.std(dim=(1, 2), keepdim=True) + 1e-5
        normalized = ((observed - mean) / std).clamp(-10, 10)
        time = torch.arange(length, device=windows.device).expand(batch, -1)
        blocks = time // (length // mask_blocks)
        predictions = []
        for strategy in range(2):
            mask = ((blocks % 2) == strategy).to(windows.dtype)
            mask = mask.unsqueeze(1).expand_as(observed)
            side = self.side_embedding(time, mask)
            strategy_index = torch.tensor([strategy], device=windows.device)

            predict_noise = partial(
                self._predict_noise,
                conditioned=mask * normalized,
                target_mask=1 - mask,
                side=side,
                strategy=strategy_index,
            )

            samples = []
            for _ in range(num_samples):
                # Preserve the source's randn_like memory layout so an
                # identical generator assigns noise to the same features.
                noise = torch.empty_like(normalized).normal_(
                    generator=generator
                )
                sample = sampler(noise, predict_noise) * std + mean
                samples.append((1 - mask) * sample)
            predictions.append(torch.stack(samples, dim=1))
        return (
            (predictions[0] + predictions[1])
            .median(dim=1)
            .values.transpose(1, 2)
        )

    def _predict_noise(
        self,
        x: Tensor,
        step: Tensor,
        *,
        conditioned: Tensor,
        target_mask: Tensor,
        side: Tensor,
        strategy: Tensor,
    ) -> Tensor:
        """Condition the denoiser without converting solver indices."""
        return self.denoiser(
            x=torch.stack([conditioned, target_mask * x], dim=1),
            side_info=side,
            step=step,
            strategy=strategy,
        )
