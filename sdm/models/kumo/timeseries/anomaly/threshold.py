# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from NVIDIA/Kumo-TS, Kumo-Anomaly/utils/adaptive_threshold.py
# at af74798bbf7d020d3298f545fb9be383560a18ce.

import torch
from torch import Tensor

from sdm import TableTensor


def _bounds(
    scores: Tensor, starts: Tensor, confidence_level: float
) -> tuple[Tensor, Tensor]:
    """Compute the SDK's expanding bounds, restarting at segment starts."""
    sums = torch.cat(
        [torch.zeros_like(scores[..., :1]), scores.cumsum(-1)], dim=-1
    )
    squares = torch.cat(
        [torch.zeros_like(scores[..., :1]), scores.square().cumsum(-1)], dim=-1
    )
    counts = (
        torch.arange(1, scores.size(-1) + 1, device=scores.device) - starts
    )
    mean = (sums[..., 1:] - sums.gather(-1, starts)) / counts
    mean = torch.where(counts == 1, scores, mean)
    variance = (
        squares[..., 1:] - squares.gather(-1, starts)
    ) / counts - mean.square()
    std = variance.clamp_min(0).sqrt()
    width = torch.where(std > 0, 1.5 * std, 0.1)
    if confidence_level > 0.95:
        width = width * 1.2
    elif confidence_level < 0.90:
        width = width * 0.8
    # The first score in each segment always has unscaled +/- 0.1 bounds.
    width = torch.where(counts == 1, 0.1, width)
    return mean - width, mean + width


def _rolling_stats(scores: Tensor, window: int) -> tuple[Tensor, Tensor]:
    """Return trailing means and sample variances with min_periods=1."""
    # Center before prefix sums to reduce cancellation for offset scores.
    centered = scores - scores[..., :1]
    sums = torch.cat(
        [torch.zeros_like(scores[..., :1]), centered.cumsum(-1)], dim=-1
    )
    squares = torch.cat(
        [torch.zeros_like(scores[..., :1]), centered.square().cumsum(-1)],
        dim=-1,
    )
    end = torch.arange(1, scores.size(-1) + 1, device=scores.device)
    start = (end - window).clamp_min(0)
    count = end - start
    total = sums[..., end] - sums[..., start]
    mean = total / count + scores[..., :1]
    variance = squares[..., end] - squares[..., start] - total.square() / count
    variance = variance.clamp_min(0) / (count - 1).clamp_min(1)
    return mean, torch.where(count > 1, variance, 0)


class SCSThreshold(torch.nn.Module):
    """Segmented Confidence Sequences (SCS) with APCA segmentation.

    Segment preprocessed observations, compute expanding score bounds per
    segment, and retain violations strictly above a whole-series percentile.
    Defaults follow the Kumo-Anomaly SDK. Decisions use future observations;
    the confidence setting is a heuristic, not a calibrated guarantee.
    Segmentation copies feature means to CPU; flags stay on the score device.

    Args:
        confidence_level: Bound width setting.
        min_segment_size: Minimum observations on each side of an APCA split.
        percentile_filter: Global score percentile used as a strict cutoff.
    """

    def __init__(
        self,
        confidence_level: float = 0.99,
        min_segment_size: int = 50,
        percentile_filter: float = 95.0,
    ) -> None:
        super().__init__()
        self.confidence_level = confidence_level
        self.min_segment_size = min_segment_size
        self.percentile_filter = percentile_filter

    def _segment_starts(self, data: Tensor) -> Tensor:
        """Return each row's segment start using the published APCA rule."""
        length = data.numel()
        rows = torch.arange(length, device=data.device)
        variation = data.std(correction=0) / (data.mean() + 1e-8)
        # Preserve the signed coefficient of variation used by the SDK.
        if variation < 0.1:
            size = max(200, length // 15)
            return rows // size * size

        starts = data.new_zeros(length, dtype=torch.long)
        sums = torch.cat([data.new_zeros(1), data.cumsum(0)])
        squares = torch.cat([data.new_zeros(1), data.square().cumsum(0)])
        start = 0
        while start < length:
            starts[start:] = start
            if length - start < 2 * self.min_segment_size:
                break
            first = start + self.min_segment_size
            last = length - self.min_segment_size
            if last <= first:
                break
            step = max(1, (last - first) // 200)
            splits = torch.arange(first, last, step, device=data.device)
            left_sum = sums[splits] - sums[start]
            right_sum = sums[-1] - sums[splits]
            error = (
                squares[-1]
                - squares[start]
                - left_sum.square() / (splits - start)
                - right_sum.square() / (length - splits)
            )
            best = error.argmin()
            no_split = (
                squares[-1]
                - squares[start]
                - (sums[-1] - sums[start]).square() / (length - start)
            )
            improvement = 0.7 if variation > 0.3 else 0.5
            if error[best] >= no_split * improvement:
                break
            start = int(splits[best])
        return starts

    @torch.no_grad()
    def forward(self, scores: Tensor, target: Tensor | TableTensor) -> Tensor:
        """Return boolean anomaly flags with shape ``[..., R]``.

        Args:
            scores: Finite reconstruction scores with shape ``[..., R]``.
            target: Finite preprocessed observations of shape ``[..., R, D]``
                without padded features. ``R`` is the number of time steps and
                ``D`` the number of features.
        """
        scores = scores.double()
        data = target.numerical if isinstance(target, TableTensor) else target
        if data.shape[:-1] != scores.shape:
            raise ValueError(
                "Target batch and row dimensions must match scores"
            )
        if scores.numel() == 0:
            return scores.bool()
        data = data.mean(-1).to(device="cpu", dtype=torch.float64)
        starts = (
            torch.stack(
                [
                    self._segment_starts(series)
                    for series in data.reshape(-1, data.size(-1))
                ]
            )
            .reshape(scores.shape)
            .to(scores.device)
        )
        lower, upper = _bounds(scores, starts, self.confidence_level)
        cutoff = scores.quantile(
            self.percentile_filter / 100, dim=-1, keepdim=True
        )
        return ((scores < lower) | (scores > upper)) & (scores > cutoff)


class MACSThreshold(torch.nn.Module):
    """Multi-Scale Adaptive Confidence Segments (MACS) from Kumo-Anomaly.

    Combine expanding score bounds with variance-based weights and a
    regime-change check, then retain violations strictly above a whole-series
    percentile. Defaults follow the SDK, including its shared expanding bounds
    across scales. Decisions use future observations; the confidence setting
    is a heuristic, not a calibrated guarantee.

    Args:
        short_window: Window used for variance-based attention weights.
        medium_window: SDK scale setting; bounds remain expanding.
        long_window: Window used to identify regime changes.
        confidence_level: Bound width setting.
        percentile_filter: Global score percentile used as a strict cutoff.
    """

    def __init__(
        self,
        short_window: int = 10,
        medium_window: int = 50,
        long_window: int = 250,
        confidence_level: float = 0.95,
        percentile_filter: float = 97.0,
    ) -> None:
        super().__init__()
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.confidence_level = confidence_level
        self.percentile_filter = percentile_filter

    @torch.no_grad()
    def forward(self, scores: Tensor) -> Tensor:
        """Return boolean anomaly flags with shape ``[..., R]``.

        Args:
            scores: Finite reconstruction scores with shape ``[..., R]``,
                where ``R`` is the number of time steps.
        """
        scores = scores.double()
        if scores.numel() == 0:
            return scores.bool()
        starts = torch.zeros_like(scores, dtype=torch.long)
        lower, upper = _bounds(scores, starts, self.confidence_level)
        window = max(2, min(self.short_window, scores.size(-1) // 10))
        _, variance = _rolling_stats(scores, window)
        maximum = variance.amax(dim=-1, keepdim=True)
        normalized = variance / torch.where(maximum > 0, maximum, 1)
        group = torch.where(normalized > 0.7, 2, (normalized > 0.3).long())
        weights = scores.new_tensor(
            [[0.1, 0.3, 0.6], [0.2, 0.6, 0.2], [0.6, 0.3, 0.1]]
        )[group]
        short_weight, medium_weight, long_weight = weights.unbind(-1)
        # Keep the SDK's weighted arithmetic even though the bounds coincide.
        combined_lower = (
            short_weight * lower + medium_weight * lower + long_weight * lower
        )
        combined_upper = (
            short_weight * upper + medium_weight * upper + long_weight * upper
        )
        mean, variance = _rolling_stats(scores, self.long_window)
        std = variance.sqrt()
        previous_mean = torch.cat(
            [torch.zeros_like(scores[..., :1]), mean[..., :-1]], dim=-1
        )
        previous_std = torch.cat(
            [torch.zeros_like(scores[..., :1]), std[..., :-1]], dim=-1
        )
        denominator = previous_std + 1e-8
        regime = ((mean - previous_mean).abs() / denominator > 2) | (
            (std - previous_std).abs() / denominator > 1.5
        )
        violations = (scores < lower) | (scores > upper)
        attention = (scores < combined_lower) | (scores > combined_upper)
        anomalies = attention & (~regime | violations)
        cutoff = scores.quantile(
            self.percentile_filter / 100, dim=-1, keepdim=True
        )
        return anomalies & (scores > cutoff)
