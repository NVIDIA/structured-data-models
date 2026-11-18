# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from sdm.prior import (
    HP,
    Correlated,
    LogUniform,
    Mixture,
    Uniform,
    UniformInt,
)
from sdm.prior.converters.base import NumericalConverter
from sdm.prior.converters.kumaraswamy import (
    KumaraswamyNumericalConverter,
)
from sdm.prior.util import standardize


class DistributionalMixtureConverter(NumericalConverter):
    """Convert latent channels to numerical column values.

    Sample values using a signal, noise scale, and subgroup probability.
    Values may be rounded or warped; sampled values may also replace
    the first channel passed to child nodes.

    Args:
        dist_num_conv_prob: Threshold for selecting three channels rather
            than one. Equality selects one channel.
        dist_count_prob: Probability of rounding three-channel values.
        dist_warp_prob: Probability of selecting from the column-warping
            pool, which includes the identity transform.
        dist_feedback_gain: Upper bound on the log-uniform scale for values
            passed to child nodes; the lower bound is its reciprocal.
        allow_kumaraswamy_warping: Include Kumaraswamy warping in the pool.
    """

    def __init__(
        self,
        dist_num_conv_prob: float = 0.1,
        dist_count_prob: float = 0.3,
        dist_warp_prob: float = 1.0,
        dist_feedback_gain: float = 10.0,
        allow_kumaraswamy_warping: bool = True,
    ) -> None:
        for name, value in (
            ("dist_num_conv_prob", dist_num_conv_prob),
            ("dist_count_prob", dist_count_prob),
            ("dist_warp_prob", dist_warp_prob),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"'{name}' must be finite and in [0, 1]")
        if not math.isfinite(dist_feedback_gain) or dist_feedback_gain < 1.0:
            raise ValueError(
                "'dist_feedback_gain' must be finite and at least 1"
            )

        self.dist_num_conv_prob = dist_num_conv_prob
        self.dist_count_prob = dist_count_prob
        self.dist_warp_prob = dist_warp_prob

        self.use_dist = Correlated(Uniform(0.0, 1.0), key="num_conv_use_dist")
        self.linear: HP[int] | Correlated[int] = Correlated(
            UniformInt(0, 1), key="distributional_linear"
        )
        self.w = Correlated(Uniform(0.0, 1.0), key="distributional_signal")
        self.shift = Correlated(Uniform(0.0, 2.0), key="distributional_shift")
        self.disp = Correlated(Uniform(0.0, 0.5), key="distributional_disp")
        self.slope = Correlated(
            LogUniform(0.1, 3.0), key="distributional_slope"
        )
        self.offset = Correlated(
            Uniform(-5.0, 5.0), key="distributional_offset"
        )
        self.ceil = Correlated(
            LogUniform(1.0, 300.0), key="distributional_ceil"
        )
        self.return_sampled: HP[int] | Correlated[int] = Correlated(
            UniformInt(0, 1), key="dist_num_conv_return_sampled"
        )
        self.feedback_gain: HP[float] | Correlated[float] = Correlated(
            LogUniform(
                1.0 / dist_feedback_gain,
                dist_feedback_gain,
            ),
            key="distributional_feedback_gain",
        )

        modes = ["id"]
        if allow_kumaraswamy_warping:
            modes.append("kumar")
        self.warp_modes = Mixture(
            modes,
            correlated=True,
            key="num_conv_mode",
        )
        self.kumaraswamy = KumaraswamyNumericalConverter(
            a=Correlated(
                LogUniform(0.2, 5.0), key="KumaraswamyNumericalConverter.a"
            ),
            b=Correlated(
                LogUniform(0.2, 5.0), key="KumaraswamyNumericalConverter.b"
            ),
        )

    def sample_num_channels(  # noqa: D102
        self,
        num_columns: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        draw = self.use_dist.sample_batch(
            size=num_columns, generator=generator
        )
        return torch.where(draw < self.dist_num_conv_prob, 3, 1)

    @staticmethod
    def _sample(
        hp: HP,
        num_columns: int,
        x: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        return (
            hp.sample_batch(
                size=num_columns,
                generator=generator,
            )
            .to(x.dtype)
            .unsqueeze(-1)
        )

    def forward(  # noqa: D102
        self,
        slices: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        xs = [value.T for value in slices.unbind()]
        if not xs:
            raise ValueError("at least one numerical slice is required")
        is_dist_values = [value.size(-1) == 3 for value in xs]
        is_dist = torch.tensor(is_dist_values, device=xs[0].device)
        for value, distributional in zip(xs, is_dist_values):
            expected = 3 if distributional else 1
            if value.size(-1) != expected:
                raise ValueError(
                    f"expected {expected} node features, got {value.size(-1)}"
                )

        num_columns = len(xs)
        num_rows = xs[0].size(0)
        first = torch.stack([value[..., 0] for value in xs], dim=0)

        linear = self._sample(
            self.linear, num_columns, first, generator
        ).bool()
        w = self._sample(self.w, num_columns, first, generator)
        shift = self._sample(self.shift, num_columns, first, generator)
        disp = self._sample(self.disp, num_columns, first, generator)
        slope = self._sample(self.slope, num_columns, first, generator)
        offset = self._sample(self.offset, num_columns, first, generator)
        ceiling = self._sample(self.ceil, num_columns, first, generator)

        out = first.clone()
        dist_indices = is_dist.nonzero().flatten()
        if dist_indices.numel() > 0:
            z = torch.stack(
                [
                    5.0
                    * torch.tanh(
                        torch.nan_to_num(standardize(xs[i], dim=-2)[0]) / 5.0
                    )
                    for i in dist_indices.tolist()
                ],
                dim=0,
            )
            subgroup = torch.rand(
                (dist_indices.numel(), num_rows),
                dtype=first.dtype,
                device=first.device,
                generator=generator,
            ) < torch.sigmoid(z[..., 2])
            loc = z[..., 0] + shift[dist_indices] * subgroup.to(first.dtype)
            eps = torch.randn(
                loc.shape,
                dtype=loc.dtype,
                device=loc.device,
                generator=generator,
            )
            eta = (
                w[dist_indices].sqrt() * loc
                + (1.0 - w[dist_indices]).sqrt()
                * torch.exp(
                    disp[dist_indices] * z[..., 1]
                    - 0.5 * disp[dist_indices].square()
                )
                * eps
            )
            value = ceiling[dist_indices] * torch.where(
                linear[dist_indices],
                eta,
                torch.sigmoid(
                    slope[dist_indices] * eta + offset[dist_indices]
                ),
            )
            count = (
                torch.rand(
                    (dist_indices.numel(),),
                    device=first.device,
                    generator=generator,
                )
                < self.dist_count_prob
            )
            out[dist_indices] = torch.where(
                count.unsqueeze(-1), value.round(), value
            )

        warp = (
            torch.rand(
                (num_columns,),
                device=first.device,
                generator=generator,
            )
            < self.dist_warp_prob
        )
        warped = out.clone()
        mode = torch.zeros(
            (num_columns,),
            dtype=torch.long,
            device=first.device,
        )
        if bool(warp.any()):
            mode[warp] = self.warp_modes._draw(
                num_samples=int(warp.sum()),
                generator=generator,
            ).to(mode.device)
        kumar = (
            mode == self.warp_modes.items.index("kumar")
            if ("kumar" in self.warp_modes.items)
            else torch.zeros_like(mode, dtype=torch.bool)
        )
        if bool(kumar.any()):
            kumar_value, _ = self.kumaraswamy.forward(
                slices=out.unsqueeze(1),
                generator=generator,
            )
            warped = torch.where(kumar.unsqueeze(-1), kumar_value, warped)
        feat = torch.where(warp.unsqueeze(-1), warped, out)

        return_sampled = self._sample(
            self.return_sampled,
            num_columns,
            first,
            generator,
        ).bool()
        gain = self._sample(
            self.feedback_gain,
            num_columns,
            first,
            generator,
        )
        feedback = list(xs)
        for i, distributional in enumerate(is_dist_values):
            if not distributional or not bool(return_sampled[i].any()):
                continue
            _, inv_std = standardize(xs[i], dim=-2)
            scale = torch.nan_to_num(
                1.0 / inv_std[..., 0],
                nan=1.0,
                posinf=1.0,
            )
            standardized, _ = standardize(out[i].unsqueeze(-1), dim=-2)
            sampled = gain[i] * scale * standardized.squeeze(-1)
            feedback[i] = xs[i].clone()
            feedback[i][..., 0] = torch.where(
                return_sampled[i],
                sampled,
                feedback[i][..., 0],
            )

        feedback = [value.T for value in feedback]
        mod = (
            torch.nested.nested_tensor_from_jagged(
                values=torch.cat(feedback), offsets=slices.offsets()
            )
            if isinstance(slices, NestedTensor)
            else torch.stack(feedback)
        )
        return feat, mod
