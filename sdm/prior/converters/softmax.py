# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from sdm.prior import HP, Correlated, LogUniform
from sdm.prior.converters import CategoricalConverter
from sdm.prior.functions import RandomFunction
from sdm.prior.root_samplers import RootSampler
from sdm.prior.util import random_weight, standardize


class SoftmaxCategoricalConverter(CategoricalConverter):
    """Convert latent channels to category indices with softmax or argmax.

    Each channel is standardized over rows, scaled, and given a random
    category bias. Categories are sampled from softmax probabilities or
    selected by argmax.

    Args:
        output_mode: Representation passed to child nodes. ``"identity"``
            keeps input slices, ``"index"`` repeats category indices across
            channels,
            and ``"embedding"`` supplies random category embeddings.
        sample_categories: Whether to sample categories rather than take
            the argmax.
        temperature: Logit multiplier. Defaults to
            ``Correlated(LogUniform(0.1, 10.0))``.
        argmax_prob: Probability of overriding ``sample_categories`` with
            argmax using category biases floored at ``-argmax_bias_floor``.
        argmax_bias_floor: Sampler for the magnitude of the negative bias
            floor. Defaults to ``Correlated(LogUniform(0.5, 6.0))``.
        root_sampler: Initial values for category embeddings. Required for
            ``"embedding"`` output mode.
        random_fn: Random function applied to category embeddings.
            Required for ``"embedding"`` output mode.
    """

    def __init__(
        self,
        output_mode: str | HP[str],
        sample_categories: bool | HP[bool] = True,
        temperature: HP[float] | Correlated[float] | float | None = None,
        argmax_prob: float = 0.0,
        argmax_bias_floor: HP[float] | Correlated[float] | None = None,
        *,
        root_sampler: RootSampler | None = None,
        random_fn: RandomFunction | None = None,
    ) -> None:
        if not 0.0 <= argmax_prob <= 1.0:
            raise ValueError("'argmax_prob' must be between 0 and 1")
        self.output_mode = HP.from_value(output_mode)
        self.sample_categories = HP.from_value(sample_categories)
        self.argmax_prob = argmax_prob
        self.argmax_bias_floor = (
            argmax_bias_floor
            if argmax_bias_floor is not None
            else Correlated(
                LogUniform(0.5, 6.0),
                key="SoftmaxCategoricalConverter.bias_floor",
            )
        )
        self.temperature = HP.from_value(
            temperature
            if temperature is not None
            else Correlated(
                LogUniform(0.1, 10.0),
                key="SoftmaxCategoricalConverter.temperature",
            )
        )
        self._root_sampler = root_sampler
        self._random_fn = random_fn

    def forward(  # noqa: D102
        self,
        slices: Tensor,  # [num_cols, num_categories, num_rows]
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = slices
        x_in = x
        offsets: Tensor | None = None
        count: Tensor | None = None
        mask: Tensor | None = None
        if isinstance(x, NestedTensor):
            offsets = x.offsets()
            count = offsets.diff()
            max_k = x._max_seqlen
            x = x.to_padded_tensor(0.0)  # [:, :max_k]
            mask = torch.arange(max_k, device=x.device) < count.view(-1, 1)
            mask = mask.unsqueeze(-1)

        # Each logit channel is standardized over rows. Padded category
        # channels are masked below and therefore do not affect real channels.
        x, _ = standardize(x, dim=-1)

        num_cols, max_k, _ = x.size()

        a = (
            self.temperature.sample_batch(
                size=num_cols,
                generator=generator,
            )
            .to(x.dtype)
            .view(num_cols, 1, 1)
        )
        w = random_weight(
            size=(num_cols, max_k),
            dtype=x.dtype,
            generator=generator,
        ).unsqueeze(-1)  # [num_cols, max_k, 1]
        b = (w + 1e-4).log()

        sample_categories = self.sample_categories.sample(generator=generator)
        if self.argmax_prob > 0.0 and bool(
            torch.rand((), device=x.device, generator=generator)
            < self.argmax_prob
        ):
            # Bound negative category biases before taking the argmax.
            sample_categories = False
            floor = (
                self.argmax_bias_floor.sample_batch(
                    size=num_cols,
                    generator=generator,
                )
                .to(x.dtype)
                .view(num_cols, 1, 1)
            )
            b = torch.maximum(b, -floor)

        logits = a * x + b
        if sample_categories:
            # Sampling requires finite logits. Mask padded categories after
            # sanitizing so they cannot receive probability mass.
            logits = torch.nan_to_num(logits, nan=0.0)

        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        # Permute `k` to the last dim so reduction/sampling reads contiguous
        # memory:
        logits = logits.transpose(1, 2).contiguous()
        if sample_categories:
            probabilities = torch.softmax(logits, dim=-1)
            out = torch.multinomial(
                probabilities.view(-1, max_k),
                num_samples=1,
                replacement=True,
                generator=generator,
            ).view(num_cols, -1)
        else:
            out = logits.argmax(dim=-1)

        output_mode = self.output_mode.sample(generator=generator)
        if output_mode == "identity":
            return out, x_in

        if output_mode == "index":
            mod = out.to(x.dtype).unsqueeze(1).expand(-1, max_k, -1)
        else:
            assert output_mode == "embedding"
            assert self._root_sampler is not None, (
                "output_mode='embedding' requires root_sampler"
            )
            assert self._random_fn is not None, (
                "output_mode='embedding' requires random_fn"
            )
            cardinalities = (
                count.tolist() if count is not None else [max_k] * num_cols
            )
            mods = []
            for c, k in enumerate(cardinalities):
                z = self._random_fn.forward(
                    x=self._root_sampler.sample(
                        num_rows=k, channels=k, generator=generator
                    ),
                    out_channels=k,
                    generator=generator,
                )
                mods.append(z[out[c]].T)

            if offsets is not None:
                mod = torch.nested.nested_tensor_from_jagged(
                    values=torch.cat(mods), offsets=offsets
                )
            else:
                mod = torch.stack(mods)

        if (
            offsets is not None
            and count is not None
            and not isinstance(mod, NestedTensor)
        ):
            mask = torch.arange(max_k, device=x.device)
            mask = mask.view(1, -1) < count.view(-1, 1)  # [C, max_k]
            mod = torch.nested.nested_tensor_from_jagged(
                values=mod[mask],
                offsets=offsets,
            )

        return out, mod
