# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, cast

import torch
from torch import Tensor

from sdm.cache import Cache


class ECOC(torch.nn.Module):
    """Extend a classifier with `error-correcting output codes <https://arxiv.org/abs/cs/9501101>`_.

    Each encoded task separates ``max_classes - 1`` original classes and
    merges the others into a rest class. Predictions average log probabilities
    over the tasks that separate each class, ignoring its rest assignments.
    The resulting scores can be normalized with softmax.

    Targets within the model's class capacity use a single ordinary forward
    pass. Larger targets use ``max(ceil(C / (max_classes - 1)),
    4 * ceil(log(C, max_classes)))`` encoded tasks, where ``C`` is the number
    of original classes. Tasks run along an additional leading batch dimension.

    The model is supplied to :meth:`forward`. Manage its parameters, device,
    and training mode directly on that model.

    Args:
        max_classes: Number of classes supported by the model.
    """

    def __init__(self, max_classes: int) -> None:
        super().__init__()
        self.max_classes = max_classes

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        y: Tensor,
        *,
        num_classes: int,
        cache: Cache | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Predict scores over the original classes.

        Args:
            model: In-context classifier accepting ``model(x, y, **kwargs)``
                and returning logits with shape
                ``[..., R_query, max_classes]``. It must support leading batch
                dimensions. To use caching, it must also accept a
                :class:`~sdm.cache.Cache` via the ``cache`` keyword argument.
            x: Context followed by query features, with shape ``[..., R, D]``
                for ``R`` rows and ``D`` features. When replaying a cache,
                provide only query features.
            y: Integer context labels in ``[0, num_classes)``, with shape
                ``[..., R_context]``. Use an empty context axis when replaying
                a cache.
            num_classes: Number of original classes, including any absent
                from the context. Must remain unchanged when replaying a cache.
            cache: Optional cache recording context state and the codebook,
                or replaying them for the same model and context across query
                batches.
            generator: Generator controlling codebook sampling. Ignored when
                replaying a cache.
            kwargs: Arguments forwarded to the model. Tensor arguments must
                broadcast over the additional leading task dimension.

        Returns:
            Scores with shape ``[..., R_query, num_classes]`` for the query
            rows. Within the class capacity these are the model's logits;
            otherwise they are averaged log probabilities.
        """
        if num_classes <= self.max_classes:
            if cache is not None:
                kwargs["cache"] = cache
            return model(x, y, **kwargs)[..., :num_classes]

        if cache is not None and cache.is_replaying:
            codebook = cast(Tensor, cache["ecoc_codebook"])
            if num_classes != codebook.size(1):
                raise ValueError(
                    "'num_classes' must match the cached ECOC codebook "
                    f"(expected {codebook.size(1)}, got {num_classes})"
                )
            kwargs["cache"] = cast(Cache, cache["ecoc_model"])
        else:
            codebook = self._draw_codebook(num_classes, x.device, generator)
            if cache is not None:
                cache["ecoc_codebook"] = codebook
                kwargs["cache"] = cache["ecoc_model"] = Cache()

        T = codebook.size(0)
        # [T, ..., R, D] and [T, ..., R_context], with T encoded tasks.
        # Separate storage also supports models that modify their inputs.
        logits = model(
            x=x.expand(T, *x.shape).clone(),
            y=codebook.index_select(dim=1, index=y.reshape(-1)).view(
                T, *y.shape
            ),
            **kwargs,
        )
        index = codebook.view(T, *(1,) * (logits.dim() - 2), num_classes)
        scores = logits.log_softmax(dim=-1).gather(
            dim=-1,
            index=index.expand(*logits.shape[:-1], num_classes),
        )  # [T, ..., R_query, C]
        active = index != self.max_classes - 1
        return scores.masked_fill(~active, 0).sum(dim=0) / active.sum(dim=0)

    def _draw_codebook(
        self,
        num_classes: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        rest = self.max_classes - 1
        num_codes = max(
            math.ceil(num_classes / rest),
            4 * math.ceil(math.log(num_classes, self.max_classes)),
        )
        draw_device = device if generator is None else generator.device
        # Bound the quadratic distance search for large targets.
        num_draws = 50 if num_classes <= 200 else 1
        codebook = torch.full(
            size=(num_draws, num_codes, num_classes),
            fill_value=rest,
            dtype=torch.long,
            device=draw_device,
        )
        coverage = torch.zeros(num_draws, num_classes, device=draw_device)
        for code in codebook.unbind(dim=1):
            priority = coverage + 0.1 * torch.rand(
                num_draws, num_classes, device=draw_device, generator=generator
            )
            chosen = priority.argsort(dim=-1)[:, :rest]
            symbols = torch.rand(
                num_draws, rest, device=draw_device, generator=generator
            ).argsort(dim=-1)
            code.scatter_(dim=-1, index=chosen, src=symbols)
            coverage.scatter_add_(
                dim=-1,
                index=chosen,
                src=torch.ones_like(chosen, dtype=coverage.dtype),
            )

        if num_draws == 1:
            return codebook[0].to(device)

        distance = torch.zeros(
            num_draws,
            num_classes,
            num_classes,
            dtype=torch.long,
            device=draw_device,
        )
        for code in codebook.unbind(dim=1):
            distance += code.unsqueeze(-1) != code.unsqueeze(-2)
        rows, cols = torch.triu_indices(
            num_classes, num_classes, offset=1, device=draw_device
        )
        pairwise = distance[:, rows, cols]
        # Prefer minimum distance, breaking ties by total pairwise distance.
        score = pairwise.amin(dim=-1) * (
            pairwise.size(-1) * num_codes + 1
        ) + pairwise.sum(dim=-1)
        return (
            codebook.index_select(0, score.argmax().view(1))
            .squeeze(0)
            .to(device)
        )
