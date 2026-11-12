# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models.base.callback import Callback

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


@dataclass(frozen=True)
class DynamicShapes:
    r"""Dimensions allowed to vary during compiled prediction.

    Args:
        query: Dynamic query dimensions.
        cache: Reserved for fitted tensor declarations: pairs of cache paths
            and dynamic dimensions. Paths name mapping keys, sequence indices,
            or attributes, starting at the fitted cache root. No cache marking
            or reuse across refits is implemented yet.
    """

    query: tuple[int, ...] = ()
    cache: tuple[tuple[tuple[str | int, ...], tuple[int, ...]], ...] = ()


def compile_predict(
    model: ICLModel,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    With ``dynamic=False``, mark the query dimensions the model declares.
    The graph is captured on two rows after every fit, so its chunk guards
    hold the rows that fit one chunk. Queries the graph rejects run eagerly,
    with the chunking of the model; nothing recompiles. Empty queries and
    queries with related tables or callbacks run eagerly. Singleton rows are
    repeated to avoid PyTorch's zero/one specialization; outputs are sliced
    back.

    Args:
        model: The model.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``isolate_recompiles`` defaults to ``True``, since all models
            would otherwise share one recompile limit.
    """
    kwargs = {"isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    shapes = model._dynamic_shapes
    if kwargs.get("dynamic") is not False or not shapes.query:
        return compiled

    captured: weakref.ref[Cache] | None = None

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        nonlocal captured
        rows = x.size(-2)
        if rows == 0 or callbacks is not None or related_tables is not None:
            return model.predict(x, related_tables, callbacks=callbacks)
        if captured is None or captured() is not model._cache:
            # Two rows capture the same graph whatever the first query is.
            compiled(_prepare_query(x[..., :2, :], shapes.query))
            # A weak reference lets a refit release the previous cache.
            captured = weakref.ref(model._cache)
        x = _prepare_query(x, shapes.query)
        # A query that misses the chunk guards of the graph runs eagerly.
        with torch.compiler.set_stance("eager_on_recompile"):
            return compiled(x)[..., :rows, :]

    return predict


def _prepare_query(x: Tensor, dims: tuple[int, ...]) -> Tensor:
    # One layout for views and singleton padding, including text storage.
    index = torch.arange(max(x.size(-2), 2), device=x.device)
    x = x.index_select(-2, index.clamp_(max=x.size(-2) - 1))
    if isinstance(x, TableTensor):
        x._mark_dynamic(dims)
        return x
    for dim in dims:
        torch._dynamo.mark_dynamic(x, dim % x.dim())
    return x
