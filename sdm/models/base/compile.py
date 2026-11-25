# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compilation of :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import Module

from sdm import RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models.base import Callback

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


def compile_predict(
    model: ICLModel,
    dynamic_dims: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    For a model with ``_compilable_models``, a graph holds one call of the
    model of a task on one batch of estimators, and is compiled on the first
    such call. All other steps of ``predict`` run uncompiled, so that the
    number of estimators and the names of columns, classes and categories do
    not enter the graph. Calls reuse a graph while its guards hold, *e.g.*,
    with the same precision, gradient mode, device and estimator batch size:
    each batch size compiles its own graph.
    :class:`~sdm.models.KumoTabular` shares a graph among up to 10 classes;
    more classes compile further graphs.

    One graph serves queries and fits of any size along the dynamic
    dimensions, except that a size of zero or one compiles a graph of its own.
    A single query row is repeated instead, and a query without rows is
    predicted without the graph.

    Other models compile ``predict`` as a whole, with static shapes.

    Args:
        model: The model.
        dynamic_dims: The dimensions of the query ``[..., R, C]`` of a model
            to mark as dynamic instead of the default of ``model``, counted
            from the last one, *e.g.*, ``(-2,)`` for the rows, which requires
            that the model predicts every query row independently of the
            other rows. The tensors that ``fit`` saved follow the query: with
            ``-2``, the context rows are dynamic as well. ``()`` adds no
            marks, and every new shape of the query compiles another graph.
            A dimension of the query that the model cannot keep dynamic fails
            to compile. Not supported for a ``predict`` compiled as a whole.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``False``, so that only the marked
            dimensions are dynamic. ``isolate_recompiles`` defaults to
            ``True``, since all models would otherwise share one recompile
            limit.
    """
    kwargs = {"dynamic": False, "isolate_recompiles": True} | kwargs
    if not model._compilable_models():
        if dynamic_dims:
            raise ValueError(
                f"{model.__class__.__name__!r} does not support dynamic "
                f"dimensions in a compiled 'predict'"
            )
        return torch.compile(model.predict, **kwargs)

    dims = (
        model._default_dynamic_dims() if dynamic_dims is None else dynamic_dims
    )
    # By model, compiled on its first call: the models of ``model`` may be
    # replaced.
    compiled: dict[Module, Callable[..., Tensor]] = {}

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        models = {}
        for task, module in model._compilable_models().items():
            if module not in compiled:
                compiled[module] = torch.compile(module, **kwargs)
            models[task] = partial(_call, module, compiled[module], dims)
        return model._predict(
            x=x,
            related_tables=related_tables,
            callbacks=callbacks,
            models=models,
        )

    return predict


def _call(
    module: Module,
    compiled: Callable[..., Tensor],
    dims: tuple[int, ...],
    x: Tensor,  # [..., R, C]
    *args: Any,
    cache: Cache,
    **kwargs: Any,
) -> Tensor:
    rows = x.size(-2)
    if rows == 0:
        return module(x, *args, cache=cache, **kwargs)
    # A graph specializes on sizes of zero and one, so a single row is
    # repeated if the rows are dynamic. It also specializes on the strides and
    # the storage offset of a view, such as a slice or the expanded query of
    # ECOC, so that the query is copied.
    repeat = rows == 1 and -2 in dims
    x = (
        torch.cat([x, x], dim=-2)
        if repeat
        else x.clone(memory_format=torch.contiguous_format)
    )
    # The graph gets the query and the fitted state on its device, which are
    # copies without the marks of earlier calls.
    for dim in dims:
        torch._dynamo.mark_dynamic(x, dim % x.dim())
    cache._mark_dynamic(dims)
    out = compiled(x, *args, cache=cache, **kwargs)
    return out[..., :1, :] if repeat else out
