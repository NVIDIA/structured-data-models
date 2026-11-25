# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import Module

from sdm import RelatedTables, TableTensor
from sdm.models.base import Callback

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


@dataclass(frozen=True)
class DynamicShapes:
    r"""The dimensions that a compiled graph of a model keeps dynamic.

    A model returns its default from ``ICLModel._default_dynamic_shapes``,
    and :func:`compile_predict` marks them.
    """

    #: The dimensions of the query. Rows (``-2``) require that the model
    #: predicts every query row independently of the other rows.
    query: tuple[int, ...] = ()


def compile_predict(
    model: ICLModel,
    dynamic_shapes: DynamicShapes | None = None,
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

    One graph serves queries of any size along the default
    :class:`DynamicShapes` of ``model``. A query without rows is predicted
    without the graph.

    Other models compile ``predict`` as a whole, with static shapes.

    Args:
        model: The model.
        dynamic_shapes: The dimensions to mark as dynamic instead of the
            default of ``model``, *e.g.*, ``DynamicShapes(query=(-2,))``
            for the query rows. ``DynamicShapes()`` marks none: the query
            shape is static, and every new shape compiles another graph. A
            dimension that the model cannot keep dynamic fails to compile.
            Not supported for a ``predict`` compiled as a whole.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``False``, so that only the marked
            dimensions are dynamic. ``isolate_recompiles`` defaults to
            ``True``, since all models would otherwise share one recompile
            limit.
    """
    kwargs = {"dynamic": False, "isolate_recompiles": True} | kwargs
    if not model._compilable_models():
        if dynamic_shapes is not None and dynamic_shapes.query:
            raise ValueError(
                f"{model.__class__.__name__!r} does not support dynamic "
                f"shapes in a compiled 'predict'"
            )
        return torch.compile(model.predict, **kwargs)

    shapes = dynamic_shapes or model._default_dynamic_shapes()
    # By model, compiled on its first call: the models of ``model`` may be
    # replaced.
    compiled: dict[Module, Callable[..., Tensor]] = {}

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        # Counted from the last dimension: the models see stacked estimators.
        dims = {dim % x.dim() - x.dim() for dim in shapes.query}
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
    dims: set[int],
    x: Tensor,  # [..., R, C]
    *args: Any,
    **kwargs: Any,
) -> Tensor:
    rows = x.size(-2)
    if rows == 0:
        return module(x, *args, **kwargs)
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
    for dim in dims:
        torch._dynamo.mark_dynamic(x, dim % x.dim())
    out = compiled(x, *args, **kwargs)
    return out[..., :1, :] if repeat else out
