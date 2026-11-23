# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from sdm import TableTensor

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


@dataclass(frozen=True)
class DynamicShapes:
    r"""The dimensions that a compiled graph of a model keeps dynamic.

    A model returns its default from ``ICLModel._default_dynamic_shapes``,
    and :func:`compile_predict` marks them.
    """

    #: The dimensions of the query. Rows (``-2``) require that the model and
    #: its recipe predict every query row independently of the other rows.
    query: tuple[int, ...] = ()


def compile_predict(
    model: ICLModel,
    dynamic_shapes: DynamicShapes | None = None,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    One graph serves queries of any size along the default
    :class:`DynamicShapes` of ``model``. With dynamic shapes, a query without
    rows is predicted without the graph.

    The lists of categories of a query are static: pass the same lists on
    every call, *e.g.*, a pandas ``Categorical`` with fixed categories. Lists
    with another number of categories or characters compile another graph.

    Args:
        model: The model.
        dynamic_shapes: The dimensions to mark as dynamic instead of the
            default of ``model``, *e.g.*, ``DynamicShapes(query=(-2,))``
            for the query rows. ``DynamicShapes()`` marks none: the query
            shape is static, and every new shape compiles another graph. A
            dimension that the model cannot keep dynamic fails to compile.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``False``, so that only the marked
            dimensions are dynamic. ``isolate_recompiles`` defaults to
            ``True``, since all models would otherwise share one recompile
            limit.
    """
    kwargs = {"dynamic": False, "isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    shapes = dynamic_shapes or model._default_dynamic_shapes()
    # Without dynamic shapes, none of the steps below are needed.
    if shapes == DynamicShapes():
        return compiled

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        rows = x.size(-2)
        if rows == 0:
            return model.predict(x, *args, **kwargs)
        # A graph specializes on sizes of zero and one, so a single row is
        # repeated if the rows are dynamic, unless callbacks see the rows.
        # Without a copy, queries that are different slices of one table, such
        # as `table[40:]` and `table[41:44]`, recompile or fail to compile.
        dims = {dim % x.dim() for dim in shapes.query}
        repeat = (
            rows == 1 and x.dim() - 2 in dims and not kwargs.get("callbacks")
        )
        x = (
            torch.cat([x, x], dim=-2)
            if repeat
            else x.clone(memory_format=torch.contiguous_format)
        )
        _mark_dynamic(x, dims)
        out = compiled(x, *args, **kwargs)
        return out[..., :1, :] if repeat else out

    return predict


def _mark_dynamic(tensor: Tensor, dims: set[int]) -> None:
    for dim in dims:
        if isinstance(tensor, TableTensor):
            tensor._mark_dynamic(dim)
        else:
            torch._dynamo.mark_dynamic(tensor, dim)
