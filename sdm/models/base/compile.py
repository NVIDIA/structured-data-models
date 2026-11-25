# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from sdm import TableTensor

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


def compile_predict(
    model: ICLModel,
    dynamic_dims: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    One graph serves queries and fits of any size along the dynamic
    dimensions, except that a fitted size of zero or one compiles a graph of
    its own. With dynamic dimensions, a query without rows is predicted
    without the graph.

    The lists of categories of a query are static: pass the same lists on
    every call, *e.g.*, a pandas ``Categorical`` with fixed categories. Lists
    with another number of categories or characters compile another graph.

    Args:
        model: The model.
        dynamic_dims: The dimensions of the query to mark as dynamic
            instead of the default of ``model``, *e.g.*, ``(-2,)`` for the
            rows. The tensors that ``fit`` saved follow the query: with
            ``-2``, the context rows are dynamic as well. ``()`` marks none:
            the shapes are static, and every new shape compiles another
            graph. A dimension that the model cannot keep dynamic fails to
            compile.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``False``, so that only the marked
            dimensions are dynamic. ``isolate_recompiles`` defaults to
            ``True``, since all models would otherwise share one recompile
            limit.
    """
    kwargs = {"dynamic": False, "isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    if dynamic_dims is None:
        dynamic_dims = model._default_dynamic_dims()
    # Without dynamic dimensions, none of the steps below are needed.
    if not dynamic_dims:
        return compiled
    for patch in model._dynamic_compile_patches():
        compiled = patch(compiled)

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        rows = x.size(-2)
        cache = model._cache
        if rows == 0 or cache is None:
            return model.predict(x, *args, **kwargs)
        # Counted from the last dimension, *e.g.*, `{-2}` for the rows, as the
        # tensors that `fit` saved declare theirs.
        dims = {dim % x.dim() - x.dim() for dim in dynamic_dims}
        # A graph specializes on sizes of zero and one, so a single row is
        # repeated if the rows are dynamic, unless callbacks see the rows.
        # Without a copy, queries that are different slices of one table, such
        # as `table[40:]` and `table[41:44]`, recompile or fail to compile.
        repeat = rows == 1 and -2 in dims and not kwargs.get("callbacks")
        x = x.clone(memory_format=torch.contiguous_format)
        if repeat:
            x = torch.cat([x, x], dim=-2)
        _mark_dynamic(x, dims)
        cache._mark_dynamic(dims)
        out = compiled(x, *args, **kwargs)
        return out[..., :1, :] if repeat else out

    return predict


def _mark_dynamic(tensor: Tensor, dims: set[int]) -> None:
    for dim in dims:
        dim %= tensor.dim()
        if isinstance(tensor, TableTensor):
            tensor._mark_dynamic(dim)
        else:
            torch._dynamo.mark_dynamic(tensor, dim)
