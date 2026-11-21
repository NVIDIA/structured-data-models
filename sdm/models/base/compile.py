# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.fx.experimental._config as fx_config
from torch import Tensor

from sdm import TableTensor

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel

ModelT = TypeVar("ModelT", bound="ICLModel")


@dataclass(frozen=True)
class DynamicShapes:
    r"""The dimensions that a compiled graph of a model keeps dynamic.

    A model declares them with :func:`dynamic_shapes`, and
    :func:`compile_predict` marks them.
    """

    #: The dimensions of the query. Rows (``-2``) require that the model and
    #: its recipe predict every query row independently of the other rows.
    query: tuple[int, ...] = ()


def dynamic_shapes(**dims: Any) -> Callable[[type[ModelT]], type[ModelT]]:
    r"""Declare the :class:`DynamicShapes` of a model.

    Args:
        **dims: The fields of :class:`DynamicShapes`.
    """

    def decorator(cls: type[ModelT]) -> type[ModelT]:
        cls._dynamic_shapes = DynamicShapes(**dims)
        return cls

    return decorator


def compile_predict(
    model: ICLModel,
    dynamic_shapes: DynamicShapes | None = None,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    One graph serves queries of any size along the :class:`DynamicShapes`
    that ``model`` declares, with any lists of categories. An empty query is
    predicted without the graph.

    Args:
        model: The model.
        dynamic_shapes: The dimensions to mark as dynamic instead of those
            that ``model`` declares, *e.g.*, ``DynamicShapes(query=(-2,))``
            for the query rows. ``DynamicShapes()`` marks none: the query
            shape is static, and every new shape compiles another graph. A
            dimension that the model cannot keep dynamic fails to compile.
            The lists of categories are dynamic in any case.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``False``, so that only the marked
            dimensions are dynamic. ``isolate_recompiles`` defaults to
            ``True``, since all models would otherwise share one recompile
            limit.
    """
    kwargs = {"dynamic": False, "isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    shapes = dynamic_shapes or model._dynamic_shapes
    # With this option PyTorch does not specialize every marked size of zero
    # or one, which lets one graph serve a list of a single category.
    compiled = fx_config.patch(backed_size_oblivious=True)(compiled)  # ty: ignore[unresolved-attribute]

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        rows = x.size(-2)
        if rows == 0:
            return model.predict(x, *args, **kwargs)
        # Operations on the rows still specialize on a single row, so it is
        # repeated if the rows are dynamic, unless callbacks see the rows.
        # Without a copy, queries that are different slices of one table, such
        # as `table[40:]` and `table[41:44]`, recompile or fail to compile. The
        # copy also keeps the marks off the tensors of the caller.
        dims = {dim % x.dim() for dim in shapes.query}
        repeat = (
            rows == 1 and x.dim() - 2 in dims and not kwargs.get("callbacks")
        )
        if repeat:
            x = torch.cat([x, x], dim=-2)
        x = x.clone(memory_format=torch.contiguous_format)
        _mark_dynamic(x, dims)
        out = compiled(x, *args, **kwargs)
        return out[..., :1, :] if repeat else out

    return predict


def _mark_dynamic(tensor: Tensor, dims: set[int]) -> None:
    if isinstance(tensor, TableTensor):
        tensor._mark_dynamic(dims)
        return
    for dim in dims:
        torch._dynamo.mark_dynamic(tensor, dim)
