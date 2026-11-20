# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch
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
    dynamic_shapes: bool | DynamicShapes = True,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    One graph serves queries of any size along the :class:`DynamicShapes`
    that ``model`` declares. Empty queries are predicted without the graph.

    Args:
        model: The model.
        dynamic_shapes: The dimensions to mark as dynamic, which must be
            among those that ``model`` declares. ``True`` marks all of them
            and ``False`` none, for callers whose queries always have the
            same shape.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``isolate_recompiles`` defaults to ``True``, since all models
            would otherwise share one recompile limit.
    """
    kwargs = {"isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    shapes = model._dynamic_shapes
    if isinstance(dynamic_shapes, DynamicShapes):
        if not set(dynamic_shapes.query) <= set(shapes.query):
            raise ValueError(
                f"Expected 'dynamic_shapes' to be among {shapes} "
                f"(got {dynamic_shapes})"
            )
        shapes = dynamic_shapes
    elif not dynamic_shapes:
        shapes = DynamicShapes()
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
        # repeated if the rows are dynamic, unless callbacks see the rows. A
        # contiguous copy has one memory layout for any view.
        repeat = (
            rows == 1 and -2 in shapes.query and not kwargs.get("callbacks")
        )
        x = (
            torch.cat([x, x], dim=-2)
            if repeat
            else x.clone(memory_format=torch.contiguous_format)
        )
        _mark_dynamic(x, shapes.query)
        out = compiled(x, *args, **kwargs)
        return out[..., :1, :] if repeat else out

    return predict


def _mark_dynamic(tensor: Tensor, dims: tuple[int, ...]) -> None:
    for dim in dims:
        dim %= tensor.dim()
        if isinstance(tensor, TableTensor):
            tensor._mark_dynamic(dim)
        else:
            torch._dynamo.mark_dynamic(tensor, dim)
