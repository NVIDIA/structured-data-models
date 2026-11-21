# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, TableTensor

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel

ModelT = TypeVar("ModelT", bound="ICLModel")


@dataclass(frozen=True)
class DynamicShapes:
    r"""The dimensions that a compiled graph of a model keeps dynamic.

    A model declares them with :func:`dynamic_shapes`, and
    :func:`compile_predict` marks them.
    """

    #: Whether the number of rows is dynamic: the query rows, and the context
    #: rows of the fitted state that declares them. Dynamic rows require that
    #: the model and its recipe predict every query row independently of the
    #: other rows.
    rows: bool = False


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

    One graph serves queries and fits of any size along the
    :class:`DynamicShapes` that ``model`` declares, and any categories,
    except that a fitted size of zero or one compiles a graph of its own.
    Uncompiled ``model.predict`` serves a query without rows, and a query
    with a categorical column that has no categories, neither in the query
    nor in the fitted table.

    Args:
        model: The model.
        dynamic_shapes: The dimensions to mark as dynamic instead of those
            that ``model`` declares, *e.g.*, ``DynamicShapes(rows=True)``
            for the query rows and the context rows of the fitted state.
            ``DynamicShapes()`` marks none: the shapes are static, and every
            new shape compiles another graph. A dimension that the model
            cannot keep dynamic fails to compile. The lists of categories
            are dynamic in any case.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``False``, so that only the marked
            dimensions are dynamic. ``isolate_recompiles`` defaults to
            ``True``, since all models would otherwise share one recompile
            limit.
    """
    kwargs = {"dynamic": False, "isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    # Dynamo ignores marks on buffers otherwise, and PyTorch 2.14.1 fails on
    # CUDA when the copies of the cache to the device join the graph.
    compiled = torch._dynamo.config.patch(
        force_parameter_static_shapes=False,
        inline_single_use_invoke_subgraph=False,
    )(compiled)
    # Inductor would order the kernels for the sizes of the first compile,
    # which costs time and memory after a fit of another size.
    compiled = torch._inductor.config.patch(
        reorder_for_peak_memory=False,
    )(compiled)
    shapes = dynamic_shapes or model._dynamic_shapes
    names = {name for name, dynamic in asdict(shapes).items() if dynamic}

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        rows = x.size(-2)
        dims = {x.dim() - 2} if shapes.rows else set()
        # A graph specializes on sizes of zero and one, so a single row is
        # repeated if the rows are dynamic, unless callbacks see the rows.
        repeat = rows == 1 and shapes.rows and not kwargs.get("callbacks")
        query = x
        if rows > 0 and isinstance(x, TableTensor):
            query = _categories(model, x, copy=repeat)
        cache = model._cache
        if rows == 0 or query is None or cache is None:
            return model.predict(x, *args, **kwargs)
        cache._mark_dynamic(names)
        # Without a copy, queries that are different slices of one table, such
        # as `table[40:]` and `table[41:44]`, recompile or fail to compile.
        query = (
            torch.cat([query, query], dim=-2)
            if repeat
            else query.clone(memory_format=torch.contiguous_format)
        )
        _mark_dynamic(query, dims)
        out = compiled(query, *args, **kwargs)
        return out[..., :1, :] if repeat else out

    return predict


def _categories(
    model: ICLModel,
    x: TableTensor,
    copy: bool,
) -> TableTensor | None:
    # Like the rows, a list of categories needs two or more entries. A column
    # of missing values takes the fitted list, as no row points into it, and
    # a single category is repeated. `torch.cat` would share the other lists
    # with the caller, so they are copied before they get marked.
    categories = x.categorical.categories
    if not copy and all(category.size(0) > 1 for category in categories):
        return x
    fitted = cast(
        "Mapping[str, Tensor]",
        (model._cache or {}).get("categories", {}),
    )
    new_categories = []
    for name, category in zip(x.categorical_columns, categories, strict=True):
        if category.size(0) == 0:
            category = fitted.get(name, category)
        if category.size(0) == 0:
            return None
        if category.size(0) == 1:
            category = torch.cat([category, category])
        elif copy:
            category = category.clone()
        new_categories.append(category)
    return x.replace_blocks(
        categorical=CategoricalTensor(
            code=x.categorical.code,
            categories=new_categories,
        ),
    )


def _mark_dynamic(tensor: Tensor, dims: set[int]) -> None:
    if isinstance(tensor, TableTensor):
        tensor._mark_dynamic(dims)
        return
    for dim in dims:
        torch._dynamo.mark_dynamic(tensor, dim)
