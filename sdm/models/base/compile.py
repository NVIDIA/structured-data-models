# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.cache import Cache
from sdm.processing import Processor
from sdm.processing.execution import RecipeExecution

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
    #: The dimensions of cached tensors and of cached keys and values, by the
    #: first dotted component of their cache keys. With the context rows, a
    #: model that is fit again on another number of them reuses the graph, if
    #: the fits share columns, categories and classes in the same order (the
    #: same ``generator`` seed).
    cache: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    #: The column dimensions of cached tensors and of cached keys and values,
    #: by the first dotted component of their cache keys. With them, a model
    #: that is fit again on a numerical table with another number of columns
    #: reuses the graph. Tables with more columns than the recipe keeps share
    #: a second graph. Such tables run without their column names, which a
    #: graph holds as constants, so the recipe must not read the names of
    #: numerical columns.
    columns: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


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
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    One graph serves queries and fits of any size along the
    :class:`DynamicShapes` that ``model`` declares. Empty queries and fits
    with fewer than two entries along a dimension are predicted without the
    graph.

    Args:
        model: The model.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``isolate_recompiles`` defaults to ``True``, since all models
            would otherwise share one recompile limit.
    """
    kwargs = {"isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    # PyTorch 2.14.1 inlines a nested compile region that is used once. The
    # regions that copy the cache to the device have to stay nested: inlined,
    # they copy tensors of dynamic size on the transfer stream within the
    # graph, and the generated code fails on CUDA with "UnboundLocalError:
    # cannot access local variable 'default_stream'". The setting holds for
    # every nested region of a call.
    config = {"inline_single_use_invoke_subgraph": False}
    shapes = model._dynamic_shapes
    if shapes.columns:
        # Marks on the fitted state of processors apply only with this option.
        config["force_parameter_static_shapes"] = False
    compiled = torch._dynamo.config.patch(config)(compiled)
    if shapes == DynamicShapes():
        return compiled

    mark_cache = _once_per_fit(lambda cache: cache._mark_dynamic(shapes.cache))

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        rows = x.size(-2)
        cache = model._cache
        if rows == 0 or cache is None or not mark_cache(cache):
            return model.predict(x, *args, **kwargs)
        if isinstance(x, TableTensor) and x.categorical.size(-1) > 0:
            # A graph specializes on the size of every list of categories.
            execution = cast(RecipeExecution, cache["recipe_execution"])
            x = execution.align_categories(x)
        # Column names are constants of a graph, so numerical features that
        # are named as in `fit` run without them, with dynamic columns.
        unnamed = bool(shapes.columns) and _named_as_fit(x, cache)
        if unnamed:
            if not _mark_columns(cache, shapes.columns):
                return model.predict(x, *args, **kwargs)
            x = TableTensor(
                columns={Stype.numerical: None},
                numerical=x.numerical if isinstance(x, TableTensor) else x,
            )
        # A graph specializes on sizes of zero and one, so a single row is
        # repeated if the rows are dynamic, unless callbacks see the rows. A
        # contiguous copy has one memory layout for any view, and shares no
        # tensor with the fitted categories.
        repeat = (
            rows == 1 and -2 in shapes.query and not kwargs.get("callbacks")
        )
        if repeat:
            x = torch.cat([x, x], dim=-2)
        x = x.clone(memory_format=torch.contiguous_format)
        _mark_dynamic(x, shapes.query)
        if unnamed:
            # A strict mark would reach the empty blocks of the other
            # semantic types, and a graph specializes on their size of zero.
            torch._dynamo.maybe_mark_dynamic(x, x.dim() - 1)
        out = compiled(x, *args, **kwargs)
        return out[..., :1, :] if repeat else out

    return predict


def _once_per_fit(mark: Callable[[Cache], bool]) -> Callable[[Cache], bool]:
    # Marks stay on the tensors of a fit, and a fit replaces the cache of a
    # model, so `mark` runs once per cache.
    memo: tuple[weakref.ref[Cache], bool] | None = None

    def mark_once(cache: Cache) -> bool:
        nonlocal memo
        entry = memo
        if entry is None or entry[0]() is not cache:
            memo = entry = (weakref.ref(cache), mark(cache))
        return entry[1]

    return mark_once


def _mark_dynamic(tensor: Tensor, dims: tuple[int, ...]) -> None:
    for dim in dims:
        dim %= tensor.dim()
        if isinstance(tensor, TableTensor):
            tensor._mark_dynamic(dim)
        else:
            torch._dynamo.mark_dynamic(tensor, dim)


def _named_as_fit(x: Tensor | TableTensor, cache: Cache) -> bool:
    # Whether `x` holds numerical features only, named as in `fit`. A tensor
    # has no names.
    columns = cache.get("x_columns")
    if not isinstance(x, TableTensor):
        return columns is None
    return x.active_stypes == {Stype.numerical} and x.columns == columns


def _mark_columns(cache: Cache, dims: Mapping[str, tuple[int, ...]]) -> bool:
    # Marks the column dimensions of the fitted state. Returns `False` if the
    # model holds fewer than two columns. A recipe may keep a fixed number of
    # the columns of wider tables, so a graph may specialize on these marks.
    dims = {"categorical_mask": (-1,), **dims}
    if not cache._mark_dynamic(dims, strict=False):
        return False
    recipe = cast(RecipeExecution, cache["recipe_execution"]).recipe
    for processor in recipe.features.modules():
        if isinstance(processor, Processor) and processor.is_fitted:
            for tensor, dim in processor._column_dims():
                if tensor.size(dim) >= 2:
                    torch._dynamo.maybe_mark_dynamic(
                        tensor, dim % tensor.dim()
                    )
    return True
