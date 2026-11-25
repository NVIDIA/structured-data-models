# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from sdm import TableTensor

if TYPE_CHECKING:
    from sdm import RelatedTables
    from sdm.models.base import Callback
    from sdm.models.base.icl import ICLModel


def compile_predict(
    model: ICLModel,
    dynamic_dims: tuple[int, ...] | None = None,
    compile_preprocessing: bool = True,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    One graph serves queries and fits of any size along the dynamic
    dimensions, and fits with any categories, except that a fitted size of
    zero or one compiles a graph of its own. With dynamic dimensions, a
    query without rows is predicted without the graph.

    The numerical columns (``-1``) are dynamic only after a fit with the
    default recipe whose processors run without column names, as those of
    :class:`~sdm.models.KumoTabular` do. The query then runs its numerical
    columns without their names, which a graph would hold as constants, and a
    query named otherwise or with callbacks is predicted without the graph.
    Other columns, such as categorical ones, keep their names: a fit on other
    ones compiles another graph. If ``compile_preprocessing`` is ``False``,
    the transformed query runs without names if it holds only numerical
    columns, as after the default recipe on numerical and categorical ones,
    and with static columns otherwise. After other fits, all columns are
    static.

    The lists of categories of a query are static: pass the same lists on
    every call, *e.g.*, a pandas ``Categorical`` with fixed categories. Lists
    with another number of categories or characters compile another graph,
    unless ``compile_preprocessing`` is ``False``.

    Args:
        model: The model.
        dynamic_dims: The dimensions of the query to mark as dynamic
            instead of the default of ``model``, *e.g.*, ``(-2,)`` for the
            rows. The tensors that ``fit`` saved follow the query: with
            ``-2``, the context rows are dynamic as well. ``()`` marks none:
            the shapes are static, and every new shape compiles another
            graph. A dimension that the model cannot keep dynamic fails to
            compile.
        compile_preprocessing: Whether the graph includes the transform of
            the query by the fitted recipe. If ``False``, the recipe
            transforms the query before the graph, and ``dynamic_dims``
            apply to the transformed query. With a recipe that encodes the
            categories, such as the default one, queries with other lists of
            categories then share one graph.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``False``, so that only the marked
            dimensions are dynamic. ``isolate_recompiles`` defaults to
            ``True``, since all models would otherwise share one recompile
            limit.
    """
    kwargs = {"dynamic": False, "isolate_recompiles": True} | kwargs
    compiled: Callable[..., TableTensor] = torch.compile(
        model.predict if compile_preprocessing else model._predict_queries,
        **kwargs,
    )
    if dynamic_dims is None:
        dynamic_dims = model._default_dynamic_dims()
    # Without dynamic dimensions, compiled preprocessing needs none of the
    # steps below.
    if compile_preprocessing and not dynamic_dims:
        return compiled

    def predict_unnamed(
        x: Tensor,
        rest: Tensor | None,
        related_tables: RelatedTables | None,
    ) -> TableTensor:
        # Inside the graph, the empty blocks of the table keep zero columns.
        x = TableTensor(columns={"numerical": None}, numerical=x)
        if rest is not None:
            x = cast(TableTensor, torch.cat([x, rest], dim=-1))
        return model.predict(x, related_tables)

    compiled_unnamed = torch.compile(predict_unnamed, **kwargs)
    for patch in model._dynamic_compile_patches():
        compiled = patch(compiled)
        compiled_unnamed = patch(compiled_unnamed)

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        rows = x.size(-2)
        cache = model._cache
        if rows == 0 or cache is None:
            return model.predict(x, related_tables, callbacks=callbacks)
        # Counted from the last dimension, *e.g.*, `{-2}` for the rows, as the
        # tensors that `fit` saved declare theirs.
        dims = {dim % x.dim() - x.dim() for dim in dynamic_dims}
        columns = cast(
            "tuple[str, ...] | None", cache.get("numerical_columns")
        )
        split = None
        if columns is None:
            # The fit keeps its column names, which a graph holds as constants.
            dims.discard(-1)
        elif -1 in dims and callbacks:
            # Callbacks may read the column names.
            return model.predict(x, related_tables, callbacks=callbacks)
        elif -1 in dims and compile_preprocessing:
            # A query named as in `fit` runs its numerical columns without
            # names.
            split = _numerical(x, columns)
            if split is None:
                return model.predict(x, related_tables)
            x = split[0]
        # A graph specializes on sizes of zero and one, so a single row is
        # repeated if the rows are dynamic, unless callbacks see the rows.
        # Without a copy, queries that are different slices of one table, such
        # as `table[40:]` and `table[41:44]`, recompile or fail to compile.
        # Static dimensions need no copy, and leave the cache unmarked for
        # later predictors.
        repeat = rows == 1 and -2 in dims and not callbacks
        if dims:
            x = x.clone(memory_format=torch.contiguous_format)
            cache._mark_dynamic(dims)
        if repeat:
            x = torch.cat([x, x], dim=-2)
        if split is not None:
            _mark_dynamic(x, dims)
            rest = split[1]
            if rest is not None:
                # The other columns keep their names, and so their number.
                rest = rest.clone(memory_format=torch.contiguous_format)
                if repeat:
                    rest = torch.cat([rest, rest], dim=-2)
                _mark_dynamic(rest, dims - {-1})
            out = compiled_unnamed(x, rest, related_tables)
        elif compile_preprocessing:
            _mark_dynamic(x, dims)
            out = compiled(x, related_tables, callbacks=callbacks)
        else:
            callbacks = () if callbacks is None else callbacks
            queries, requires_grad = model._transform_query(
                x,
                related_tables,
                callbacks=callbacks,
            )
            if -1 in dims and all(
                query.x.size(-1) == query.x.numerical.size(-1)
                for query in queries
            ):
                # Transformed queries of only numerical columns run without
                # names.
                queries = tuple(
                    query._replace(
                        x=TableTensor(
                            columns={"numerical": None},
                            numerical=query.x.numerical,
                        )
                    )
                    for query in queries
                )
            else:
                # Other columns, such as text ones, keep their names.
                dims.discard(-1)
            for query in queries:
                _mark_dynamic(query.x, dims)
            out = compiled(
                queries,
                device=x.device,
                callbacks=callbacks,
                requires_grad=requires_grad,
            )
        return out[..., :1, :] if repeat else out

    return predict


def _numerical(
    x: Tensor, columns: tuple[str, ...]
) -> tuple[Tensor, TableTensor | None] | None:
    # The numerical features of `x` if they are these columns, and its other
    # columns, if any. A tensor has the names of `TableTensor.from_tensor`.
    if isinstance(x, TableTensor):
        if x.numerical_columns != columns:
            return None
        rest = (
            x.drop_stypes("numerical") if x.size(-1) > len(columns) else None
        )
        return x.numerical, rest
    names = tuple(map(str, range(x.size(-1))))
    return (x, None) if x.is_floating_point() and names == columns else None


def _mark_dynamic(tensor: Tensor, dims: set[int]) -> None:
    for dim in dims:
        dim %= tensor.dim()
        if isinstance(tensor, TableTensor):
            tensor._mark_dynamic(dim)
        else:
            torch._dynamo.mark_dynamic(tensor, dim)
