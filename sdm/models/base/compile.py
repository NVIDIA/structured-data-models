# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models.base import Callback
from sdm.tensor.table import TableSchema

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


def compile_predict(
    model: ICLModel,
    dynamic_dims: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile the model of ``model.predict`` using :func:`torch.compile`.

    The fitted recipe transforms the query before the graph, which starts at
    the transformed query. With a recipe that encodes the categories, such as
    the default one of :class:`~sdm.models.KumoTabular`, queries with any
    lists of categories share one graph.

    One graph serves queries and fits of any size along the dynamic
    dimensions, except that a fitted size of zero or one compiles a graph of
    its own. A query without rows is predicted without the graph.

    The columns (``-1``) are dynamic for a transformed query of only numerical
    columns, as the default recipe of :class:`~sdm.models.KumoTabular` returns
    for numerical and categorical ones. It runs without column names, which
    the graph would otherwise hold as constants. Other transformed queries,
    and queries with callbacks, keep their names and static columns.

    Args:
        model: The model.
        dynamic_dims: The dimensions of the transformed query to mark as
            dynamic instead of the default of ``model``, *e.g.*, ``(-2,)``
            for the rows, which requires that the model and its recipe
            predict every query row independently of the other rows. The
            tensors that ``fit`` saved follow the query: with ``-2``, the
            context rows are dynamic as well. ``()`` marks none: the shapes
            are static, and every new shape compiles another graph. A
            dimension that the model cannot keep dynamic fails to compile.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``False``, so that only the marked
            dimensions are dynamic. ``isolate_recompiles`` defaults to
            ``True``, since all models would otherwise share one recompile
            limit.
    """
    kwargs = {"dynamic": False, "isolate_recompiles": True} | kwargs
    compiled: Callable[..., TableTensor] = torch.compile(
        model._predict_queries,
        **kwargs,
    )
    if dynamic_dims is None:
        dynamic_dims = model._default_dynamic_dims()
    if dynamic_dims:
        for patch in model._dynamic_compile_patches():
            compiled = patch(compiled)

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
        # A graph specializes on sizes of zero and one, so a single row is
        # repeated if the rows are dynamic, unless callbacks see the rows.
        # Without a copy, queries that are different slices of one table, such
        # as `table[40:]` and `table[41:44]`, recompile or fail to compile.
        repeat = rows == 1 and -2 in dims and not callbacks
        x = (
            torch.cat([x, x], dim=-2)
            if repeat
            else x.clone(memory_format=torch.contiguous_format)
        )
        callbacks = () if callbacks is None else callbacks
        queries, requires_grad = model._transform_query(
            x,
            related_tables,
            callbacks=callbacks,
        )
        if (
            -1 in dims
            and not callbacks
            and all(
                query.x.size(-1) == query.x.numerical.size(-1)
                for query in queries
            )
        ):
            # Numerical columns run without names, which a graph would hold
            # as constants, so they are checked before the graph.
            schemas = [
                schema
                for i in range(cast(int, cache["num_batches"]))
                for schema in cast(
                    tuple[TableSchema, ...],
                    cast(Cache, cache[i])["x_schemas"],
                )
            ]
            for query, schema in zip(queries, schemas, strict=True):
                model._validate_query(
                    x_context=schema,
                    x_query=query.x,
                    related_context_tables=None,
                    related_query_tables=None,
                )
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
            # Other columns, such as text ones, keep their names, and so do
            # queries with callbacks, which may read them.
            dims.discard(-1)
        cache._mark_dynamic(dims)
        for query in queries:
            for dim in dims:
                query.x._mark_dynamic(dim % x.dim())
        out = compiled(
            queries,
            device=x.device,
            callbacks=callbacks,
            requires_grad=requires_grad,
        )
        return out[..., :1, :] if repeat else out

    return predict
