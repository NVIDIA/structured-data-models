# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

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

    #: The dimensions of the transformed query. Rows (``-2``) require that the
    #: model and its recipe predict every query row independently of the other
    #: rows.
    query: tuple[int, ...] = ()


def compile_predict(
    model: ICLModel,
    dynamic_shapes: DynamicShapes | None = None,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile the model of ``model.predict`` using :func:`torch.compile`.

    The fitted recipe transforms the query before the graph, which starts at
    the transformed query. With a recipe that encodes the categories, such as
    the default one of :class:`~sdm.models.KumoTabular`, queries with any
    lists of categories share one graph.

    One graph serves queries of any size along the default
    :class:`DynamicShapes` of ``model``. A query without rows is predicted
    without the graph.

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
    compiled = torch.compile(model._predict_queries, **kwargs)
    shapes = dynamic_shapes or model._default_dynamic_shapes()

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        rows = x.size(-2)
        if rows == 0:
            return model.predict(x, related_tables, callbacks=callbacks)
        # A graph specializes on sizes of zero and one, so a single row is
        # repeated if the rows are dynamic, unless callbacks see the rows.
        # Without a copy, queries that are different slices of one table, such
        # as `table[40:]` and `table[41:44]`, recompile or fail to compile.
        dims = {dim % x.dim() for dim in shapes.query}
        repeat = rows == 1 and x.dim() - 2 in dims and not callbacks
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
        for query in queries:
            for dim in dims:
                query.x._mark_dynamic(dim)
        out = compiled(
            queries,
            device=x.device,
            callbacks=callbacks,
            requires_grad=requires_grad,
        )
        return out[..., :1, :] if repeat else out

    return predict
