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
from sdm._memory import single_chunk
from sdm.models.base.callback import Callback

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


@dataclass(frozen=True)
class DynamicShapes:
    r"""The dimensions one compiled graph of a model keeps dynamic.

    A model declares them as :attr:`~sdm.models.ICLModel._dynamic_shapes`.
    """

    #: The dimensions of the query. Rows (``-2``) require that the model and
    #: its recipe predict every query row independently of the other rows.
    query: tuple[int, ...] = ()
    # TODO: Declare context rows and columns as the dimensions of tensors in
    # the fitted cache, and mark them in `compile_predict`.


def compile_predict(
    model: ICLModel,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    With ``dynamic=False``, the graph keeps the :class:`DynamicShapes` the
    model declares dynamic. It predicts a query in a single chunk, so its
    memory grows with the query: split large queries before calling it.
    Empty queries and queries with related tables or callbacks are predicted
    eagerly.

    Args:
        model: The model.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``isolate_recompiles`` defaults to ``True``, since all models
            would otherwise share one recompile limit.
    """
    kwargs = {"isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    dims = model._dynamic_shapes.query
    if kwargs.get("dynamic") is not False or not dims:
        return compiled

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        rows = x.size(-2)
        if rows == 0 or related_tables is not None or callbacks:
            return model.predict(x, related_tables, callbacks=callbacks)
        # A contiguous copy has the same layout for any view of a query. A
        # single row would specialize, so it is repeated.
        x = (
            torch.cat([x, x], dim=-2)
            if rows == 1
            else x.clone(memory_format=torch.contiguous_format)
        )
        _mark_dynamic(x, dims)
        with single_chunk():
            out = compiled(x)
        return out[..., :1, :] if rows == 1 else out

    return predict


def _mark_dynamic(tensor: Tensor, dims: tuple[int, ...]) -> None:
    for dim in dims:
        dim %= tensor.dim()
        if isinstance(tensor, TableTensor):
            tensor._mark_dynamic(dim)
        else:
            torch._dynamo.mark_dynamic(tensor, dim)
