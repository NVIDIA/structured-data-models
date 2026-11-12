# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.models.base.callback import Callback

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


@dataclass(frozen=True)
class DynamicShapes:
    r"""Dimensions allowed to vary during compiled prediction.

    Args:
        query: Dynamic query dimensions.
        cache: Reserved for fitted tensor declarations: pairs of cache paths
            and dynamic dimensions. Paths name mapping keys, sequence indices,
            or attributes, starting at the fitted cache root. No cache marking
            or reuse across refits is implemented yet.
    """

    query: tuple[int, ...] = ()
    cache: tuple[tuple[tuple[str | int, ...], tuple[int, ...]], ...] = ()


def compile_predict(
    model: ICLModel,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    With ``dynamic=False``, mark the query dimensions the model declares.
    Chunk guards determine when queries must be split without recompiling.
    Rejected batches are halved; only batches of at most two rows may
    compile, including after a refit. Empty queries and queries with related
    tables or callbacks run eagerly. Singleton rows are repeated to avoid
    PyTorch's zero/one specialization; outputs are sliced back. Predictions
    must be independent across query rows.

    Args:
        model: The model.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``isolate_recompiles`` defaults to ``True``, since all models
            would otherwise share one recompile limit.
    """
    kwargs = {"isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    shapes = model._dynamic_shapes
    if kwargs.get("dynamic") is not False or not shapes.query:
        return compiled

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        rows = x.size(-2)
        if rows == 0 or callbacks is not None or related_tables is not None:
            return model.predict(x, related_tables, callbacks=callbacks)
        # Python chunk loops capture fixed iteration counts. Crossing their
        # guards needs another graph, so larger queries must split outside it.
        # The guards alone decide the size; no row capacity is configured.
        outputs = []
        start, size = 0, rows
        while start < rows:
            query = _prepare_query(
                x=x[..., start : start + size, :], dims=shapes.query
            )
            try:
                with torch.compiler.set_stance("fail_on_recompile"):
                    out = compiled(query, related_tables, callbacks=callbacks)
            except RuntimeError as error:
                if not _is_guard_miss(error):
                    raise
                if size > 2:
                    size = (size + 1) // 2
                    continue
                out = compiled(query, related_tables, callbacks=callbacks)
                if size < rows - start:
                    size = rows - start
                    continue
            outputs.append(out[..., :size, :])
            start += size
            size = min(size, rows - start)
        return torch.cat(outputs, dim=-2) if len(outputs) > 1 else outputs[0]

    return predict


def _prepare_query(x: Tensor, dims: tuple[int, ...]) -> Tensor:
    # One layout for views and singleton padding, including text storage.
    index = torch.arange(max(x.size(-2), 2), device=x.device)
    x = x.index_select(-2, index.clamp_(max=x.size(-2) - 1))
    if isinstance(x, TableTensor):
        x._mark_dynamic(dims)
        return x
    for dim in dims:
        torch._dynamo.mark_dynamic(x, dim % x.dim())
    return x


def _is_guard_miss(error: RuntimeError) -> bool:
    # PyTorch 2.14.1 eval_frame.fail_callback raises an untyped RuntimeError.
    return str(error).startswith(
        "Detected recompile when torch.compile stance is 'fail_on_recompile'."
    )
