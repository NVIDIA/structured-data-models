# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models.base.callback import Callback

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel


@dataclass(frozen=True)
class DynamicShapes:
    r"""The sizes one compiled graph of a model serves; ``None`` is static.

    A model declares them per batch of estimators in
    :meth:`~sdm.models.ICLModel._dynamic_shapes`.
    """

    #: The query rows the model predicts in one pass, i.e. without splitting
    #: its work by the number of rows. Requires that the model and its
    #: recipe predict query rows independently of each other.
    query_rows: int | None = None
    # TODO: Declare context rows and columns here as dimensions of tensors in
    # the fitted cache, and mark them in `compile_predict`.


def compile_predict(
    model: ICLModel,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    With ``dynamic=False``, one graph serves the :class:`DynamicShapes` the
    fitted model declares: queries are split into chunks of at most
    ``query_rows`` rows, whose rows are marked as dynamic. Empty queries,
    queries with related tables or callbacks, and models without a pass of at
    least two rows are predicted eagerly.

    Args:
        model: The model.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``isolate_recompiles`` defaults to ``True``, since all models
            would otherwise share one recompile limit.
    """
    kwargs = {"isolate_recompiles": True} | kwargs
    compiled = torch.compile(model.predict, **kwargs)
    if kwargs.get("dynamic") is not False:
        return compiled

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        max_rows = _max_query_rows(model, x.device)
        if max_rows is None:
            return compiled(x, related_tables, callbacks=callbacks)
        if (
            max_rows < 2
            or x.size(-2) == 0
            or related_tables is not None
            or callbacks
        ):
            return model.predict(x, related_tables, callbacks=callbacks)

        outs: list[Tensor] = []
        for start in range(0, x.size(-2), max_rows):
            end = min(start + max_rows, x.size(-2))
            # Gather every chunk, so all chunks share one layout, and repeat
            # the row of single-row chunks: a single row would specialize.
            index = torch.arange(start, max(end, start + 2), device=x.device)
            chunk = x.index_select(-2, index.clamp_(max=end - 1))
            _mark_rows(chunk, max_rows)
            outs.append(compiled(chunk)[..., : end - start, :])
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)
        return cast(TableTensor, out)

    return predict


def _max_query_rows(model: ICLModel, device: torch.device) -> int | None:
    # The query rows every batch of estimators predicts in one pass, or
    # `None` if the model is not fitted or does not declare them.
    if model._cache is None:
        return None
    rows = [
        model._dynamic_shapes(cast(Cache, model._cache[i]), device).query_rows
        for i in range(cast(int, model._cache["num_batches"]))
    ]
    return None if None in rows else min(cast(list[int], rows))


def _mark_rows(x: Tensor, max_rows: int) -> None:
    tensors = [x]
    if isinstance(x, TableTensor):
        # Marks reach inner tensors of the same rank only; the text layout
        # has one more dimension.
        text = x.text
        tensors.append(text._layout)
        # The text storage varies in size with the rows.
        for storage in (text._data, text._offset, text._valid):
            if storage is not None:
                torch._dynamo.maybe_mark_dynamic(storage, 0)
    for tensor in tensors:
        torch._dynamo.mark_dynamic(tensor, x.dim() - 2, min=2, max=max_rows)
