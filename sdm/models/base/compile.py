# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of compiled model functions.

:class:`DynamicShapes`, :func:`_compile` and :func:`_mark` serve any compiled
function of a model; :func:`compile_predict` adds what only a query needs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models.base.callback import Callback
from sdm.processing import Processor
from sdm.processing.execution import RecipeExecution

if TYPE_CHECKING:
    from sdm.models.base.icl import ICLModel

# PyTorch 2.14 requires an upper bound for dynamic sizes with a lower bound.
_MAX_SIZE = 2**31 - 1


#: A dimension of a fitted tensor.
Dim = tuple[Tensor, int]


@dataclass(frozen=True)
class DynamicShapes:
    r"""The sizes one compiled graph of a model serves; unset is static.

    A model declares them per batch of estimators in
    :meth:`~sdm.models.ICLModel._dynamic_shapes`. Sizes below two always
    compile their own graph.
    """

    #: The query rows the model predicts in one pass, i.e. without splitting
    #: its work by the number of rows. Requires that the model and its
    #: recipe predict query rows independently of each other.
    query_rows: int | None = None
    #: The columns of numerical tables the model predicts in one pass. Such
    #: queries run with unnamed columns, which the recipe must not read.
    query_columns: int | None = None
    #: The context-row dimensions of fitted tensors.
    context_row_dims: Sequence[Dim] = ()
    #: The column dimensions of fitted tensors of the model; those of the
    #: recipe are known.
    column_dims: Sequence[Dim] = ()


def compile_predict(
    model: ICLModel,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    With ``dynamic=False``, one graph serves the :class:`DynamicShapes` the
    fitted model declares, also across fits: queries are split into chunks of
    at most ``query_rows`` rows, whose rows are marked as dynamic, as are the
    declared dimensions of fitted tensors. Numerical tables of at most
    ``query_columns`` columns that are named as in :meth:`fit` run unnamed
    with dynamic columns. Empty queries, queries with related tables or
    callbacks, and models without a pass of at least two rows are predicted
    eagerly.

    Args:
        model: The model.
        **kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``isolate_recompiles`` defaults to ``True``, since all models
            would otherwise share one recompile limit.
    """
    compiled = _compile(model.predict, **kwargs)
    if kwargs.get("dynamic") is not False:
        return compiled

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        shapes = _dynamic_shapes(model, x.device)
        max_rows = _min_or_none([shape.query_rows for shape in shapes])
        if max_rows is None:
            return compiled(x, related_tables, callbacks=callbacks)
        if (
            max_rows < 2
            or x.size(-2) == 0
            or related_tables is not None
            or callbacks
        ):
            return model.predict(x, related_tables, callbacks=callbacks)

        _mark(
            [dim for shape in shapes for dim in shape.context_row_dims],
            _MAX_SIZE,
        )
        max_columns = _min_or_none([shape.query_columns for shape in shapes])
        if (
            max_columns is not None
            and 2 <= x.size(-1) <= max_columns
            and isinstance(x, TableTensor)
            and x.active_stypes == {Stype.numerical}
            and x.columns == cast(Cache, model._cache)["x_columns"]
        ):
            # Column names are constants of the graph, so the number of
            # columns varies only without them.
            x = TableTensor(
                columns={Stype.numerical: None},
                numerical=x.numerical,
            )
            _mark(
                [
                    *_recipe_column_dims(model),
                    *(dim for shape in shapes for dim in shape.column_dims),
                ],
                max_columns,
            )

        outs: list[Tensor] = []
        for start in range(0, x.size(-2), max_rows):
            end = min(start + max_rows, x.size(-2))
            # Gather every chunk, so all chunks share one layout, and repeat
            # the row of single-row chunks: a single row would specialize.
            index = torch.arange(start, max(end, start + 2), device=x.device)
            chunk = x.index_select(-2, index.clamp_(max=end - 1))
            _mark_query(chunk, max_rows)
            outs.append(compiled(chunk)[..., : end - start, :])
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)
        return cast(TableTensor, out)

    return predict


def _compile(
    function: Callable[..., Any], **kwargs: Any
) -> Callable[..., Any]:
    # Compile a model function whose dynamic sizes are marked per call.
    kwargs = {"isolate_recompiles": True} | kwargs
    compiled = torch.compile(function, **kwargs)
    if kwargs.get("dynamic") is not False:
        return compiled
    return torch._dynamo.config.patch(
        # Marks on fitted buffers apply only with this option.
        force_parameter_static_shapes=False,
        # Inductor miscompiles copies of fitted tensors of dynamic size
        # across CUDA streams otherwise.
        inline_single_use_invoke_subgraph=False,
    )(compiled)


def _dynamic_shapes(
    model: ICLModel,
    device: torch.device,
) -> list[DynamicShapes]:
    # The declaration of every batch of estimators of a fitted model.
    if model._cache is None:
        return []
    return [
        model._dynamic_shapes(cast(Cache, model._cache[i]), device)
        for i in range(cast(int, model._cache["num_batches"]))
    ]


def _min_or_none(sizes: list[int | None]) -> int | None:
    # The size every batch of estimators serves, or `None` if any is static.
    if not sizes or None in sizes:
        return None
    return min(cast(list[int], sizes))


def _mark(dims: Sequence[Dim], max_size: int) -> None:
    # Marks do not survive copies of the fitted state, so mark before every
    # call. Sizes 0 and 1 always specialize.
    for tensor, dim in dims:
        if 2 <= tensor.size(dim) <= max_size:
            torch._dynamo.mark_dynamic(
                tensor, dim % tensor.dim(), min=2, max=max_size
            )


def _recipe_column_dims(model: ICLModel) -> Iterator[Dim]:
    # The column dimensions of what the recipe and `ICLModel.fit` cache.
    cache = cast(Cache, model._cache)
    recipe = cast(RecipeExecution, cache["recipe_execution"]).recipe
    for processor in recipe.features.modules():
        if isinstance(processor, Processor) and processor.is_fitted:
            yield from processor._column_dims()
    for i in range(cast(int, cache["num_batches"])):
        yield cast(Tensor, cast(Cache, cache[i])["categorical_mask"]), -1


def _mark_query(x: Tensor, max_rows: int) -> None:
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
        if x.numerical_columns is None:
            # Strict marks would reach the empty blocks, too.
            torch._dynamo.maybe_mark_dynamic(x, x.dim() - 1)
    for tensor in tensors:
        torch._dynamo.mark_dynamic(tensor, x.dim() - 2, min=2, max=max_rows)
