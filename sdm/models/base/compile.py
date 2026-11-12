# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shapes of a compiled :meth:`~sdm.models.ICLModel.predict`."""

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

#: A dimension of a fitted tensor.
Dim = tuple[Tensor, int]


@dataclass(frozen=True)
class DynamicShapes:
    r"""The sizes one compiled graph of a model serves; unset is static.

    A model declares them per batch of estimators in
    :meth:`~sdm.models.ICLModel._dynamic_shapes`. Sizes below two always
    compile their own graph.
    """

    #: The query rows one pass of the model holds in memory; larger queries
    #: are split. The graph does not depend on the capacity, which only
    #: trades memory for passes, as long as attention batches fit their
    #: kernels. Requires that the model and its recipe predict query rows
    #: independently of each other.
    query_rows: int | None = None
    #: The context-row dimensions of fitted tensors.
    context_row_dims: Sequence[Dim] = ()
    #: The column dimensions of fitted tensors of the model; those of the
    #: recipe are known. Numerical tables then run with unnamed columns,
    #: which the recipe must not read.
    column_dims: Sequence[Dim] = ()


def compile_predict(
    model: ICLModel,
    **kwargs: Any,
) -> Callable[..., TableTensor]:
    r"""Compile ``model.predict`` using :func:`torch.compile`.

    With ``dynamic=False``, one graph serves the :class:`DynamicShapes` the
    fitted model declares, also across fits: the query rows and the declared
    dimensions of fitted tensors are marked as dynamic, and queries are split
    into chunks of at most ``query_rows`` rows. Numerical tables that are
    named as in :meth:`fit` run unnamed with dynamic columns. Empty queries,
    queries with related tables or callbacks, and models without capacity for
    two rows are predicted eagerly.

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
    compiled = torch._dynamo.config.patch(
        # Marks on fitted buffers apply only with this option.
        force_parameter_static_shapes=False,
        # Inductor miscompiles copies of fitted tensors of dynamic size
        # across CUDA streams otherwise.
        inline_single_use_invoke_subgraph=False,
    )(compiled)

    def predict(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        shapes = _dynamic_shapes(model, x.device)
        rows = [shape.query_rows for shape in shapes]
        if not rows or None in rows:
            return compiled(x, related_tables, callbacks=callbacks)
        capacity = min(cast(list[int], rows))
        if (
            capacity < 2
            or x.size(-2) == 0
            or related_tables is not None
            or callbacks
        ):
            return model.predict(x, related_tables, callbacks=callbacks)

        _mark([dim for shape in shapes for dim in shape.context_row_dims])
        column_dims = [dim for shape in shapes for dim in shape.column_dims]
        if (
            column_dims
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
            # The recipe may keep a fixed number of the columns.
            _mark([*_recipe_column_dims(model), *column_dims], strict=False)

        outs: list[Tensor] = []
        for start in range(0, x.size(-2), capacity):
            end = min(start + capacity, x.size(-2))
            # Gather every chunk, so all chunks share one layout, and repeat
            # the row of single-row chunks: a single row would specialize.
            index = torch.arange(start, max(end, start + 2), device=x.device)
            chunk = x.index_select(-2, index.clamp_(max=end - 1))
            _mark_query(chunk)
            outs.append(compiled(chunk)[..., : end - start, :])
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)
        return cast(TableTensor, out)

    return predict


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


def _mark(dims: Sequence[Dim], strict: bool = True) -> None:
    # Marks do not survive copies of the fitted state, so mark before every
    # call. Sizes 0 and 1 always specialize, which strict marks reject.
    dynamo = torch._dynamo
    mark = dynamo.mark_dynamic if strict else dynamo.maybe_mark_dynamic
    for tensor, dim in dims:
        if tensor.size(dim) >= 2:
            mark(tensor, dim % tensor.dim())


def _recipe_column_dims(model: ICLModel) -> Iterator[Dim]:
    # The column dimensions of what the recipe and `ICLModel.fit` cache.
    cache = cast(Cache, model._cache)
    recipe = cast(RecipeExecution, cache["recipe_execution"]).recipe
    for processor in recipe.features.modules():
        if isinstance(processor, Processor) and processor.is_fitted:
            yield from processor._column_dims()
    for i in range(cast(int, cache["num_batches"])):
        yield cast(Tensor, cast(Cache, cache[i])["categorical_mask"]), -1


def _mark_query(x: Tensor) -> None:
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
        torch._dynamo.mark_dynamic(tensor, x.dim() - 2)
