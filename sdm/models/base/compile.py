# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Compiled components of :meth:`~sdm.models.ICLModel.predict`.

A compiled predict runs the steps that organize the work as plain Python:
validation, the loop over estimator batches, the prefetch of the fitted cache
and the stacking of estimators. It compiles only components, functions that
do the work on tensors, *e.g.*, the model of one estimator batch.
:meth:`~sdm.models.ICLModel._predict` takes the components in place of its
default steps, so that each model chooses its own.

A graph of a component specializes on the Python values among its inputs,
*e.g.*, names of columns, and on sizes of zero and one. Keep both outside of
components to reuse their graphs after fits on other data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor

if TYPE_CHECKING:
    from sdm.models.base import Callback


def compile_component(
    fn: Callable[..., Any],
    **kwargs: Any,
) -> Callable[..., Any]:
    r"""Compile a component of predict using :func:`torch.compile`.

    Args:
        fn: The component.
        kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``True``, so that all sizes are dynamic
            from the first call; with ``None``, sizes are static until they
            change or are marked, *e.g.*, by
            :func:`torch._dynamo.maybe_mark_dynamic`. ``isolate_recompiles``
            defaults to ``True``, since all components would otherwise share
            one recompile limit.
    """
    kwargs = {"dynamic": True, "isolate_recompiles": True} | kwargs
    # A copy, from which `torch.compile` removes some options.
    if kwargs.get("options") is not None:
        kwargs["options"] = dict(kwargs["options"])
    # Sizes that are equal in the first call would otherwise share one symbol,
    # and compile another graph once they differ.
    return torch.fx.experimental._config.patch(use_duck_shape=False)(
        torch.compile(fn, **kwargs)
    )


def repeat_single_rows(
    predict: Callable[..., TableTensor],
    uncompiled: Callable[..., TableTensor],
) -> Callable[..., TableTensor]:
    r"""Keep queries of zero and one rows away from compiled components.

    A query without rows runs ``uncompiled``, and a query of one row is
    repeated and its first row returned, unless callbacks or related tables
    see the rows.

    Args:
        predict: The compiled predict.
        uncompiled: The uncompiled predict.
    """

    def wrapper(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        rows = x.size(-2)
        if rows == 0:
            return uncompiled(x, related_tables, callbacks=callbacks)
        if rows > 1 or related_tables is not None or callbacks:
            return predict(x, related_tables, callbacks=callbacks)
        out = predict(torch.cat([x, x], dim=-2), callbacks=callbacks)
        return cast(TableTensor, out[..., :1, :])

    return wrapper
