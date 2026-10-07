# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.explain.base import ICLExplainer
from sdm.models import ICLModel
from sdm.models.callback import Callback
from sdm.models.input_callbacks import CaptureInputs, EnableInputGradients


class _GradientCallback(Callback):
    requires_grad = True

    def __init__(
        self,
        output: Callable[[TableTensor], Tensor],
        inputs: CaptureInputs,
    ) -> None:
        self._output = output
        self._inputs = inputs
        self.result: (
            tuple[TableTensor, RelatedTables[TableTensor] | None] | None
        ) = None

    def on_model_forward_end(
        self,
        model: torch.nn.Module,
        out: TableTensor,
    ) -> TableTensor:
        # TODO: Support AMP when TableTensor dtype casts preserve autograd.
        objective = self._output(out).sum()
        inputs: list[tuple[str | None, TableTensor]] = []
        for x, related_tables in self._inputs.inputs:
            inputs.append((None, x))
            if related_tables is not None:
                inputs.extend(related_tables.tables.items())
        grads = torch.autograd.grad(
            objective,
            [table.numerical for _, table in inputs],
            allow_unused=True,
            retain_graph=True,
        )
        grad_tables: dict[str | None, TableTensor] = {}
        for (table_name, table), grad in zip(
            inputs,
            grads,
        ):
            grad_tables[table_name] = TableTensor(
                columns={Stype.numerical: table.columns[Stype.numerical]},
                numerical=(
                    torch.zeros_like(table.numerical) if grad is None else grad
                ),
            )

        self.result = (
            grad_tables[None],
            (
                related_tables.replace_tables(
                    {name: grad_tables[name] for name in related_tables.tables}
                )
                if related_tables is not None
                else None
            ),
        )
        return out


class GradientExplainer(
    ICLExplainer[tuple[TableTensor, RelatedTables[TableTensor] | None]]
):
    r"""Return gradients of selected outputs with respect to query inputs.

    Args:
        output: Map a model output to the tensor values to differentiate.
    """

    def __init__(
        self,
        *,
        output: Callable[[TableTensor], Tensor],
    ) -> None:
        self._output = output

    def _explain_forward(
        self,
        model: ICLModel,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
        inputs = CaptureInputs()
        callback = _GradientCallback(self._output, inputs)
        model(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
            recipe=recipe,
            generator=generator,
            callbacks=(EnableInputGradients(), inputs, callback),
            **kwargs,
        )
        assert callback.result is not None
        return callback.result

    def _explain_predict(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
        inputs = CaptureInputs()
        callback = _GradientCallback(self._output, inputs)
        model.predict(
            x=x_query,
            related_tables=related_query_tables,
            callbacks=(EnableInputGradients(), inputs, callback),
        )
        assert callback.result is not None
        return callback.result
