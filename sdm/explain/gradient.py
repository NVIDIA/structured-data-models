# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.explain.base import ICLExplainer
from sdm.models import ICLModel
from sdm.models.callback import Callback


@dataclass(frozen=True)
class GradientExplanationOutput:
    r"""Gradients with respect to preprocessed query inputs.

    Args:
        x: Gradients for the primary query table.
        related_tables: Gradients for the related query tables, or ``None``
            when the model call has no related query tables.
    """

    x: TableTensor
    related_tables: RelatedTables[TableTensor] | None = None


class _GradientCallback(Callback):
    requires_grad = True

    def __init__(self, output: Callable[[TableTensor], Tensor]) -> None:
        self._output = output
        self._inputs: list[tuple[str | None, tuple[str, ...], Tensor]] = []
        self._related_tables: RelatedTables[TableTensor] | None = None
        self.result: GradientExplanationOutput | None = None

    def on_query_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        x = self._capture(None, x)
        if related_tables is not None:
            related_tables = related_tables.replace_tables(
                {
                    name: self._capture(name, table)
                    for name, table in related_tables.tables.items()
                }
            )
        self._related_tables = related_tables
        return x, related_tables

    def _capture(
        self,
        table_name: str | None,
        table: TableTensor,
    ) -> TableTensor:
        numerical = table.numerical.detach().requires_grad_(True)
        self._inputs.append(
            (table_name, table.columns[Stype.numerical], numerical)
        )
        return table.replace_blocks(numerical=numerical)

    def on_model_forward_end(
        self,
        model: torch.nn.Module,
        out: TableTensor,
    ) -> TableTensor:
        # TODO: Support AMP when TableTensor dtype casts preserve autograd.
        objective = self._output(out).sum()
        grads = torch.autograd.grad(
            objective,
            [numerical for _, _, numerical in self._inputs],
            allow_unused=True,
        )
        grad_tables: dict[str | None, TableTensor] = {}
        for (table_name, columns, numerical), grad in zip(
            self._inputs,
            grads,
        ):
            grad_tables[table_name] = TableTensor(
                columns={Stype.numerical: columns},
                numerical=(
                    torch.zeros_like(numerical) if grad is None else grad
                ),
            )

        self.result = GradientExplanationOutput(
            x=grad_tables[None],
            related_tables=(
                self._related_tables.replace_tables(
                    {
                        name: grad_tables[name]
                        for name in self._related_tables.tables
                    }
                )
                if self._related_tables is not None
                else None
            ),
        )
        return out


class GradientExplainer(ICLExplainer[GradientExplanationOutput]):
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
    ) -> GradientExplanationOutput:
        callback = _GradientCallback(self._output)
        model(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
            recipe=recipe,
            generator=generator,
            callbacks=(callback,),
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
    ) -> GradientExplanationOutput:
        callback = _GradientCallback(self._output)
        model.predict(
            x=x_query,
            related_tables=related_query_tables,
            callbacks=(callback,),
        )
        assert callback.result is not None
        return callback.result
