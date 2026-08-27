from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.callbacks import Callback
from sdm.explain.base import ICLExplainer
from sdm.models import ICLModel


@dataclass(frozen=True)
class GradientExplanationOutput:
    r"""Gradients with respect to preprocessed query inputs.

    Args:
        x: Gradients for the primary query table.
        related_tables: Gradients keyed by related query table name, or
            ``None`` when the model call has no related query tables.
    """

    x: TableTensor
    related_tables: Mapping[str, TableTensor] | None = None


class _GradientCallback(Callback):
    requires_grad = True

    def __init__(self, output: Callable[[TableTensor], Tensor]) -> None:
        self._output = output
        self._inputs: list[tuple[str | None, tuple[str, ...], Tensor]] = []
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
        self._on_forward_end(out)
        return out

    def _on_forward_end(self, prediction: TableTensor) -> None:
        if not self._inputs:
            raise RuntimeError("The model did not expose its query inputs")

        # TODO: Support AMP when TableTensor dtype casts preserve autograd.
        objective = self._output(prediction).sum()
        gradients = torch.autograd.grad(
            objective,
            [numerical for _, _, numerical in self._inputs],
            allow_unused=True,
        )
        x: TableTensor | None = None
        related_tables: dict[str, TableTensor] = {}
        for (table_name, columns, numerical), gradient in zip(
            self._inputs,
            gradients,
        ):
            gradient_table = TableTensor(
                columns={Stype.numerical: columns},
                numerical=(
                    torch.zeros_like(numerical)
                    if gradient is None
                    else gradient
                ),
            )
            if table_name is None:
                x = gradient_table
            else:
                related_tables[table_name] = gradient_table

        if x is None:
            raise RuntimeError("The model did not expose its primary query")
        self.result = GradientExplanationOutput(
            x=x,
            related_tables=related_tables or None,
        )


class GradientExplainer(ICLExplainer[GradientExplanationOutput]):
    r"""Return gradients with respect to preprocessed query inputs.

    The explainer differentiates one public model execution with respect to
    the numerical blocks produced by the input recipe for the primary query
    and any related query tables. Returned gradients are signed and
    unnormalized. Recipe-derived numerical columns retain their transformed
    column names.

    Args:
        output: Select the prediction values to differentiate. Selected values
            are summed before differentiation.
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
    ) -> GradientExplanationOutput:
        callback = _GradientCallback(self._output)
        model.predict(
            x=x_query,
            related_tables=related_query_tables,
            callbacks=(callback,),
        )
        assert callback.result is not None
        return callback.result
