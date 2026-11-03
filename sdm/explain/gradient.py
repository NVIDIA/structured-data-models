# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.explain.base import ICLExplainer
from sdm.models import Callback, ICLModel


class GradientExplainer(
    ICLExplainer[tuple[TableTensor, RelatedTables[TableTensor] | None]]
):
    """Return gradients showing how small input changes affect predictions.

    Gradients describe model predictions with respect to preprocessed
    numerical query inputs, not the original input data.

    For each output column, differentiate the sum of predictions over query
    rows and batch dimensions. Requires exactly one estimator.

    Attribution tables have shape ``[C, ..., R, D]``, where ``C`` is the output
    count, ``R`` the query rows, and ``D`` the preprocessed input columns.
    """

    def _explain_predict(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables[TableTensor] | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
        callbacks = (EnableInputGradients(), CaptureInputs())
        prediction = model.predict(
            x=x_query,
            related_tables=related_query_tables,
            callbacks=callbacks,
        )
        if len(callbacks[1].inputs) != 1:
            raise RuntimeError(
                "GradientExplainer requires exactly one estimator"
            )
        x, related_tables = callbacks[1].inputs[0]

        scores = prediction.numerical

        leaves = [x.numerical]
        if related_tables is not None:
            leaves.extend(
                table.numerical for table in related_tables.tables.values()
            )
        # scores: [..., R, C] -> explanations: [C, ..., R, D]
        gradient_x, *gradient_related = zip(
            *[
                torch.autograd.grad(
                    scores[..., index].sum(),
                    leaves,
                    retain_graph=index < scores.size(-1) - 1,
                    allow_unused=True,
                )
                for index in range(scores.size(-1))
            ]
        )
        x_attributions = cast(
            TableTensor,
            torch.stack(
                [
                    x.replace_blocks(
                        numerical=(
                            torch.zeros_like(x.numerical)
                            if gradient is None
                            else gradient
                        )
                    )
                    for gradient in gradient_x
                ]
            ),
        )
        if related_tables is None:
            return x_attributions, None

        related_attributions = related_tables.replace_tables(
            tables={
                name: cast(
                    TableTensor,
                    torch.stack(
                        [
                            table.replace_blocks(
                                numerical=(
                                    torch.zeros_like(table.numerical)
                                    if gradient is None
                                    else gradient
                                )
                            )
                            for gradient in gradients
                        ]
                    ),
                )
                for (name, table), gradients in zip(
                    related_tables.tables.items(),
                    gradient_related,
                )
            }
        )
        return x_attributions, related_attributions


def _enable_gradients(table: TableTensor) -> TableTensor:
    return table.replace_blocks(
        numerical=table.numerical.detach().requires_grad_(True)
    )


def _enable_related_gradients(
    related_tables: RelatedTables[TableTensor] | None,
) -> RelatedTables[TableTensor] | None:
    if related_tables is None:
        return None
    return related_tables.replace_tables(
        {
            name: _enable_gradients(table)
            for name, table in related_tables.tables.items()
        }
    )


class EnableInputGradients(Callback):
    """Enable gradients for preprocessed numerical query inputs."""

    requires_grad = True

    def on_query_preprocessing_end(  # noqa: D102
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
        return _enable_gradients(x), _enable_related_gradients(related_tables)


class EnableContextGradients(Callback):
    """Enable gradients for preprocessed numerical context inputs."""

    requires_grad = True

    def on_context_preprocessing_end(  # noqa: D102
        self,
        model: torch.nn.Module,
        x: TableTensor,
        y: TableTensor,
        x_context_only: TableTensor | None,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[
        TableTensor,
        TableTensor,
        TableTensor | None,
        RelatedTables[TableTensor] | None,
    ]:
        return (
            _enable_gradients(x),
            _enable_gradients(y),
            None
            if x_context_only is None
            else _enable_gradients(x_context_only),
            _enable_related_gradients(related_tables),
        )


class CaptureInputs(Callback):
    """Collect preprocessed query inputs."""

    def __init__(self) -> None:
        self.inputs: list[
            tuple[TableTensor, RelatedTables[TableTensor] | None]
        ] = []

    def on_query_preprocessing_end(  # noqa: D102
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
        self.inputs.append((x, related_tables))
        return x, related_tables


class CaptureContext(Callback):
    """Collect preprocessed context inputs."""

    def __init__(self) -> None:
        self.contexts: list[
            tuple[
                TableTensor,
                TableTensor,
                TableTensor | None,
                RelatedTables[TableTensor] | None,
            ]
        ] = []

    def on_context_preprocessing_end(  # noqa: D102
        self,
        model: torch.nn.Module,
        x: TableTensor,
        y: TableTensor,
        x_context_only: TableTensor | None,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[
        TableTensor,
        TableTensor,
        TableTensor | None,
        RelatedTables[TableTensor] | None,
    ]:
        self.contexts.append((x, y, x_context_only, related_tables))
        return x, y, x_context_only, related_tables
