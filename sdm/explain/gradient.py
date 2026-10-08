# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.explain.base import ICLExplainer
from sdm.models import ICLModel
from sdm.models.callback import CaptureInputs, EnableInputGradients


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
