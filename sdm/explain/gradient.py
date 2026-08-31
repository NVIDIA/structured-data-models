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
    """Return per-output gradients for preprocessed query inputs."""

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
        if not scores.requires_grad:
            raise RuntimeError(
                "The model output is not differentiable with respect to "
                "its query inputs"
            )

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
        x = cast(
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
            return x, None

        return x, related_tables.replace_tables(
            {
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
