# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import RelatedTables, TableTensor
from sdm.models import Callback


class EnableInputGradients(Callback):
    """Enable gradients for preprocessed numerical query inputs."""

    requires_grad = True

    def on_query_preprocessing_end(  # noqa: D102
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
        x = x.replace_blocks(
            numerical=x.numerical.detach().requires_grad_(True)
        )
        if related_tables is not None:
            related_tables = related_tables.replace_tables(
                {
                    name: table.replace_blocks(
                        numerical=table.numerical.detach().requires_grad_(True)
                    )
                    for name, table in related_tables.tables.items()
                }
            )
        return x, related_tables


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
        x = x.replace_blocks(
            numerical=x.numerical.detach().requires_grad_(True)
        )
        y = y.replace_blocks(
            numerical=y.numerical.detach().requires_grad_(True)
        )
        if x_context_only is not None:
            x_context_only = x_context_only.replace_blocks(
                numerical=x_context_only.numerical.detach().requires_grad_(
                    True
                )
            )
        if related_tables is not None:
            related_tables = related_tables.replace_tables(
                {
                    name: table.replace_blocks(
                        numerical=table.numerical.detach().requires_grad_(True)
                    )
                    for name, table in related_tables.tables.items()
                }
            )
        return x, y, x_context_only, related_tables


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
