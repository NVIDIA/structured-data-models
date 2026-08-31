# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

import torch

from sdm import RelatedTables, TableTensor


class Callback:
    r"""Base class for callbacks applied to one model call."""

    #: Whether this callback requires gradient calculation during model calls.
    requires_grad: ClassVar[bool] = False

    def on_context_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, TableTensor, RelatedTables[TableTensor] | None]:
        r"""Run after context preprocessing.

        Args:
            model: Model receiving the callback.
            x: The feature tensor of in-context examples with shape
                ``[..., R, D]`` with ``R`` rows and ``D`` columns.
            y: The targets of in-context examples with shape ``[..., R, 1]``.
            related_tables: Related context for in-context examples.
        """
        return x, y, related_tables

    def on_query_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
        r"""Run after query preprocessing.

        Args:
            model: Model receiving the callback.
            x: The feature tensor of query examples with shape ``[..., R, D]``
                with ``R`` rows and ``D`` columns.
            related_tables: Related context for query examples.
        """
        return x, related_tables

    def on_model_forward_end(
        self,
        model: torch.nn.Module,
        out: TableTensor,
    ) -> TableTensor:
        """Run after the model forward pass.

        Args:
            model: Model receiving the callback.
            out: The output produced by the model.
        """
        return out


class EnableInputGradients(Callback):  # noqa: D101
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


class CaptureInputs(Callback):  # noqa: D101
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
