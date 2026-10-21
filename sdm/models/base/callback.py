# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

import torch

from sdm import RelatedTables, TableTensor


class Callback:
    r"""Base class for callbacks attached to a model."""

    #: Whether this callback requires gradient calculation.
    requires_grad: ClassVar[bool] = False

    def on_context_preprocessing_end(
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
        r"""Called after context preprocessing.

        Args:
            model: The model receiving the callback.
            x: The feature tensor of in-context examples with shape
                ``[..., R, D]`` with ``R`` rows and ``D`` columns.
            y: The targets of in-context examples with shape ``[..., R, 1]``.
            x_context_only: The feature tensor available only in context with
                shape ``[..., R, D_only]``, or ``None`` when absent.
            related_tables: Related context for in-context examples.
        """
        return x, y, x_context_only, related_tables

    def on_query_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
        r"""Called after query preprocessing.

        Args:
            model: the model receiving the callback.
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
        """Called after the model forward pass.

        Args:
            model: the model receiving the callback.
            out: The output produced by the model.
        """
        return out
