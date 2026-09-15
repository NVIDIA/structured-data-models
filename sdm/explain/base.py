# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Any, Generic, TypeVar, final, overload

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, TableTensor
from sdm.models import ICLModel

T = TypeVar("T", covariant=True)


class ICLExplainer(abc.ABC, Generic[T]):  # noqa: D101
    @overload
    def explain(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables[TableTensor] | None = None,
        *,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        related_context_tables: RelatedTables[TableTensor] | None = None,
        recipe: Recipe | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> T: ...

    @overload
    def explain(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables[TableTensor] | None = None,
    ) -> T: ...

    @final
    def explain(  # noqa: D102
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables[TableTensor] | None = None,
        *,
        x_context: Tensor | TableTensor | None = None,
        y_context: Tensor | TableTensor | None = None,
        related_context_tables: RelatedTables[TableTensor] | None = None,
        recipe: Recipe | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> T:

        if x_context is not None:
            assert y_context is not None
            return self._explain_forward(
                model,
                x_context,
                y_context,
                x_query,
                related_context_tables,
                related_query_tables,
                recipe=recipe,
                generator=generator,
                **kwargs,
            )

        if model._cache is None:
            raise RuntimeError(
                f"{model.__class__.__name__!r} is not fitted. Pass required "
                f"context to 'explain()' or call "
                f"'{model.__class__.__name__}.fit()' first."
            )

        return self._explain_predict(
            model,
            x_query,
            related_query_tables,
            generator=generator,
        )

    @abc.abstractmethod
    def _explain_forward(
        self,
        model: ICLModel,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        related_context_tables: RelatedTables[TableTensor] | None = None,
        related_query_tables: RelatedTables[TableTensor] | None = None,
        *,
        recipe: Recipe | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> T: ...

    @abc.abstractmethod
    def _explain_predict(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables[TableTensor] | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> T: ...
