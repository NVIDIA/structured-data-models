import contextlib
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.callbacks import Callback
from sdm.models import ICLModel


class _ScaleModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_related_tables = False

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        if x_query is None:
            assert x_context is not None
            assert cache is not None
            cache["scale"] = x_context.numerical.mean()
            return x_context

        if cache is None:
            assert x_context is not None
            scale = x_context.numerical.mean()
        else:
            scale = cast(Tensor, cache["scale"])
        return x_query.replace_blocks(numerical=x_query.numerical * scale)

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe()


class _GradientCallback(Callback):
    def __init__(self) -> None:
        self.input: Tensor | None = None
        self.gradient: Tensor | None = None

    def execution_context(
        self,
        model: torch.nn.Module,
        /,
    ) -> contextlib.AbstractContextManager[None]:
        return torch.inference_mode(False)

    def on_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        self.input = x.numerical.detach().requires_grad_(True)
        return x.replace_blocks(numerical=self.input), related_tables

    def on_forward_end(
        self,
        model: torch.nn.Module,
        prediction: TableTensor,
    ) -> None:
        assert self.input is not None
        (self.gradient,) = torch.autograd.grad(
            prediction.numerical.sum(),
            self.input,
        )


@pytest.mark.parametrize("fitted", [False, True])
def test_gradient_explanation(fitted: bool) -> None:
    model = _ScaleModel()
    callback = _GradientCallback()
    x_context = torch.tensor([[1.0], [3.0]])
    y_context = torch.tensor([[0.0], [1.0]])
    x_query = torch.tensor([[5.0]])

    if fitted:
        model.fit(x_context, y_context)
        model.predict(x_query, callbacks=(callback,))
    else:
        model(x_context, y_context, x_query, callbacks=(callback,))

    torch.testing.assert_close(callback.gradient, torch.tensor([[2.0]]))
