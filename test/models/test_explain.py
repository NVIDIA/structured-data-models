import contextlib
from collections.abc import Iterator
from typing import Any, ClassVar, cast

import pytest
import torch
from torch import Tensor

import sdm.processing as sp
from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.callbacks import Callback
from sdm.explain import ICLExplainer
from sdm.models import ICLModel


class _ScaleModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_related_tables: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

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
        if cache is not None and cache.is_recording:
            assert x_context is not None
            cache["scale"] = (x_context.numerical * self.weight).mean()
            return x_context

        assert x_query is not None
        if cache is None:
            assert x_context is not None
            scale = (x_context.numerical * self.weight).mean()
        else:
            scale = cast(Tensor, cache["scale"])
        return x_query.replace_blocks(numerical=x_query.numerical * scale)

    @classmethod
    def default_recipe(cls) -> sp.Recipe:
        return sp.Recipe()


class _GradientCallback(Callback):
    def __init__(self) -> None:
        self.input: Tensor | None = None
        self.gradient: Tensor | None = None

    @contextlib.contextmanager
    def execution_context(
        self,
        model: torch.nn.Module,
        /,
    ) -> Iterator[None]:
        with torch.inference_mode(False), torch.enable_grad():
            yield

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
        self._finish(prediction)

    def _finish(self, prediction: TableTensor) -> None:
        assert self.input is not None
        (self.gradient,) = torch.autograd.grad(
            prediction.numerical.sum(), self.input
        )


class _GradientExplainer(ICLExplainer[Tensor]):
    def _result(self, callback: _GradientCallback) -> Tensor:
        assert callback.gradient is not None
        return callback.gradient

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
    ) -> Tensor:
        callback = _GradientCallback()
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
        return self._result(callback)

    def _explain_predict(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
    ) -> Tensor:
        callback = _GradientCallback()
        model.predict(
            x=x_query,
            related_tables=related_query_tables,
            callbacks=(callback,),
        )
        return self._result(callback)


@pytest.mark.parametrize("fitted", [False, True])
def test_explainer_controls_model_grad_mode(fitted: bool) -> None:
    model = _ScaleModel()
    explainer = _GradientExplainer()
    x_context = torch.tensor([[1.0], [3.0]])
    y_context = torch.tensor([[0.0], [1.0]])
    x_query = torch.tensor([[5.0]])

    with torch.inference_mode():
        if fitted:
            model.fit(x=x_context, y=y_context)
            gradient = explainer.explain(model, x_query)
        else:
            gradient = explainer.explain(
                model,
                x_query,
                x_context=x_context,
                y_context=y_context,
            )

    torch.testing.assert_close(gradient, torch.tensor([[2.0]]))
