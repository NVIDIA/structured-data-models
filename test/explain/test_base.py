import contextlib

import pytest
import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.callbacks import Callback
from sdm.models import TabICLv2


class _AutogradCallback(Callback):
    input: Tensor
    completed: bool = False

    def execution_context(
        self,
        model: torch.nn.Module,
    ) -> contextlib.AbstractContextManager[None]:
        return torch.inference_mode(False)

    def on_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        assert torch.is_inference_mode_enabled()
        self.input = x.numerical.detach().requires_grad_()
        return x.replace_blocks(numerical=self.input), related_tables

    def on_forward_end(
        self,
        model: torch.nn.Module,
        prediction: TableTensor,
    ) -> None:
        assert not torch.is_inference_mode_enabled()
        torch.autograd.grad(prediction.numerical.sum(), self.input)
        self.completed = True


@pytest.mark.parametrize("fitted", [False, True])
def test_callback_context_supports_autograd(fitted: bool) -> None:
    model = TabICLv2(pretrained=False)
    callback = _AutogradCallback()
    x_context = torch.eye(2)
    y_context = torch.arange(2)[:, None]
    x_query = torch.ones(1, 2)

    if fitted:
        model.fit(x_context, y_context)
        model.predict(x_query, callbacks=(callback,))
    else:
        model(x_context, y_context, x_query, callbacks=(callback,))

    assert callback.completed
