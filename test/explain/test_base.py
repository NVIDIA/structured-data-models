import pytest
import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.callbacks import Callback
from sdm.models import TabICLv2


class MyCallback(Callback):
    requires_grad = True
    input: Tensor
    completed: bool = False

    def on_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        assert torch.is_grad_enabled()
        self.input = x.numerical.requires_grad_()
        return x, related_tables

    def on_forward_end(
        self,
        model: torch.nn.Module,
        prediction: TableTensor,
    ) -> None:
        torch.autograd.grad(prediction.numerical.sum(), self.input)
        self.completed = True


@pytest.mark.parametrize("fitted", [False, True])
def test_callback_requires_grad_supports_autograd(fitted: bool) -> None:
    model = TabICLv2(pretrained=False)
    callback = MyCallback()
    callbacks = (Callback(), callback)
    x_context = torch.eye(2)
    y_context = torch.arange(2)[:, None]
    x_query = torch.ones(1, 2)

    if fitted:
        model.fit(x_context, y_context)
        model.predict(x_query, callbacks=callbacks)
    else:
        model(x_context, y_context, x_query, callbacks=callbacks)

    assert callback.completed
