import functools

import pytest
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models import TabFM
from sdm.models.tabfm import model as tabfm_module
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_forward(
    device: torch.device,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tabfm_module,
        "_TabFM",
        functools.partial(
            tabfm_module._TabFM,
            channels=64,
            num_inducing_points=128,
            num_readout_tokens=4,
            num_icl_layers=4,
        ),
    )

    model = TabFM(
        task="regression" if dtype.is_floating_point else "classification",
        checkpoint_path=None,
        device=device,
    )
    if device.type == "cpu":
        assert repr(model) == "TabFM()"
    else:
        assert repr(model) == "TabFM(device=cuda:0)"

    x = TableTensor(
        numerical=torch.randn(8, 3, device=device),
        categorical=CategoricalTensor.from_tensor(
            torch.randint(0, 2, (8, 3), device=device)
        ),
    )
    x_context, x_query = x.split(5, dim=0)

    if dtype.is_floating_point:
        y_context = torch.randn(5, 1, device=device)
    else:
        y_context = torch.tensor([0, 1, 0, 1, 0], device=device).unsqueeze(-1)

    torch.manual_seed(1)
    out = model(x_context, y_context, x_query)
    assert out.dtype == x_context.dtype
    assert out.device == device
    assert torch.is_inference(out)
    if dtype.is_floating_point:
        assert out.size() == (1, 3, 1)
    else:
        assert out.size() == (1, 3, 2)

    torch.manual_seed(1)
    model.fit(x_context, y_context)
    assert model._cache is not None
    assert model._cache.size() > 0
    assert model.predict(x_query).allclose(out)
    model.clear()
