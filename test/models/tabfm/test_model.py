import functools
from datetime import UTC, datetime

import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import TabFM
from sdm.models.tabfm import model as tabfm_module
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


def test_default_recipe_converts_datetime_features() -> None:
    timestamps = torch.tensor(
        [
            [datetime(2019, 1, 2, tzinfo=UTC).timestamp() * 1_000_000],
            [datetime(2020, 4, 6, tzinfo=UTC).timestamp() * 1_000_000],
            [datetime(2021, 8, 12, tzinfo=UTC).timestamp() * 1_000_000],
            [torch.iinfo(torch.int64).min],
        ],
        dtype=torch.int64,
    )
    table = TableTensor(datetime=timestamps)

    output = TabFM.default_recipe().features.fit_transform_ensemble(
        EnsembleTable(table, num_members=1),
        generator=torch.Generator().manual_seed(0),
    )

    member = output.table(0)
    assert member.active_stypes == {Stype.numerical}
    assert member.numerical.dtype == table.numerical.dtype
    assert member.numerical.size(-1) == 5
    assert member.numerical.isfinite().all()


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

    generator = torch.Generator(device=device).manual_seed(1)
    out = model(x_context, y_context, x_query, generator=generator)
    assert out.dtype == x_context.dtype
    assert out.device == device
    assert torch.is_inference(out)
    if dtype.is_floating_point:
        assert out.size() == (3, 1)
    else:
        assert out.size() == (3, 2)

    generator = torch.Generator(device=device).manual_seed(1)
    model.fit(x_context, y_context, generator=generator)
    assert model._cache is not None
    assert model._cache.size() > 0
    assert model.predict(x_query).allclose(out)
    model.clear()
