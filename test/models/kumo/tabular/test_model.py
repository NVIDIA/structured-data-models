from typing import Any, ClassVar

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models.kumo.tabular import model as model_module
from sdm.models.kumo.tabular.model import _KumoTabular


def test_parameter_count() -> None:
    model = _KumoTabular(
        num_classes=10,
        num_quantiles=0,
        device="meta",
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        34_188_428
    )


@pytest.mark.parametrize(
    ("num_classes", "num_quantiles"),
    [(10, 0), (0, 999)],
)
def test_forward_does_not_mutate_input(
    monkeypatch: pytest.MonkeyPatch,
    num_classes: int,
    num_quantiles: int,
) -> None:
    class CellEmbedding(torch.nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()

        def forward(
            self,
            x: torch.Tensor,
            categorical_mask: torch.Tensor,
        ) -> torch.Tensor:
            assert categorical_mask.dtype == torch.bool
            return x.unsqueeze(-1).expand(*x.shape, 128)

    class TableEncoder(torch.nn.Module):
        seen: torch.Tensor

        def __init__(self, **_: object) -> None:
            super().__init__()

        def forward(
            self,
            x: torch.Tensor,
            num_context_rows: int,
            *,
            cache: Cache | None,
        ) -> torch.Tensor:
            assert num_context_rows == 2
            assert cache is sentinel_cache
            type(self).seen = x.detach().clone()
            return x.mean(dim=-2).repeat(1, 1, 4)

    class ICLBlock(torch.nn.Module):
        def __init__(self, out_channels: int, **_: object) -> None:
            super().__init__()
            self.out_channels = out_channels

        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            *,
            cache: Cache | None,
            batch_size_limit: int | None,
        ) -> torch.Tensor:
            assert cache is sentinel_cache
            assert batch_size_limit == 7
            return x.new_empty(
                (
                    *x.shape[:-2],
                    x.size(-2) - y.size(-1),
                    self.out_channels,
                )
            )

    monkeypatch.setattr(model_module, "CellEmbedding", CellEmbedding)
    monkeypatch.setattr(model_module, "TableEncoder", TableEncoder)
    monkeypatch.setattr(model_module, "ICLBlock", ICLBlock)
    model = _KumoTabular(
        num_classes=num_classes,
        num_quantiles=num_quantiles,
    )

    x = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    original = x.clone()
    y = (
        torch.tensor([[0, 1]])
        if num_classes > 0
        else torch.tensor([[0.25, -1.5]])
    )
    categorical_mask = torch.tensor([[False, True, False, True, False, False]])
    sentinel_cache = Cache()
    out = model(
        x=x,
        y=y,
        categorical_mask=categorical_mask,
        cache=sentinel_cache,
        batch_size_limit=7,
    )

    torch.testing.assert_close(x, original)
    assert out.size() == (1, 2, num_classes or num_quantiles)
    torch.testing.assert_close(
        TableEncoder.seen[..., 2:, :, :],
        original[..., 2:, :, None].expand(1, 2, 6, 128),
    )
    expected_context = original[..., :2, :, None].expand(1, 2, 6, 128)
    y_emb = (
        torch.nn.functional.one_hot(y, num_classes=num_classes)
        if num_classes > 0
        else y.unsqueeze(-1)
    )
    expected_context = expected_context + model.y_encoder(
        y_emb.to(model.y_encoder.weight.dtype)
    ).unsqueeze(-2)
    torch.testing.assert_close(
        TableEncoder.seen[..., :2, :, :],
        expected_context,
    )


class _RecordingCore(torch.nn.Module):
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        **_: object,
    ) -> None:
        super().__init__()
        assert (num_classes, num_quantiles) == (10, 0)

    def forward(
        self,
        x: torch.Tensor,  # [..., R, C]
        y: torch.Tensor,  # [..., R_context]
        categorical_mask: torch.Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> torch.Tensor:  # [..., R_query, 10]
        type(self).calls.append(
            {
                "x": x.clone(),
                "y": y.clone(),
                "categorical_mask": categorical_mask.clone(),
                "batch_size_limit": batch_size_limit,
            }
        )
        query = x[..., y.size(-1) :, :]
        return query.sum(dim=-1, keepdim=True) + torch.arange(
            10,
            device=x.device,
            dtype=x.dtype,
        )


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> KumoTabular:
    _RecordingCore.calls.clear()
    monkeypatch.setattr(model_module, "_KumoTabular", _RecordingCore)
    return KumoTabular()


def _features() -> tuple[TableTensor, TableTensor]:
    x = TableTensor(
        columns={Stype.numerical: ("n0", "n1"), Stype.categorical: ("c",)},
        numerical=torch.tensor(
            [
                [100.0, 200.0],
                [101.0, 201.0],
                [102.0, 202.0],
                [103.0, 203.0],
                [104.0, 204.0],
            ]
        ),
        categorical=CategoricalTensor.from_tensor(
            torch.tensor([[0], [1], [0], [0], [1]])
        ),
    )
    return x.split(3, dim=0)


def _target(num_classes: int = 3) -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [2], [0]]),
            categories=(torch.arange(num_classes).mul(10),),
        ),
    )


def _recipe() -> sp.Recipe:
    return sp.Recipe(
        features=[sp.ToNumerical()],
        output=[sp.ReduceEstimators(method="mean")],
    )


def test_forward(model: KumoTabular) -> None:
    x_context, x_query = _features()

    out = model(
        x_context,
        _target(),
        x_query,
        recipe=_recipe(),
        batch_size_limit=7,
    )

    assert out.size() == (2, 3)
    assert out.columns[Stype.numerical] == ("0", "10", "20")

    call = _RecordingCore.calls[0]
    assert call["x"][..., :2].equal(
        torch.cat((x_context.numerical, x_query.numerical), dim=-2)
    )
    assert call["y"].equal(torch.tensor([0, 2, 0]))
    # Columns turned numerical by the recipe stay marked as categorical.
    assert call["categorical_mask"].tolist() == [False, False, True]
    assert call["batch_size_limit"] == 7


def test_fit_predict(model: KumoTabular) -> None:
    x_context, x_query = _features()
    target = _target()

    expected = model(x_context, target, x_query, recipe=_recipe())
    model.fit(x_context, target, recipe=_recipe())
    actual = model.predict(x_query)

    assert actual.allclose(expected)
    assert actual.columns == expected.columns
    # The mask is derived from the context schema, which `predict` lacks.
    fit_call, predict_call = _RecordingCore.calls[-2:]
    assert predict_call["categorical_mask"].equal(fit_call["categorical_mask"])
