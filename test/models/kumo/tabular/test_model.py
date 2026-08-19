import pytest
import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular import model as model_module
from sdm.models.kumo.tabular.model import _KumoTabular


def test_parameter_count() -> None:
    model = _KumoTabular(device="meta")

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        34_188_428
    )


def test_forward_does_not_mutate_input(
    monkeypatch: pytest.MonkeyPatch,
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
        def __init__(self, **_: object) -> None:
            super().__init__()

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
            return x[..., y.size(-1) :, :10]

    monkeypatch.setattr(model_module, "CellEmbedding", CellEmbedding)
    monkeypatch.setattr(model_module, "TableEncoder", TableEncoder)
    monkeypatch.setattr(model_module, "ICLBlock", ICLBlock)
    model = _KumoTabular()

    x = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    original = x.clone()
    y = torch.tensor([[0, 1]])
    categorical_mask = torch.tensor([[False, True, False, True, False, False]])
    sentinel_cache = Cache()
    out = model(
        x,
        y,
        categorical_mask,
        cache=sentinel_cache,
        batch_size_limit=7,
    )

    torch.testing.assert_close(x, original)
    assert out.size() == (1, 2, 10)
    torch.testing.assert_close(
        TableEncoder.seen[..., 2:, :, :],
        original[..., 2:, :, None].expand(1, 2, 6, 128),
    )
    expected_context = original[..., :2, :, None].expand(1, 2, 6, 128)
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=10)
    expected_context = expected_context + model.y_encoder(
        y_one_hot.to(model.y_encoder.weight.dtype)
    ).unsqueeze(-2)
    torch.testing.assert_close(
        TableEncoder.seen[..., :2, :, :],
        expected_context,
    )
