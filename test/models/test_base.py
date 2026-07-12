import pytest
import torch
from sdm.cache import Cache
from sdm.models.base import BaseModel
from sdm.processing import Recipe
from torch import Tensor


class LegacyModel(BaseModel):
    """Model implementing the pre-``batch_size_limit`` private hook."""

    supports_related_tables = False

    def _forward(
        self,
        x: Tensor,
        y: Tensor,
        related_tables: object,
        cache: Cache | None = None,
    ) -> Tensor:
        del related_tables, cache
        return x[..., y.size(-1) :, :1]

    def default_recipe(self) -> Recipe:
        return Recipe()


def test_legacy_base_model_forward_compatibility() -> None:
    """The default path does not pass new keywords to existing subclasses."""
    model = LegacyModel()
    x = torch.randn(5, 2)
    y = torch.randn(3)

    torch.testing.assert_close(model(x, y), x[3:, :1])
    model.fit(x[:3], y)
    torch.testing.assert_close(model.predict(x[3:]), x[3:, :1])

    with pytest.raises(TypeError, match="batch_size_limit"):
        model(x, y, batch_size_limit=2)
