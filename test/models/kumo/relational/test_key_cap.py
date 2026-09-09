from typing import Any

import pytest
import torch

from sdm.models import KumoRelational
from sdm.models.tabiclv2.row_embedding import RowEmbedding


@pytest.mark.parametrize(
    "config", [{}, {"max_keys": 40_000}, {"max_keys": None}]
)
def test_public_key_cap_reaches_each_tasks_row_embedding(
    config: dict[str, int | None],
) -> None:
    # Meta construction avoids downloading checkpoints or allocating weights.
    model = KumoRelational(None, False, "meta", **config)
    seen = []

    class Capture(torch.nn.Module):
        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            seen.append(kwargs["max_keys"])
            return x

    for task_model in model.models.values():
        task_model.row_embedding = Capture()
        context, query = task_model._embed_table(
            x_context=torch.zeros(3, 2),
            x_query=torch.zeros(2, 2),
            y=torch.zeros(3),
            task_row=torch.arange(3),
            num_classes=None,
            cache_key="entity",
            cache=None,
            generator=None,
        )
        assert context.shape == (3, 2)
        assert query.shape == (2, 2)
    assert seen == [config.get("max_keys", 20_000)] * len(model.models)


@pytest.mark.parametrize(("max_keys", "expected"), [(2, 2), (8, 5), (None, 5)])
def test_key_cap_limits_context_keys_without_dropping_query_rows(
    max_keys: int | None, expected: int
) -> None:
    encoder = RowEmbedding(
        num_classes=0,
        channels=8,
        num_layers=1,
        num_heads=2,
        group_size=1,
        num_inducing_points=2,
        num_readout_tokens=1,
        norm_bias=True,
    )
    key_rows = []

    def capture(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        key_rows.append(kwargs["key_value"].size(-2))

    encoder.col_layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    output = encoder(torch.randn(7, 2), torch.randn(5), max_keys=max_keys)
    assert key_rows == [expected]
    assert output.shape == (7, 8)
