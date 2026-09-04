import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.row_embedding import RowEmbedding
from sdm.nn import GatedLogScale, InducedTransformerBlock
from sdm.testing import withCUDA


@withCUDA
def test_row_embedding(device: torch.device) -> None:
    encoder = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=4,
        num_heads=2,
        group_size=3,
        num_frequencies=32,
        num_inducing_points=4,
        num_readout_tokens=2,
        device=device,
    )
    x = torch.randn(2, 5, 3, device=device)
    y = torch.randn(2, 3, device=device)
    categorical_mask = torch.zeros(2, 3, device=device, dtype=torch.bool)

    with torch.no_grad():
        out = encoder(x, y, categorical_mask)
    assert out.size() == (2, 5, 32)
    assert out.device == device

    cache = Cache()
    with torch.no_grad():
        encoder(x[:, :3], y, categorical_mask, cache=cache)
        out = encoder(
            x[:, 3:],
            y[:, :0],
            categorical_mask,
            cache=cache.freeze(),
        )
    assert out.size() == (2, 2, 32)
    assert out.device == device


def test_row_log_scale() -> None:
    plain = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_frequencies=32,
        num_inducing_points=4,
        num_readout_tokens=2,
    )
    encoder = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_frequencies=32,
        num_inducing_points=4,
        num_readout_tokens=2,
        row_log_scale=True,
    )

    assert not any(
        isinstance(module, GatedLogScale) for module in plain.modules()
    )
    for block in encoder.row_blocks:
        assert isinstance(block, KumoTabularTransformerBlock)
        assert isinstance(block.attn.sdpa.query_scaling, GatedLogScale)

    for block in encoder.col_blocks:
        assert isinstance(block, InducedTransformerBlock)
        for attention_block in (block.inducing_block, block.output_block):
            assert isinstance(attention_block, KumoTabularTransformerBlock)
            assert not isinstance(
                attention_block.attn.sdpa.query_scaling,
                GatedLogScale,
            )

    x = torch.randn(2, 5, 3)
    y = torch.randn(2, 3)
    categorical_mask = torch.zeros(2, 3, dtype=torch.bool)
    with torch.no_grad():
        out = encoder(x, y, categorical_mask)
    assert out.size() == (2, 5, 32)
    assert out.isfinite().all()
