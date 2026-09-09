import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.cell_embedding import (
    FourierNanIndicatorCellEmbedding,
)
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


def test_gated_row_log_scale() -> None:
    encoder = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_frequencies=32,
        num_inducing_points=4,
        num_readout_tokens=2,
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


@withCUDA
def test_nan_indicator_cache_matches_joint_forward(
    device: torch.device,
) -> None:
    encoder = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_frequencies=4,
        num_inducing_points=4,
        num_readout_tokens=2,
        device=device,
    ).eval()
    assert isinstance(
        encoder.cell_embedding,
        FourierNanIndicatorCellEmbedding,
    )

    x = torch.randn(2, 5, 3, device=device)
    x[0, 0, 0] = torch.nan
    x[0, 4, 1] = torch.nan
    x[1, 1, 2] = torch.nan
    x[1, 3, 0] = torch.nan
    y = torch.randn(2, 3, device=device)
    categorical_mask = torch.zeros(2, 3, dtype=torch.bool, device=device)

    with torch.no_grad():
        expected = encoder(x, y, categorical_mask)[..., 3:, :]
        cache = Cache()
        encoder(x[..., :3, :], y, categorical_mask, cache=cache)
        actual = encoder(
            x[..., 3:, :],
            y[..., :0],
            categorical_mask,
            cache=cache.freeze(),
        )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
