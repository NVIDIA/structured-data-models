import pytest
import torch
from sdm.models._shared import ICLBlock, RowEmbedding
from sdm.models.kumorfm.model import _KumoRFM
from sdm.models.tabiclv2.icl import ICLBlock as CompatibilityICLBlock
from sdm.models.tabiclv2.model import _TabICLv2
from sdm.models.tabiclv2.row_embedding import (
    RowEmbedding as CompatibilityRowEmbedding,
)
from sdm.nn import Attention
from sdm.testing import withCUDA


def test_models_use_shared_component_definitions() -> None:
    tabiclv2 = _TabICLv2(
        num_classes=3,
        num_quantiles=0,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=3,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        norm_bias=True,
    )
    kumorfm = _KumoRFM(
        num_classes=3,
        num_quantiles=0,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=3,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        norm_bias=True,
    )

    assert type(tabiclv2.row_embedding) is RowEmbedding
    assert type(kumorfm.row_embedding) is RowEmbedding
    assert type(tabiclv2.icl_block) is ICLBlock
    assert type(kumorfm.icl_block) is ICLBlock


def test_tabiclv2_component_imports_remain_compatible() -> None:
    assert CompatibilityRowEmbedding is RowEmbedding
    assert CompatibilityICLBlock is ICLBlock


@withCUDA
def test_row_embedding_mixed_radix_digit(device: torch.device) -> None:
    row_embedding = RowEmbedding(
        num_classes=10,
        channels=8,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
        device=device,
    )
    for module in row_embedding.modules():
        # Randomly initialize to return non-zero output.
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.02)

    x = torch.randn(8, 6, device=device)

    # The labels 5 * a + b and 5 * b + a decompose into the digits (a, b) and
    # (b, a) under bases [5, 5], so averaging over digits is unchanged after
    # swapping them.
    a = torch.tensor([4, 0, 1, 2, 3], device=device)
    b = torch.tensor([4, 1, 2, 3, 0], device=device)
    y = 5 * a + b
    y_swapped = 5 * b + a
    out = row_embedding(x, y, num_classes=25)
    torch.testing.assert_close(
        out,
        row_embedding(x, y_swapped, num_classes=25),
    )


@withCUDA
@pytest.mark.parametrize("num_classes", [0, 3])
def test_icl_block_forward(
    device: torch.device,
    num_classes: int,
) -> None:
    block = ICLBlock(
        num_classes=num_classes,
        channels=8,
        num_layers=2,
        num_heads=2,
        norm_bias=True,
        device=device,
    )
    x = torch.randn(5, 8, device=device)
    if num_classes:
        y = torch.tensor([0, 1, 2], device=device)
    else:
        y = torch.randn(3, device=device)

    out = block(x, y)

    assert out.size() == (2, 8)
    assert out.dtype == x.dtype
    assert out.device == device
