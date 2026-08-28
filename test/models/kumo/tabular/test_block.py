import torch

from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.testing import withCUDA


@withCUDA
def test_transformer_block(device: torch.device) -> None:
    block = KumoTabularTransformerBlock(
        channels=32,
        num_heads=4,
        device=device,
    )
    query = torch.randn(2, 5, 32, device=device)
    key_value = torch.randn(2, 3, 32, device=device)

    block(query=query, key_value=key_value)
