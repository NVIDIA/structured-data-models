from typing import cast

import pytest
import torch

from sdm.models.tabfm.attention import _attention_transforms
from sdm.nn import Attention, RotaryEmbedding, SoftplusScale
from sdm.testing import withCUDA

# Frozen from google-research/tabfm@b8a8b090 with key and value sharing the
# same source, matching SDM's supported attention API.
_GOOGLE_OUTPUT = {
    torch.float32: [
        -0.2720957100391388,
        0.12698763608932495,
        0.44607099890708923,
        -0.19484566152095795,
        -0.27348652482032776,
        0.12753960490226746,
        0.4485657513141632,
        -0.1904081106185913,
    ],
    torch.bfloat16: [
        -0.271484375,
        0.126953125,
        0.4453125,
        -0.1953125,
        -0.2734375,
        0.1279296875,
        0.447265625,
        -0.1904296875,
    ],
}


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_matches_google(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    query_transform, key_transform = _attention_transforms(
        channels=4,
        num_heads=2,
        rope_theta=100_000.0,
        device=device,
        dtype=dtype,
    )
    module = Attention(
        channels=4,
        num_query_heads=2,
        query_transform=query_transform,
        key_transform=key_transform,
        scale=1.0,
        device=device,
        dtype=dtype,
    ).eval()
    query_transform = cast(torch.nn.Sequential, module.query_transform)
    key_transform = cast(torch.nn.Sequential, module.key_transform)
    query_rope = cast(RotaryEmbedding, query_transform[0])
    query_norm = cast(torch.nn.RMSNorm, query_transform[1])
    query_scale = cast(SoftplusScale, query_transform[2])
    key_rope = cast(RotaryEmbedding, key_transform[0])
    key_norm = cast(torch.nn.RMSNorm, key_transform[1])
    with torch.no_grad():
        base = torch.arange(16, device=device).view(4, 4)
        bias = torch.tensor([-0.07, 0.03, 0.11, -0.05], device=device)
        weights = [
            (base - 7.5) * (0.013 + index * 0.002) for index in range(3)
        ]
        module.qkv_lin.weight.copy_(torch.cat(weights).to(dtype))
        module.qkv_lin.bias.copy_(
            torch.cat([bias * (index + 1) for index in range(3)]).to(dtype)
        )
        module.out_lin.weight.copy_(((base - 7.5) * 0.019).to(dtype))
        module.out_lin.bias.copy_((bias * 4).to(dtype))
        query_rope.inv_freq.copy_(
            torch.tensor([0.37], device=device, dtype=dtype)
        )
        query_norm.weight.copy_(
            torch.tensor([0.8, 1.2], device=device, dtype=dtype)
        )
        query_scale.weight.copy_(
            torch.tensor([-0.4, 0.6], device=device, dtype=dtype)
        )
        key_rope.inv_freq.copy_(
            torch.tensor([0.37], device=device, dtype=dtype)
        )
        key_norm.weight.copy_(
            torch.tensor([1.1, 0.7], device=device, dtype=dtype)
        )

    query = torch.tensor(
        [[[-0.8, 0.2, 0.5, 1.1], [0.7, -0.3, 1.2, -0.4]]],
        device=device,
        dtype=dtype,
    )
    context = torch.tensor(
        [
            [
                [0.3, -0.7, 0.9, 0.1],
                [-0.2, 0.6, -0.5, 1.0],
                [1.1, 0.4, -0.9, -0.3],
            ]
        ],
        device=device,
        dtype=dtype,
    )
    mask = torch.tensor(
        [[[True, False, True], [False, True, True]]],
        device=device,
    )

    output = module(query=query, key_value=context, attn_mask=mask)
    expected = torch.tensor(
        _GOOGLE_OUTPUT[dtype],
        device=device,
        dtype=dtype,
    ).view_as(output)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 2e-3)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
