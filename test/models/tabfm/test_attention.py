import math

import pytest
import torch
import torch.nn.functional as F

from sdm.models.tabfm.attention import MultiheadAttention, _RMSNorm
from sdm.nn import RotaryEmbedding
from sdm.testing import withCUDA

# Generated with google-research/tabfm@b8a8b090c66d1b9e7af278003461582219996b6a
# using its PyTorch MultiheadAttention and the fixed state/input below.
_GOOGLE_OUTPUT = {
    torch.float32: [
        -0.2866353690624237,
        0.12349238246679306,
        0.45362013578414917,
        -0.1762521117925644,
        -0.2820703387260437,
        0.12766975164413452,
        0.4574098289012909,
        -0.1728500872850418,
    ],
    torch.bfloat16: [
        -0.28515625,
        0.12353515625,
        0.453125,
        -0.1767578125,
        -0.28125,
        0.1279296875,
        0.45703125,
        -0.1728515625,
    ],
}


def _load_google_fixture(module: MultiheadAttention) -> None:
    with torch.no_grad():
        base = torch.arange(
            16,
            device=module.per_dim_scale.device,
            dtype=torch.float32,
        ).view(4, 4)
        bias = torch.tensor(
            [-0.07, 0.03, 0.11, -0.05],
            device=module.per_dim_scale.device,
        )
        for index, name in enumerate(
            ("q_proj", "k_proj", "v_proj", "out_proj")
        ):
            layer = getattr(module, name)
            layer.weight.copy_((base - 7.5) * (0.013 + index * 0.002))
            layer.bias.copy_(bias * (index + 1))
        module.query_ln.weight.copy_(
            torch.tensor([0.8, 1.2], device=module.per_dim_scale.device)
        )
        module.key_ln.weight.copy_(
            torch.tensor([1.1, 0.7], device=module.per_dim_scale.device)
        )
        module.per_dim_scale.copy_(
            torch.tensor([-0.4, 0.6], device=module.per_dim_scale.device)
        )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_matches_pinned_google_output(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = MultiheadAttention(
        channels=4,
        num_heads=2,
        device=device,
        dtype=dtype,
    ).eval()
    _load_google_fixture(module)
    rope = RotaryEmbedding(
        channels=2,
        layout="interleaved",
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        rope.inv_freq.copy_(torch.tensor([0.37], device=device, dtype=dtype))

    query = torch.tensor(
        [[[-0.8, 0.2, 0.5, 1.1], [0.7, -0.3, 1.2, -0.4]]],
        device=device,
        dtype=dtype,
    )
    key = torch.tensor(
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
    value = torch.tensor(
        [
            [
                [-0.4, 1.0, 0.2, -0.6],
                [0.8, -0.1, 0.5, 0.3],
                [-0.7, 0.9, -0.2, 1.2],
            ]
        ],
        device=device,
        dtype=dtype,
    )
    attn_mask = torch.tensor(
        [[[True, False, True], [False, True, True]]],
        device=device,
    )

    with torch.no_grad():
        output = module(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            rope=rope,
        )
    expected = torch.tensor(
        _GOOGLE_OUTPUT[dtype],
        device=device,
        dtype=dtype,
    ).view(1, 2, 4)

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 2e-3)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_keeps_full_float32_path(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = _RMSNorm(channels=4, device=device, dtype=dtype)
    x = torch.tensor(
        [[0.11, -1.7, 2.3, -0.04], [4.2, 0.31, -0.8, 1.1]],
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        module.weight.copy_(
            torch.tensor([0.7, 1.3, -0.2, 2.1], device=device, dtype=dtype)
        )

    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    expected = (
        x_float * (variance + module.eps).rsqrt() * module.weight.float()
    ).to(dtype)

    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)


@withCUDA
def test_attention_mask_true_means_allowed(device: torch.device) -> None:
    module = MultiheadAttention(channels=2, num_heads=1, device=device)
    with torch.no_grad():
        module.q_proj.weight.zero_()
        module.q_proj.bias.zero_()
        module.k_proj.weight.zero_()
        module.k_proj.bias.zero_()
        module.v_proj.weight.copy_(torch.eye(2, device=device))
        module.v_proj.bias.zero_()
        module.out_proj.weight.copy_(torch.eye(2, device=device))
        module.out_proj.bias.zero_()

    query = torch.randn(1, 3, 2, device=device)
    key = torch.randn(1, 2, 2, device=device)
    value = torch.tensor([[[1.0, 10.0], [2.0, 20.0]]], device=device)

    first = module(
        query=query,
        key=key,
        value=value,
        attn_mask=torch.tensor([[[True, False]]], device=device),
    )
    second = module(
        query=query,
        key=key,
        value=value,
        attn_mask=torch.tensor([[[False, True]]], device=device),
    )

    torch.testing.assert_close(first, value[:, :1].expand_as(first))
    torch.testing.assert_close(second, value[:, 1:].expand_as(second))


def test_attention_broadcasts_batch_dimensions() -> None:
    torch.manual_seed(0)
    module = MultiheadAttention(channels=8, num_heads=2)
    query = torch.randn(1, 3, 8)
    key = torch.randn(2, 1, 5, 8)
    value = torch.randn(2, 1, 5, 8)
    attn_mask = torch.randint(0, 2, (2, 1, 3, 5), dtype=torch.bool)
    attn_mask[..., 0] = True

    output = module(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )
    expected = module(
        query=query.expand(2, 1, 3, 8),
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    assert output.shape == (2, 1, 3, 8)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("query_varying", [False, True])
def test_attention_accepts_google_singleton_head_mask(
    query_varying: bool,
) -> None:
    torch.manual_seed(0)
    module = MultiheadAttention(channels=8, num_heads=2)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)
    query_length = query.size(1) if query_varying else 1
    mask = torch.randint(0, 2, (2, query_length, 5), dtype=torch.bool)
    mask[..., 0] = True

    expected = module(query=query, key=key, value=value, attn_mask=mask)
    output = module(
        query=query,
        key=key,
        value=value,
        attn_mask=mask.unsqueeze(-3),
    )

    assert output.shape == expected.shape == query.shape
    torch.testing.assert_close(output, expected)


def test_attention_uses_checkpoint_compatible_state() -> None:
    module = MultiheadAttention(channels=8, num_heads=2)

    assert isinstance(module.query_ln, _RMSNorm)
    assert isinstance(module.key_ln, _RMSNorm)
    assert module.query_ln.eps == module.key_ln.eps == 1e-6
    assert module.sdpa.scale == 1.0
    actual_scale = (
        1.442695041 / math.sqrt(4) * F.softplus(module.per_dim_scale.float())
    )
    torch.testing.assert_close(actual_scale, torch.full((4,), 0.5))
    assert set(module.state_dict()) == {
        "per_dim_scale",
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
        "query_ln.weight",
        "key_ln.weight",
    }


@pytest.mark.parametrize(
    ("channels", "num_heads", "match"),
    [(0, 1, "channels"), (8, 0, "num_heads"), (8, 3, "num_heads")],
)
def test_attention_rejects_invalid_configuration(
    channels: int,
    num_heads: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MultiheadAttention(channels=channels, num_heads=num_heads)


def test_attention_compiles_fullgraph() -> None:
    torch.manual_seed(0)
    module = MultiheadAttention(channels=8, num_heads=2)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)

    expected = module(query=query, key=key, value=value, attn_mask=mask)
    compiled = torch.compile(module, fullgraph=True, backend="eager")
    output = compiled(query=query, key=key, value=value, attn_mask=mask)

    torch.testing.assert_close(output, expected)
