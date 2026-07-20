import copy

import pytest
import torch
from test.models.tabfm._testing import frozen_google_state_dict
from torch import Tensor

from sdm.models.tabfm.attention import (
    _Encoder,
    _MultiheadAttentionBlock,
)
from sdm.testing import fullgraph, onlyFullTest, withCUDA

# Golden outputs from Google's PyTorch MultiheadAttentionBlock and Encoder.
# Source revision: b8a8b090c66d1b9e7af278003461582219996b6a.
_BLOCK_OUTPUTS = {
    torch.float32: (
        "-3.1050215 -1.1874523 1.1965286 4.0469208 "
        "-1.6031406 -0.6675859 0.7346196 2.6034760 "
        "-1.6469489 -0.7462716 0.6274487 2.4742122 "
        "-3.0659342 -1.1470807 1.2414379 4.0996213"
    ),
    torch.bfloat16: (
        "-3.09375 -1.1875 1.203125 4.0625 "
        "-1.59375 -0.66796875 0.734375 2.609375 "
        "-1.640625 -0.7421875 0.6328125 2.5 "
        "-3.0625 -1.1484375 1.2421875 4.09375"
    ),
}

_ENCODER_OUTPUTS = {
    (False, torch.float32): (
        "-5.2946734 -1.7140485 2.7826867 8.1955328 "
        "-1.7942053 -0.0415165 2.3072815 5.2521877 "
        "-2.6994452 -0.3522018 2.8018029 6.7625694 "
        "-5.4566631 -1.8841162 2.6058195 8.0131435"
    ),
    (False, torch.bfloat16): (
        "-5.28125 -1.71875 2.78125 8.1875 "
        "-1.7734375 -0.0234375 2.3125 5.25 "
        "-2.6875 -0.341796875 2.8125 6.75 "
        "-5.46875 -1.890625 2.609375 8.0"
    ),
    (True, torch.float32): (
        "-5.2946734 -1.7140485 2.7826867 8.1955328 "
        "-1.5858865 0.1364819 2.4388621 5.3212528 "
        "-3.0016642 -0.5592161 2.7286716 6.8619995 "
        "-5.4566631 -1.8841162 2.6058195 8.0131435"
    ),
    (True, torch.bfloat16): (
        "-5.28125 -1.71875 2.78125 8.1875 "
        "-1.5625 0.1484375 2.4375 5.28125 "
        "-3.0 -0.5546875 2.734375 6.84375 "
        "-5.46875 -1.890625 2.609375 8.0"
    ),
}


def _assert_close_to_golden(
    output: Tensor,
    values: str,
) -> None:
    expected = output.new_tensor([float(value) for value in values.split()])
    expected = expected.view_as(output)
    if output.dtype == torch.float32:
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_block_matches_google(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = _MultiheadAttentionBlock(
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        device=device,
        dtype=dtype,
    )
    module.load_state_dict(frozen_google_state_dict(module))
    norms = (
        module.pre_attn_ln,
        module.post_attn_ln,
        module.pre_ff_ln,
        module.post_ff_ln,
    )
    assert all(type(norm) is type(module.attn.query_ln) for norm in norms)
    assert all(norm.eps == 1e-6 for norm in norms)
    query = torch.tensor(
        [
            [[-0.8, -0.3, 0.2, 0.7], [0.6, 0.1, -0.4, -0.9]],
            [[0.9, 0.4, -0.1, -0.6], [-0.7, -0.2, 0.3, 0.8]],
        ],
        device=device,
        dtype=dtype,
    )
    key = torch.tensor(
        [
            [[0.2, -0.1, 0.4, -0.3], [-0.8, 0.7, -0.6, 0.5]],
            [[0.5, -0.4, 0.3, -0.2], [-0.1, 0.6, -0.7, 0.8]],
        ],
        device=device,
        dtype=dtype,
    )
    value = torch.tensor(
        [
            [[-0.3, 0.1, 0.5, 0.9], [0.8, 0.4, 0.0, -0.4]],
            [[0.7, 0.2, -0.3, -0.8], [-0.5, 0.0, 0.5, 1.0]],
        ],
        device=device,
        dtype=dtype,
    )
    mask = torch.tensor(
        [[[True, False], [True, True]], [[True, True], [False, True]]],
        device=device,
    )

    output = module(query=query, key=key, value=value, attn_mask=mask)

    _assert_close_to_golden(output, _BLOCK_OUTPUTS[dtype])
    torch.testing.assert_close(
        module(query=query, key=key),
        module(query=query, key=key, value=query),
    )


@pytest.mark.parametrize("num_tokens", [5, 0])
def test_feedforward_chunking_preserves_gradients(num_tokens: int) -> None:
    unchunked = _MultiheadAttentionBlock(
        channels=8,
        num_heads=2,
        feedforward_channels=16,
    )
    chunked = copy.deepcopy(unchunked)
    chunked.ffn_chunk_size = 3
    unchunked_input = torch.randn(2, num_tokens, 8, requires_grad=True)
    chunked_input = unchunked_input.detach().clone().requires_grad_()

    expected = unchunked(unchunked_input)
    output = chunked(chunked_input)
    expected.square().sum().backward()
    output.square().sum().backward()

    torch.testing.assert_close(output, expected)
    assert chunked_input.grad is not None
    assert unchunked_input.grad is not None
    torch.testing.assert_close(chunked_input.grad, unchunked_input.grad)
    for (expected_name, expected_parameter), (name, parameter) in zip(
        unchunked.named_parameters(),
        chunked.named_parameters(),
        strict=True,
    ):
        assert name == expected_name
        if expected_parameter.grad is None:
            assert parameter.grad is None
            continue
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected_parameter.grad)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rope_theta", [None, 100_000.0])
def test_encoder_matches_google(
    device: torch.device,
    dtype: torch.dtype,
    rope_theta: float | None,
) -> None:
    module = _Encoder(
        num_blocks=2,
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )
    module.load_state_dict(frozen_google_state_dict(module))
    x = torch.tensor(
        [
            [[-0.7, -0.2, 0.3, 0.8], [0.6, 0.1, -0.4, -0.9]],
            [[0.9, 0.4, -0.1, -0.6], [-0.8, -0.3, 0.2, 0.7]],
        ],
        device=device,
        dtype=dtype,
    )
    mask = torch.tensor(
        [[[True, False], [True, True]], [[True, True], [False, True]]],
        device=device,
    )

    output = module(x=x, attn_mask=mask)

    _assert_close_to_golden(
        output,
        _ENCODER_OUTPUTS[(rope_theta is not None, dtype)],
    )
    state_keys = set(module.state_dict())
    rope_keys = {key for key in state_keys if key.startswith("rope.")}
    if rope_theta is not None:
        assert rope_keys == {"rope.inv_freq"}
        assert module.rope is not None
        assert not module.rope.inv_freq.requires_grad
        torch.testing.assert_close(
            module.rope.inv_freq,
            module.rope.inv_freq.new_full((1,), 0.75),
        )
    else:
        assert rope_keys == set()
    assert "rope.freqs" not in state_keys


@onlyFullTest
def test_attention_block_compiles_with_chunking() -> None:
    module = _MultiheadAttentionBlock(
        channels=8,
        num_heads=2,
        feedforward_channels=16,
        ffn_chunk_size=3,
    )
    query = torch.randn(2, 5, 8)

    expected = module(query)
    torch.testing.assert_close(fullgraph(module)(query), expected)


def test_attention_blocks_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="feedforward_channels"):
        _MultiheadAttentionBlock(8, 2, 0)
    with pytest.raises(ValueError, match="ffn_chunk_size"):
        _MultiheadAttentionBlock(8, 2, 16, ffn_chunk_size=0)
    with pytest.raises(ValueError, match="num_blocks"):
        _Encoder(0, 8, 2, 16)
    with pytest.raises(ValueError, match="rope_theta"):
        _Encoder(1, 8, 2, 16, rope_theta=0)
