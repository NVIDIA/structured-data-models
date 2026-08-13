import copy

import pytest
import torch
import torch.nn.functional as F

from sdm.nn import ChunkedFeedForward, SwiGLUFeedForward
from sdm.testing import withCUDA


def test_swiglu_feedforward() -> None:
    module = SwiGLUFeedForward(channels=4, feedforward_channels=7)
    tensor = torch.randn(2, 3, 4)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(1)

    projected = tensor.sum(dim=-1, keepdim=True) + 1
    hidden = F.silu(projected) * projected
    expected = (7 * hidden + 1).expand_as(tensor)
    output = module(tensor)

    torch.testing.assert_close(output, expected)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [0, 5])
def test_chunked_feedforward_output_and_gradients(
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    unchunked = ChunkedFeedForward(
        SwiGLUFeedForward(
            channels=4,
            feedforward_channels=7,
            device=device,
            dtype=dtype,
        ),
        chunk_size=None,
    )
    chunked = ChunkedFeedForward(
        copy.deepcopy(unchunked.module),
        chunk_size=3,
    )
    expected_input = torch.randn(
        2,
        num_tokens,
        4,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    tensor = expected_input.detach().clone().requires_grad_()
    expected = unchunked(expected_input)
    output = chunked(tensor)

    torch.testing.assert_close(output, expected)
    expected.square().sum().backward()
    output.square().sum().backward()
    torch.testing.assert_close(tensor.grad, expected_input.grad)
    expected_parameters = dict(unchunked.named_parameters())
    atol, rtol = (2e-3, 2e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    for name, parameter in chunked.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            expected_parameters[name].grad,
            atol=atol,
            rtol=rtol,
        )
