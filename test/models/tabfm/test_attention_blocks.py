import pytest
import torch
from test.models.tabfm._testing import google_state_dict

from sdm.models.tabfm.attention import _Encoder
from sdm.testing import withCUDA

_GOOGLE_OUTPUT = {
    torch.float32: (
        "-5.2946734 -1.7140485 2.7826867 8.1955328 "
        "-1.5858865 0.1364819 2.4388621 5.3212528 "
        "-3.0016642 -0.5592161 2.7286716 6.8619995 "
        "-5.4566631 -1.8841162 2.6058195 8.0131435"
    ),
    torch.bfloat16: (
        "-5.28125 -1.71875 2.78125 8.1875 "
        "-1.5625 0.1484375 2.4375 5.28125 "
        "-3.0 -0.5546875 2.734375 6.84375 "
        "-5.46875 -1.890625 2.609375 8.0"
    ),
}


def _google_value(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name.endswith("rope.freqs"):
        return tensor.new_full(tensor.shape, 0.75)
    if name.endswith("_ln.weight"):
        offset = (sum(name.encode()) % 7) * 0.05
        return (
            torch.linspace(
                0.8,
                1.2,
                tensor.numel(),
                device=tensor.device,
            )
            .add_(offset)
            .reshape_as(tensor)
        )
    if name.endswith("per_dim_scale"):
        return torch.linspace(
            -0.3,
            0.2,
            tensor.numel(),
            device=tensor.device,
        ).reshape_as(tensor)
    offset = (sum(name.encode()) % 11 - 5) * 0.003
    start, end = (-0.04, 0.04) if name.endswith(".bias") else (-0.15, 0.15)
    return (
        torch.linspace(
            start,
            end,
            tensor.numel(),
            device=tensor.device,
        )
        .add_(offset)
        .reshape_as(tensor)
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_encoder_matches_google(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = _Encoder(
        num_blocks=2,
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        ffn_chunk_size=3,
        device=device,
        dtype=dtype,
    )
    module.load_state_dict(google_state_dict(module, _google_value))
    tensor = torch.tensor(
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

    output = module(tensor, mask)
    expected = output.new_tensor(
        [float(value) for value in _GOOGLE_OUTPUT[dtype].split()]
    ).view_as(output)
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
