import copy

import pytest
import torch
from test.models.tabfm._testing import frozen_google_state_dict
from torch import Tensor

from sdm.models.tabfm.attention import _Encoder, _RMSNorm
from sdm.models.tabfm.row import _RowInteraction
from sdm.testing import fullgraph, onlyFullTest, withCUDA

# Generated with google-research/tabfm@b8a8b090c66d1b9e7af278003461582219996b6a
# using its PyTorch RowInteraction and the fixed state/input below.
_GOOGLE_FULL_OUTPUTS = {
    torch.float32: (
        "-0.48183170 1.53545964 -0.24401887 1.65711296 "
        "-0.34009823 1.61856401 -0.20027296 1.67025566 "
        "-0.34751529 1.61501539 -0.36660564 1.60549104 "
        "-0.17629908 1.67632532 -0.37937042 1.59880447 "
        "-0.17635179 1.67631292 -0.47907764 1.53739727 "
        "-0.42534956 1.57255363 -0.27208564 1.64724958"
    ),
    torch.bfloat16: (
        "-0.48046875 1.53906250 -0.24511719 1.66406250 "
        "-0.34179688 1.62500000 -0.19921875 1.67187500 "
        "-0.34960938 1.61718750 -0.36718750 1.60937500 "
        "-0.17480469 1.67968750 -0.38085938 1.60156250 "
        "-0.17578125 1.67968750 -0.48046875 1.53906250 "
        "-0.42578125 1.57812500 -0.27148438 1.64843750"
    ),
}


def _google_name(name: str) -> str:
    if name == "tf_row.rope.inv_freq":
        return "tf_row.rope.freqs"
    return name


def _google_input(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(
        [
            [
                [[-0.8, 0.1], [0.3, 0.9], [-0.4, 0.7]],
                [[0.6, -0.2], [-0.5, 1.0], [0.2, -0.9]],
            ],
            [
                [[0.7, -0.3], [-0.6, 0.8], [0.4, -0.1]],
                [[-0.9, 0.5], [0.1, -0.7], [0.8, 0.2]],
            ],
        ],
        device=device,
        dtype=dtype,
    )


def _row_interaction(
    *,
    output_full: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    row_chunk_size: int | None = None,
) -> _RowInteraction:
    return _RowInteraction(
        num_blocks=2,
        channels=2,
        num_heads=1,
        feedforward_channels=3,
        num_cls=1,
        output_full=output_full,
        row_chunk_size=row_chunk_size,
        device=device,
        dtype=dtype,
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_full", [True, False])
def test_row_interaction_matches_google(
    device: torch.device,
    dtype: torch.dtype,
    output_full: bool,
) -> None:
    module = _row_interaction(
        output_full=output_full,
        device=device,
        dtype=dtype,
    ).eval()
    module.load_state_dict(frozen_google_state_dict(module, rope_value=0.37))
    x = _google_input(device, dtype)
    active_features = torch.tensor([2, 1], device=device)

    output = module(x, active_features)
    expected = output.new_tensor(
        [float(value) for value in _GOOGLE_FULL_OUTPUTS[dtype].split()]
    ).view(2, 2, 3, 2)
    if not output_full:
        expected = expected[:, :, :1].reshape(2, 2, 2)

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 2e-3)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


def test_row_interaction_uses_google_checkpoint_paths() -> None:
    module = _row_interaction(output_full=True)

    assert isinstance(module.tf_row, _Encoder)
    assert isinstance(module.out_ln, _RMSNorm)
    expected_keys = {f"tf_row.{key}" for key in module.tf_row.state_dict()}
    expected_keys.add("out_ln.weight")
    assert set(module.state_dict()) == expected_keys

    google_keys = {_google_name(key) for key in expected_keys}
    assert google_keys.symmetric_difference(expected_keys) == {
        "tf_row.rope.freqs",
        "tf_row.rope.inv_freq",
    }
    assert module.tf_row.rope is not None
    assert not module.tf_row.rope.inv_freq.requires_grad


@pytest.mark.parametrize("output_full", [True, False])
@pytest.mark.parametrize("num_rows", [3, 0])
def test_row_chunking_preserves_outputs_and_gradients(
    output_full: bool,
    num_rows: int,
) -> None:
    unchunked = _RowInteraction(
        num_blocks=1,
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        num_cls=1,
        output_full=output_full,
        row_chunk_size=None,
    )
    chunked = copy.deepcopy(unchunked)
    chunked.row_chunk_size = 4
    unchunked_input = torch.randn(
        2,
        num_rows,
        4,
        4,
        requires_grad=True,
    )
    chunked_input = unchunked_input.detach().clone().requires_grad_()
    active_features = torch.tensor([3, 1])

    expected = unchunked(unchunked_input, active_features)
    output = chunked(chunked_input, active_features)
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


def test_padded_features_are_keys_only_when_active() -> None:
    module = _RowInteraction(
        num_blocks=2,
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        num_cls=1,
        output_full=True,
        row_chunk_size=2,
    ).eval()
    x = torch.randn(2, 3, 5, 4)
    active_features = torch.tensor([3, 1])
    changed = x.clone()
    changed[0, :, 4:] += 100
    changed[1, :, 2:] -= 100

    output = module(x, active_features)
    changed_output = module(changed, active_features)

    torch.testing.assert_close(output[0, :, :4], changed_output[0, :, :4])
    torch.testing.assert_close(output[1, :, :2], changed_output[1, :, :2])
    assert not torch.equal(output[0, :, 4:], changed_output[0, :, 4:])
    assert not torch.equal(output[1, :, 2:], changed_output[1, :, 2:])

    module.output_full = False
    torch.testing.assert_close(
        module(x, active_features),
        module(changed, active_features),
    )


@onlyFullTest
@pytest.mark.parametrize("output_full", [True, False])
def test_row_interaction_compiles_fullgraph(output_full: bool) -> None:
    module = _RowInteraction(
        num_blocks=1,
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        num_cls=1,
        output_full=output_full,
        row_chunk_size=2,
    )
    x = torch.randn(2, 3, 4, 4)
    active_features = torch.tensor([3, 1])

    expected = module(x, active_features)
    torch.testing.assert_close(fullgraph(module)(x, active_features), expected)


def test_row_interaction_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="num_cls"):
        _RowInteraction(1, 4, 2, 6, num_cls=0)
    with pytest.raises(ValueError, match="row_chunk_size"):
        _RowInteraction(1, 4, 2, 6, num_cls=1, row_chunk_size=0)

    module = _RowInteraction(1, 4, 2, 6, num_cls=2)
    with pytest.raises(ValueError, match="CLS token prefix"):
        module(torch.randn(1, 2, 1, 4))
    for active_features in (
        torch.ones(1, 1, dtype=torch.long),
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
    ):
        with pytest.raises(ValueError, match="active_features"):
            module(torch.randn(1, 2, 3, 4), active_features)
