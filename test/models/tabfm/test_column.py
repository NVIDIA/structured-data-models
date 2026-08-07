import pytest
import torch
from test.models.tabfm._testing import frozen_google_state_dict
from torch import Tensor

from sdm.models.tabfm.column import (
    _InducedSelfAttentionBlock,
    _SetTransformer,
)
from sdm.testing import fullgraph, onlyFullTest, withCUDA

# Golden outputs from Google's PyTorch SetTransformer.
# Source revision: b8a8b090c66d1b9e7af278003461582219996b6a.
_OUTPUTS = {
    torch.float32: (
        "-6.6198606 -3.0886831 1.3817555 6.7914562 "
        "-4.9345789 -1.2790974 3.3261728 8.3812323 "
        "-5.9915938 -2.1584525 2.6241486 8.3562088 "
        "-4.2752519 -1.4229653 2.3797851 7.1329999 "
        "-5.0281920 -2.4866776 0.9490242 5.2789135 "
        "-6.4747896 -2.9525535 1.4608181 6.7653255 "
        "-5.1326904 -2.5910935 0.8448665 5.1751900 "
        "-6.3664818 -2.8421769 1.5734911 6.8805227"
    ),
    torch.bfloat16: (
        "-6.625 -3.09375 1.3828125 6.78125 "
        "-4.9375 -1.28125 3.328125 8.375 "
        "-6.0 -2.15625 2.625 8.375 "
        "-4.25 -1.421875 2.375 7.125 "
        "-5.0 -2.484375 0.953125 5.28125 "
        "-6.46875 -2.9375 1.4609375 6.75 "
        "-5.125 -2.59375 0.84765625 5.1875 "
        "-6.375 -2.84375 1.578125 6.875"
    ),
}


def _frozen_google_state(module: torch.nn.Module) -> dict[str, Tensor]:
    state = frozen_google_state_dict(module)
    for name, tensor in state.items():
        if not name.endswith("ind_vectors"):
            continue
        offset = (sum(name.encode()) % 5 - 2) * 0.02
        state[name] = (
            torch.linspace(
                -0.25,
                0.25,
                tensor.numel(),
                device=tensor.device,
            )
            .add_(offset)
            .reshape(tensor.shape)
        )
    return state


def _source(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(
        [
            [
                [-0.9, -0.6, -0.3, 0.0],
                [0.2, 0.5, 0.8, 0.6],
                [-0.7, -0.2, 0.3, 0.8],
                [0.9, 0.4, -0.1, -0.6],
            ],
            [
                [0.8, 0.3, -0.2, -0.7],
                [-0.6, -0.1, 0.4, 0.9],
                [0.7, 0.2, -0.3, -0.8],
                [-0.5, 0.0, 0.5, 1.0],
            ],
        ],
        device=device,
        dtype=dtype,
    )


def _mask(device: torch.device) -> Tensor:
    return torch.tensor(
        [[True, True, False, False], [True, False, True, False]],
        device=device,
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_set_transformer_matches_google(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = _SetTransformer(
        num_blocks=2,
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        num_inducing_points=3,
        device=device,
        dtype=dtype,
    )
    for block in module.blocks:
        assert isinstance(block, _InducedSelfAttentionBlock)
        assert torch.equal(
            block.ind_vectors,
            torch.zeros_like(block.ind_vectors),
        )
    state = _frozen_google_state(module)
    module.load_state_dict(state, strict=True)

    output = module(src=_source(device, dtype), attn_mask=_mask(device))

    expected = output.new_tensor(
        [float(value) for value in _OUTPUTS[dtype].split()]
    ).view_as(output)
    if dtype == torch.float32:
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    block = module.blocks[0]
    assert isinstance(block, _InducedSelfAttentionBlock)
    mab_keys = set(block.mab1.state_dict())
    block_keys = {"ind_vectors"}
    block_keys |= {f"mab{index}.{key}" for index in (1, 2) for key in mab_keys}
    expected_keys = {
        f"blocks.{index}.{key}" for index in range(2) for key in block_keys
    }
    assert set(module.state_dict()) == expected_keys


def test_induced_attention_obeys_context_mask_and_backpropagates() -> None:
    module = _InducedSelfAttentionBlock(4, 2, 8, 3)
    module.load_state_dict(_frozen_google_state(module), strict=True)
    source = torch.linspace(-1.0, 1.0, 20).view(1, 5, 4).requires_grad_()
    mask = torch.tensor([[True, True, False, False, False]])

    output = module(src=source, attn_mask=mask)
    query_changed = source.detach().clone()
    query_changed[:, 2].add_(10)
    query_changed_output = module(src=query_changed, attn_mask=mask)
    unchanged = torch.tensor([True, True, False, True, True])
    torch.testing.assert_close(
        output[:, unchanged], query_changed_output[:, unchanged]
    )
    assert not torch.allclose(output[:, 2], query_changed_output[:, 2])

    context_changed = source.detach().clone()
    context_changed[:, 0].add_(2)
    context_changed_output = module(src=context_changed, attn_mask=mask)
    assert not torch.allclose(output[:, 1], context_changed_output[:, 1])

    output.square().sum().backward()
    assert source.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_set_transformer_broadcasts_batch_mask() -> None:
    module = _SetTransformer(2, 4, 2, 8, 3)
    module.load_state_dict(_frozen_google_state(module), strict=True)
    source = torch.linspace(-1.0, 1.0, 96).view(2, 3, 4, 4)
    mask = _mask(torch.device("cpu")).unsqueeze(1)  # [2, 1, 4]

    output = module(src=source, attn_mask=mask)

    for index in range(source.size(1)):
        expected = module(src=source[:, index], attn_mask=mask[:, 0])
        torch.testing.assert_close(output[:, index], expected)


@onlyFullTest
def test_set_transformer_compiles_fullgraph() -> None:
    module = _SetTransformer(2, 4, 2, 8, 3)
    module.load_state_dict(_frozen_google_state(module), strict=True)
    source = torch.linspace(-1.0, 1.0, 96).view(2, 3, 4, 4)
    mask = _mask(torch.device("cpu")).unsqueeze(1)

    expected = module(src=source, attn_mask=mask)
    torch.testing.assert_close(fullgraph(module)(source, mask), expected)


def test_induced_attention_rejects_empty_configuration() -> None:
    with pytest.raises(ValueError, match="num_inducing_points"):
        _InducedSelfAttentionBlock(4, 2, 8, 0)
    with pytest.raises(ValueError, match="num_blocks"):
        _SetTransformer(0, 4, 2, 8, 3)
