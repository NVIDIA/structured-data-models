import copy

import pytest
import torch
from test.models.tabfm._testing import frozen_google_state_dict
from torch import Tensor

from sdm.models.tabfm.column import _ColumnEmbedding
from sdm.testing import fullgraph, onlyFullTest, withCUDA

# Golden outputs from Google's PyTorch ColEmbedding.
# Source revision: b8a8b090c66d1b9e7af278003461582219996b6a.
_OUTPUTS = {
    torch.float32: (
        "-0.1257782 0.5771171 1.4352868 2.4487295 "
        "-0.2586916 0.4869585 1.4006101 2.4822628 "
        "-0.3558666 0.4188041 1.3702981 2.4986148 "
        "-0.4391733 0.3588288 1.3408818 2.5069859 "
        "-0.4934367 0.3189725 1.3199701 2.5095553 "
        "-0.5409067 0.2835808 1.3005122 2.5098872 "
        "-0.6841823 0.1736588 1.2349101 2.4995713 "
        "-0.7214430 0.1442726 1.2160776 2.4939713 "
        "-0.7474773 0.1235348 1.2024640 2.4893095 "
        "-0.7805479 0.0969425 1.1846172 2.4824760 "
        "-0.8031039 0.0786411 1.1720816 2.4772172 "
        "-0.8316552 0.0552802 1.1557816 2.4698486"
    ),
    torch.bfloat16: (
        "-0.1328125 0.578125 1.4375 2.453125 "
        "-0.263671875 0.490234375 1.40625 2.484375 "
        "-0.361328125 0.41796875 1.3671875 2.5 "
        "-0.4453125 0.359375 1.34375 2.5 "
        "-0.5 0.318359375 1.3203125 2.5 "
        "-0.54296875 0.28515625 1.3046875 2.5 "
        "-0.6875 0.1748046875 1.234375 2.5 "
        "-0.7265625 0.1455078125 1.21875 2.5 "
        "-0.75 0.123046875 1.203125 2.484375 "
        "-0.78125 0.09765625 1.1875 2.484375 "
        "-0.80859375 0.07861328125 1.171875 2.46875 "
        "-0.8359375 0.056884765625 1.15625 2.46875"
    ),
}


def _frozen_google_state(module: torch.nn.Module) -> dict[str, Tensor]:
    state = frozen_google_state_dict(module)
    for name, tensor in state.items():
        if name.endswith("ind_vectors"):
            offset = (sum(name.encode()) % 5 - 2) * 0.02
            start, end = -0.25, 0.25
        elif name == "ln_w.weight":
            offset = (sum(name.encode()) % 7) * 0.05
            start, end = 0.8, 1.2
        else:
            continue
        state[name] = (
            torch.linspace(
                start,
                end,
                tensor.numel(),
                device=tensor.device,
            )
            .add_(offset)
            .reshape(tensor.shape)
        )
    return state


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_column_embedding_matches_google(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = _ColumnEmbedding(
        channels=4,
        num_blocks=2,
        num_heads=2,
        feedforward_channels=6,
        num_inducing_points=3,
        col_chunk_size=2,
        device=device,
        dtype=dtype,
    )
    module.load_state_dict(_frozen_google_state(module), strict=True)
    x = torch.linspace(-0.95, 0.95, 48, device=device, dtype=dtype).view(
        2, 3, 2, 4
    )
    context_size = torch.tensor([1, 2], device=device)

    output = module(x=x, context_size=context_size)

    expected = output.new_tensor(
        [float(value) for value in _OUTPUTS[dtype].split()]
    ).view_as(output)
    if dtype == torch.float32:
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    expected_keys = {f"tf_col.{key}" for key in module.tf_col.state_dict()}
    expected_keys |= {"out_w.weight", "out_w.bias", "ln_w.weight"}
    assert set(module.state_dict()) == expected_keys


def test_column_embedding_isolates_query_rows_and_columns() -> None:
    module = _ColumnEmbedding(4, 1, 2, 8, 3, col_chunk_size=None)
    module.load_state_dict(_frozen_google_state(module), strict=True)
    x = torch.linspace(-1.0, 1.0, 32).view(1, 4, 2, 4)
    context_size = torch.tensor([2])

    output = module(x=x, context_size=context_size)
    query_changed = x.clone()
    query_changed[:, 2, 0].add_(10)
    query_output = module(x=query_changed, context_size=context_size)

    torch.testing.assert_close(output[:, :, 1], query_output[:, :, 1])
    unchanged_rows = torch.tensor([True, True, False, True])
    torch.testing.assert_close(
        output[:, unchanged_rows, 0],
        query_output[:, unchanged_rows, 0],
    )
    assert not torch.allclose(output[:, 2, 0], query_output[:, 2, 0])

    context_changed = x.clone()
    context_changed[:, 0, 0].add_(2)
    context_output = module(x=context_changed, context_size=context_size)
    assert not torch.allclose(output[:, 3, 0], context_output[:, 3, 0])
    torch.testing.assert_close(output[:, :, 1], context_output[:, :, 1])


def test_column_chunking_preserves_outputs_and_gradients() -> None:
    unchunked = _ColumnEmbedding(4, 1, 2, 8, 3, col_chunk_size=None)
    unchunked.load_state_dict(_frozen_google_state(unchunked), strict=True)
    chunked = copy.deepcopy(unchunked)
    chunked.col_chunk_size = 4
    unchunked_input = torch.linspace(-1.0, 1.0, 72).view(2, 3, 3, 4)
    unchunked_input.requires_grad_()
    chunked_input = unchunked_input.detach().clone().requires_grad_()
    context_size = torch.tensor([1, 2])

    unchunked_output = unchunked(unchunked_input, context_size)
    chunked_output = chunked(chunked_input, context_size)
    unchunked_output.square().sum().backward()
    chunked_output.square().sum().backward()

    torch.testing.assert_close(chunked_output, unchunked_output)
    assert unchunked_input.grad is not None
    assert chunked_input.grad is not None
    torch.testing.assert_close(chunked_input.grad, unchunked_input.grad)
    for (expected_name, expected_parameter), (name, parameter) in zip(
        unchunked.named_parameters(),
        chunked.named_parameters(),
        strict=True,
    ):
        assert name == expected_name
        assert expected_parameter.grad is not None
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected_parameter.grad)


@onlyFullTest
def test_column_embedding_compiles_with_chunks() -> None:
    module = _ColumnEmbedding(4, 1, 2, 8, 3, col_chunk_size=2)
    module.load_state_dict(_frozen_google_state(module), strict=True)
    x = torch.linspace(-1.0, 1.0, 48).view(2, 3, 2, 4)
    context_size = torch.tensor([1, 2])

    expected = module(x, context_size)
    torch.testing.assert_close(fullgraph(module)(x, context_size), expected)


def test_column_embedding_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="col_chunk_size"):
        _ColumnEmbedding(4, 1, 2, 8, 3, col_chunk_size=0)

    module = _ColumnEmbedding(4, 1, 2, 8, 3)
    assert module.col_chunk_size == 16
    x = torch.randn(2, 3, 2, 4)
    with pytest.raises(ValueError, match="context_size"):
        module(x, torch.ones(2, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="context_size"):
        module(x, torch.ones(2))
