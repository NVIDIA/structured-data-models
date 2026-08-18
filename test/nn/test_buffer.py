import copy

import torch

from sdm import StringTensor
from sdm.nn._buffer import BufferList
from sdm.testing import onlyCUDA, withCUDA


class _TensorSubclass(torch.Tensor):
    pass


def test_buffer_list_is_an_indexed_buffer_collection() -> None:
    buffers = BufferList(
        [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0]),
        ]
    )

    assert len(buffers) == 2
    assert tuple(buffers.state_dict()) == ("0", "1", "_extra_state")
    first = buffers[0]
    second = buffers[1]
    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert first.equal(torch.tensor([1.0, 2.0]))
    assert second.equal(torch.tensor([3.0]))


def test_buffer_list_loads_tensor_subclasses() -> None:
    tensor = torch.tensor([1.0]).as_subclass(_TensorSubclass)
    source = BufferList([tensor])
    restored = BufferList()

    restored.load_state_dict(source.state_dict())

    loaded = restored[0]
    original = source[0]
    assert type(loaded) is _TensorSubclass
    assert isinstance(original, torch.Tensor)
    assert torch.equal(loaded, tensor)
    loaded.add_(1)
    assert torch.equal(original, tensor)


def test_buffer_list_loads_nested_string_tensor() -> None:
    source = BufferList(
        [BufferList([StringTensor.from_list(["red", "blue"])])]
    )
    restored = BufferList()

    restored.load_state_dict(source.state_dict())

    nested = restored[0]
    assert isinstance(nested, BufferList)
    values = nested[0]
    assert isinstance(values, StringTensor)
    assert values.tolist() == ["red", "blue"]


@withCUDA
def test_buffer_list_preserves_existing_buffer_on_load(
    device: torch.device,
) -> None:
    buffer = torch.empty(1, dtype=torch.float64, device=device)
    restored = BufferList([buffer])

    restored.load_state_dict(BufferList([torch.tensor([3.0])]).state_dict())

    loaded = restored[0]
    assert isinstance(loaded, torch.Tensor)
    assert loaded is buffer
    assert loaded.dtype == torch.float64
    assert loaded.device == device
    assert loaded.equal(
        torch.tensor([3.0], dtype=torch.float64, device=device)
    )


def test_buffer_list_registers_as_a_nested_module() -> None:
    module = torch.nn.Module()
    module.buffer_list = BufferList([torch.tensor([1.0])])

    assert tuple(module.state_dict()) == (
        "buffer_list.0",
        "buffer_list._extra_state",
    )

    restored = torch.nn.Module()
    restored.buffer_list = BufferList()
    restored.load_state_dict(module.state_dict())
    loaded = restored.buffer_list[0]
    assert isinstance(loaded, torch.Tensor)
    assert torch.equal(loaded, torch.tensor([1.0]))


@onlyCUDA
def test_buffer_list_cuda_and_cpu() -> None:
    buffers = BufferList([torch.tensor([1.0])]).cuda()

    cuda_buffer = buffers[0]
    assert isinstance(cuda_buffer, torch.Tensor)
    assert cuda_buffer.is_cuda
    buffers.cpu()
    cpu_buffer = buffers[0]
    assert isinstance(cpu_buffer, torch.Tensor)
    assert not cpu_buffer.is_cuda


@withCUDA
def test_buffer_list_moves_device_and_floating_dtype(
    device: torch.device,
) -> None:
    buffers = BufferList(
        [
            torch.tensor([1.0]),
            torch.tensor([2], dtype=torch.long),
        ]
    ).to(device=device, dtype=torch.float64)

    floating = buffers[0]
    integer = buffers[1]
    assert isinstance(floating, torch.Tensor)
    assert isinstance(integer, torch.Tensor)
    assert floating.device == device
    assert floating.dtype == torch.float64
    assert integer.device == device
    assert integer.dtype == torch.long


def test_buffer_list_deepcopy_has_independent_storage() -> None:
    buffers = BufferList([torch.tensor([1.0, 2.0])])

    cloned = copy.deepcopy(buffers)
    cloned_buffer = cloned[0]
    original_buffer = buffers[0]
    assert isinstance(cloned_buffer, torch.Tensor)
    assert isinstance(original_buffer, torch.Tensor)
    cloned_buffer.add_(1)

    assert torch.equal(original_buffer, torch.tensor([1.0, 2.0]))
    assert torch.equal(cloned_buffer, torch.tensor([2.0, 3.0]))


def test_buffer_list_loads_nested_uneven_and_empty_lists() -> None:
    source = BufferList(
        [
            BufferList([torch.tensor([1.0]), torch.tensor([2.0])]),
            BufferList(),
            BufferList([torch.tensor([3.0])]),
        ]
    )
    restored = BufferList()

    restored.load_state_dict(source.state_dict())

    assert len(restored) == 3
    empty = restored[1]
    first = restored[0]
    last = restored[2]
    assert isinstance(empty, BufferList)
    assert isinstance(first, BufferList)
    assert isinstance(last, BufferList)
    assert len(empty) == 0
    first_value = first[0]
    second_value = first[1]
    last_value = last[0]
    assert isinstance(first_value, torch.Tensor)
    assert isinstance(second_value, torch.Tensor)
    assert isinstance(last_value, torch.Tensor)
    assert torch.equal(first_value, torch.tensor([1.0]))
    assert torch.equal(second_value, torch.tensor([2.0]))
    assert torch.equal(last_value, torch.tensor([3.0]))


@withCUDA
def test_buffer_list_moves_nested_state(
    device: torch.device,
) -> None:
    buffers = BufferList(
        [
            torch.tensor([1.0]),
            BufferList(
                [
                    torch.tensor([2.0]),
                    torch.tensor([3], dtype=torch.long),
                ]
            ),
        ]
    ).to(device=device, dtype=torch.float64)

    direct = buffers[0]
    nested = buffers[1]
    assert isinstance(direct, torch.Tensor)
    assert isinstance(nested, BufferList)
    assert direct.device == device
    assert direct.dtype == torch.float64
    nested_float = nested[0]
    nested_long = nested[1]
    assert isinstance(nested_float, torch.Tensor)
    assert isinstance(nested_long, torch.Tensor)
    assert nested_float.device == device
    assert nested_float.dtype == torch.float64
    assert nested_long.device == device
    assert nested_long.dtype == torch.long


def test_buffer_list_deepcopy_has_independent_nested_storage() -> None:
    buffers = BufferList([BufferList([torch.tensor([1.0, 2.0])])])

    cloned = copy.deepcopy(buffers)
    cloned_nested = cloned[0]
    assert isinstance(cloned_nested, BufferList)
    cloned_tensor = cloned_nested[0]
    assert isinstance(cloned_tensor, torch.Tensor)
    cloned_tensor.add_(1)

    nested = buffers[0]
    assert isinstance(nested, BufferList)
    original = nested[0]
    assert isinstance(original, torch.Tensor)
    assert torch.equal(original, torch.tensor([1.0, 2.0]))
    assert torch.equal(cloned_tensor, torch.tensor([2.0, 3.0]))
