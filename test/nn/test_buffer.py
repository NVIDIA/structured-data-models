import copy

import torch

from sdm.nn._buffer import BufferList
from sdm.testing import withCUDA


class _TensorSubclass(torch.Tensor):
    pass


def test_buffer_list_is_an_indexed_collection() -> None:
    buffers = BufferList(
        [
            BufferList([torch.tensor([1.0, 2.0]), torch.tensor([3.0])]),
            BufferList(),
            torch.tensor([5.0]),
        ]
    )
    inner = buffers[0]
    empty = buffers[1]

    assert len(buffers) == 3
    assert len(inner) == 2
    assert len(empty) == 0
    direct = buffers[2]
    assert isinstance(direct, torch.Tensor)
    assert direct.equal(torch.tensor([5.0]))


def test_buffer_list_roundtrips_state_dict() -> None:
    source = BufferList(
        [
            BufferList([torch.tensor([1.0]), torch.tensor([2.0])]),
            BufferList(),
            torch.tensor([5.0]),
        ]
    )
    restored = BufferList()

    restored.load_state_dict(source.state_dict())

    assert len(restored) == 3
    first = restored[0]
    assert isinstance(first, BufferList)
    assert len(first) == 2
    first_value = first[0]
    second_value = first[1]
    assert isinstance(first_value, torch.Tensor)
    assert isinstance(second_value, torch.Tensor)
    assert first_value.equal(torch.tensor([1.0]))
    assert second_value.equal(torch.tensor([2.0]))
    assert len(restored[1]) == 0
    direct = restored[2]
    assert isinstance(direct, torch.Tensor)
    assert direct.equal(torch.tensor([5.0]))


def test_buffer_list_loads_tensor_subclasses() -> None:
    tensor = torch.tensor([1.0]).as_subclass(_TensorSubclass)
    source = BufferList([tensor])
    restored = BufferList()

    restored.load_state_dict(source.state_dict())

    loaded = restored[0]
    assert type(loaded) is _TensorSubclass
    assert torch.equal(loaded, tensor)
    loaded.add_(1)
    original = source[0]
    assert isinstance(original, torch.Tensor)
    assert loaded.equal(torch.tensor([2.0]))
    assert original.equal(torch.tensor([1.0]))


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


@withCUDA
def test_buffer_list_moves_device_and_dtype(
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


def test_buffer_list_deepcopy_has_independent_storage() -> None:
    buffers = BufferList([torch.tensor([1.0, 2.0])])

    cloned = copy.deepcopy(buffers)
    cloned_tensor = cloned[0]
    assert isinstance(cloned_tensor, torch.Tensor)
    cloned_tensor.add_(1)

    original = buffers[0]
    assert isinstance(original, torch.Tensor)
    assert torch.equal(original, torch.tensor([1.0, 2.0]))
    assert torch.equal(cloned_tensor, torch.tensor([2.0, 3.0]))
