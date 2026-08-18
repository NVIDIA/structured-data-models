import copy

import torch

from sdm import StringTensor
from sdm.nn._buffer import BufferList
from sdm.testing import onlyCUDA, withCUDA


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

    assert len(buffers) == 3
    assert isinstance(buffers[0], BufferList)
    assert isinstance(buffers[1], BufferList)
    assert isinstance(buffers[2], torch.Tensor)
    assert len(buffers[0]) == 2
    assert len(buffers[1]) == 0
    assert buffers[0][0].equal(torch.tensor([1.0, 2.0]))
    assert buffers[0][1].equal(torch.tensor([3.0]))
    assert buffers[2].equal(torch.tensor([5.0]))


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

    assert isinstance(restored[0], BufferList)
    assert isinstance(restored[1], BufferList)
    assert isinstance(restored[2], torch.Tensor)
    assert restored[0][0].equal(source[0][0])
    assert restored[0][1].equal(source[0][1])
    assert restored[2].equal(source[2])


def test_buffer_list_loads_tensor_subclasses() -> None:
    tensor = torch.tensor([1.0]).as_subclass(_TensorSubclass)
    source = BufferList([tensor])
    restored = BufferList()

    restored.load_state_dict(source.state_dict())

    loaded = restored[0]
    assert type(loaded) is _TensorSubclass
    assert torch.equal(loaded, tensor)
    loaded.add_(1)
    assert torch.equal(source[0], tensor)


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
    module.buffer_list = BufferList(
        [BufferList([torch.tensor([1.0])])]
    )

    restored = torch.nn.Module()
    restored.buffer_list = BufferList()
    restored.load_state_dict(module.state_dict())
    nested = restored.buffer_list[0]
    assert isinstance(nested, BufferList)
    assert torch.equal(nested[0], torch.tensor([1.0]))


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
