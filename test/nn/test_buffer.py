import copy

import torch

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
    assert tuple(name for name, _ in buffers.named_buffers()) == ("0", "1")
    assert buffers[0].equal(torch.tensor([1.0, 2.0]))
    assert tuple(buffers)[1].equal(torch.tensor([3.0]))


def test_buffer_list_loads_tensor_subclasses() -> None:
    tensor = torch.tensor([1.0]).as_subclass(_TensorSubclass)
    restored = BufferList()

    restored.load_state_dict(BufferList([tensor]).state_dict())

    assert type(restored[0]) is _TensorSubclass
    assert torch.equal(restored[0], tensor)


def test_buffer_list_registers_as_a_nested_module() -> None:
    module = torch.nn.Module()
    module.states = BufferList([torch.tensor([1.0])])

    assert tuple(module.state_dict()) == ("states.0",)
    assert tuple(name for name, _ in module.named_buffers()) == ("states.0",)

    restored = torch.nn.Module()
    restored.states = BufferList()
    restored.load_state_dict(module.state_dict())
    assert torch.equal(restored.states[0], torch.tensor([1.0]))


@onlyCUDA
def test_buffer_list_cuda_and_cpu() -> None:
    buffers = BufferList([torch.tensor([1.0])]).cuda()

    assert buffers[0].is_cuda
    assert not buffers.cpu()[0].is_cuda


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

    assert buffers[0].device == device
    assert buffers[0].dtype == torch.float64
    assert buffers[1].device == device
    assert buffers[1].dtype == torch.long


def test_buffer_list_deepcopy_has_independent_storage() -> None:
    buffers = BufferList([torch.tensor([1.0, 2.0])])

    cloned = copy.deepcopy(buffers)
    cloned[0].add_(1)

    assert torch.equal(buffers[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(cloned[0], torch.tensor([2.0, 3.0]))
