import pytest
import torch

from sdm.nn.memory import (
    attention_batch_size_limit,
    cuda_memory_limits,
)


@pytest.mark.parametrize(
    (
        "free_mib",
        "total_mib",
        "fraction",
        "allocated_mib",
        "reserved_mib",
        "expected_mib",
    ),
    [
        (10_240, 20_480, 1.0, 1_024, 2_048, 9_011.2),
        (10_240, 20_480, 0.25, 512, 1_024, 3_686.4),
        (600, 20_480, 1.0, 0, 0, 88),
    ],
)
def test_cuda_memory_limits(
    monkeypatch: pytest.MonkeyPatch,
    free_mib: int,
    total_mib: int,
    fraction: float,
    allocated_mib: int,
    reserved_mib: int,
    expected_mib: float,
) -> None:
    mib = 1024**2
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (free_mib * mib, total_mib * mib),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_per_process_memory_fraction",
        lambda device: fraction if device == torch.device("cuda:0") else 0,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda _device: allocated_mib * mib,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda _device: reserved_mib * mib,
    )

    headroom, allocator_limit = cuda_memory_limits(torch.device("cuda"))

    assert headroom == pytest.approx(expected_mib * mib, abs=1)
    assert allocator_limit == int(total_mib * mib * fraction)


def test_attention_batch_size_limit() -> None:
    query = torch.empty(2, 10, 16)
    key_value = torch.empty(2, 20, 16)
    bytes_per_batch = 12 * 20 * 16 * 4

    assert (
        attention_batch_size_limit(
            None,
            query,
            key_value,
            work_byte_limit=bytes_per_batch,
        )
        == 1
    )
    assert (
        attention_batch_size_limit(
            None,
            query,
            key_value,
            work_byte_limit=2 * bytes_per_batch,
        )
        is None
    )
    assert (
        attention_batch_size_limit(
            1,
            query,
            key_value,
            work_byte_limit=2 * bytes_per_batch,
        )
        == 1
    )
