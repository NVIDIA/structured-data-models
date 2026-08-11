import pytest
import torch

from sdm.nn.memory import (
    attention_batch_size_limit,
    cuda_attention_memory_limit,
    cuda_memory_availability,
)
from sdm.testing import withCUDA


@pytest.mark.parametrize(
    (
        "free_mib",
        "total_mib",
        "fraction",
        "allocated_mib",
        "reserved_mib",
        "expected_available_mib",
        "expected_attention_mib",
    ),
    [
        (10_240, 20_480, 1.0, 1_024, 2_048, 9_011.2, 1_024),
        (10_240, 20_480, 0.25, 512, 1_024, 3_686.4, 256),
        (600, 20_480, 1.0, 0, 0, 88, 88),
    ],
)
def test_cuda_memory_availability(
    monkeypatch: pytest.MonkeyPatch,
    free_mib: int,
    total_mib: int,
    fraction: float,
    allocated_mib: int,
    reserved_mib: int,
    expected_available_mib: float,
    expected_attention_mib: float,
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

    available_memory, capacity = cuda_memory_availability(torch.device("cuda"))

    assert available_memory == pytest.approx(
        expected_available_mib * mib,
        abs=1,
    )
    assert capacity == int(total_mib * mib * fraction)
    assert cuda_attention_memory_limit(torch.device("cuda")) == pytest.approx(
        expected_attention_mib * mib,
        abs=1,
    )


@pytest.mark.parametrize(
    ("dtype", "estimated_element_size"),
    [
        (torch.float16, 4),
        (torch.float32, 4),
        (torch.float64, 8),
    ],
)
def test_attention_batch_size_limit(
    dtype: torch.dtype,
    estimated_element_size: int,
) -> None:
    query = torch.empty(2, 10, 16, dtype=dtype)
    key_value = torch.empty(2, 20, 16, dtype=dtype)
    bytes_per_batch = 12 * 20 * 16 * estimated_element_size

    assert (
        attention_batch_size_limit(
            None,
            query,
            key_value,
            attention_memory_limit=bytes_per_batch,
        )
        == 1
    )
    assert (
        attention_batch_size_limit(
            None,
            query,
            key_value,
            attention_memory_limit=2 * bytes_per_batch,
        )
        is None
    )
    assert (
        attention_batch_size_limit(
            1,
            query,
            key_value,
            attention_memory_limit=2 * bytes_per_batch,
        )
        == 1
    )


@withCUDA
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float64],
)
def test_attention_math_backend_batch_size_limit(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    num_heads = 2
    query = torch.empty(2, 1_000, 16, device=device, dtype=dtype)
    key_value = torch.empty(2, 1_000, 16, device=device, dtype=dtype)
    score_bytes = (
        num_heads
        * query.size(-2)
        * key_value.size(-2)
        * max(query.element_size(), 4)
    )
    uses_math_backend = device.type == "cuda" and (
        dtype == torch.float64
        or (
            dtype == torch.bfloat16
            and torch.cuda.get_device_capability(device)[0] < 8
        )
    )

    limit = attention_batch_size_limit(
        None,
        query,
        key_value,
        attention_memory_limit=score_bytes,
        num_heads=num_heads,
    )

    assert limit == (1 if uses_math_backend else None)
