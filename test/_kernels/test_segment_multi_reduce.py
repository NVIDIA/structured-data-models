import importlib

import pytest
import torch

from sdm._kernels import segment_multi_reduce
from sdm._kernels.segment_multi_reduce import _torch_segment_multi_reduce
from sdm.testing import onlyCUDA


@onlyCUDA
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    ("dtype", "num_channels", "atol", "rtol"),
    [
        (torch.float16, 513, 2e-3, 2e-3),
        (torch.bfloat16, 512, 2e-2, 2e-2),
        (torch.float32, 7, 2e-5, 2e-5),
    ],
)
def test_segment_multi_reduce(
    offset_dtype: torch.dtype,
    dtype: torch.dtype,
    num_channels: int,
    atol: float,
    rtol: float,
) -> None:
    degrees = [0, 1, 3, 0, 257]
    offsets = torch.tensor(
        [0, *torch.tensor(degrees).cumsum(0).tolist()],
        device="cuda",
        dtype=offset_dtype,
    )
    src = torch.randn(
        sum(degrees),
        num_channels,
        device="cuda",
        dtype=dtype,
    )
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )
    with torch.inference_mode():
        actual = module.segment_multi_reduce(src, offsets)

    expected = _torch_segment_multi_reduce(src, offsets)
    for result, reference in zip(actual, expected, strict=True):
        assert result.shape == (len(degrees), num_channels)
        assert result.is_contiguous()
        torch.testing.assert_close(
            result,
            reference,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )


@onlyCUDA
def test_segment_multi_reduce_std_threshold() -> None:
    src = torch.tensor(
        [
            [-0.002, -0.004, -0.002, -0.004, -0.002, -0.004, -0.002],
            [0.002, 0.004, 0.002, 0.004, 0.002, 0.004, 0.002],
        ],
        device="cuda",
    )
    offsets = torch.tensor([0, 2], device="cuda")
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    _, _, std, _, _ = module.segment_multi_reduce(src, offsets)

    expected = torch.tensor(
        [[0.0, 0.004, 0.0, 0.004, 0.0, 0.004, 0.0]],
        device="cuda",
    )
    torch.testing.assert_close(std, expected, atol=1e-6, rtol=1e-6)


@onlyCUDA
def test_segment_multi_reduce_empty() -> None:
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    no_segments = module.segment_multi_reduce(
        torch.empty(0, 3, device="cuda"),
        torch.tensor([0], device="cuda"),
    )
    no_channels = module.segment_multi_reduce(
        torch.empty(2, 0, device="cuda"),
        torch.tensor([0, 2], device="cuda"),
    )
    empty_segments = module.segment_multi_reduce(
        torch.empty(0, 7, device="cuda"),
        torch.zeros(4, device="cuda", dtype=torch.int64),
    )

    assert all(result.shape == (0, 3) for result in no_segments)
    assert all(result.shape == (1, 0) for result in no_channels)
    assert all(result.shape == (3, 7) for result in empty_segments)
    assert all(torch.count_nonzero(result) == 0 for result in empty_segments)


@onlyCUDA
def test_segment_multi_reduce_nonfinite() -> None:
    src = torch.tensor(
        [
            [float("nan"), 1.0, float("inf"), -float("inf")],
            [2.0, float("nan"), 4.0, 4.0],
        ],
        device="cuda",
    )
    offsets = torch.tensor([0, 2], device="cuda")
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    actual = module.segment_multi_reduce(src, offsets)

    for result, reference in zip(
        actual,
        _torch_segment_multi_reduce(src, offsets),
        strict=True,
    ):
        torch.testing.assert_close(result, reference, equal_nan=True)


@onlyCUDA
def test_segment_multi_reduce_triton_requires_contiguous_inputs() -> None:
    src = torch.randn(7, 6, device="cuda")
    offsets = torch.tensor([0, -1, 2, -1, 7, -1], device="cuda")
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    with pytest.raises(ValueError, match="src and offsets must be contiguous"):
        module.segment_multi_reduce(src[:, ::2], offsets[::2].contiguous())

    with pytest.raises(ValueError, match="src and offsets must be contiguous"):
        module.segment_multi_reduce(src[:, ::2].contiguous(), offsets[::2])


@onlyCUDA
def test_segment_multi_reduce_noncontiguous() -> None:
    src = torch.randn(7, 6, device="cuda")
    offsets = torch.tensor([0, -1, 2, -1, 7, -1], device="cuda")
    src = src[:, ::2]
    offsets = offsets[::2]

    actual = segment_multi_reduce(src, offsets)

    for result, reference in zip(
        actual,
        _torch_segment_multi_reduce(src, offsets),
        strict=True,
    ):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_segment_multi_reduce_torch(dtype: torch.dtype) -> None:
    src = torch.arange(18, dtype=dtype).reshape(6, 3)
    offsets = torch.tensor([0, 0, 2, 6])

    actual = segment_multi_reduce(src, offsets)

    for result, reference in zip(
        actual,
        _torch_segment_multi_reduce(src, offsets),
        strict=True,
    ):
        torch.testing.assert_close(result, reference)


@onlyCUDA
def test_segment_multi_reduce_compile() -> None:
    src = torch.randn(7, 8, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2, 2, 7], device="cuda")
    compiled = torch.compile(
        segment_multi_reduce,
        fullgraph=True,
        backend="eager",
    )

    actual = compiled(src, offsets)

    for result, reference in zip(
        actual,
        _torch_segment_multi_reduce(src, offsets),
        strict=True,
    ):
        torch.testing.assert_close(result, reference)
