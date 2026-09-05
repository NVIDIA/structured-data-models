import importlib

import pytest
import torch

from sdm._kernels import segment_multi_reduce
from sdm._kernels.segment_multi_reduce import _eager_segment_multi_reduce
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
    index = torch.arange(
        sum(degrees),
        device="cuda",
        dtype=offset_dtype,
    ).remainder(23)
    src = torch.randn(
        23,
        num_channels,
        device="cuda",
        dtype=dtype,
    )
    edge_attr = torch.randn(
        sum(degrees),
        num_channels,
        device="cuda",
        dtype=dtype,
    )
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    with torch.inference_mode():
        actual = module.segment_multi_reduce(src, index, edge_attr, offsets)

    expected = _eager_segment_multi_reduce(src, index, edge_attr, offsets)
    assert actual.shape == (len(degrees), 5, num_channels)
    assert actual.is_contiguous()
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )


@onlyCUDA
@pytest.mark.parametrize(
    ("dtype", "num_channels", "atol", "rtol"),
    [
        (torch.bfloat16, 512, 2e-2, 2e-2),
        (torch.float32, 7, 2e-5, 2e-5),
    ],
)
@pytest.mark.parametrize("edge_type_dtype", [torch.int32, torch.int64])
def test_segment_multi_reduce_edge_type(
    dtype: torch.dtype,
    num_channels: int,
    atol: float,
    rtol: float,
    edge_type_dtype: torch.dtype,
) -> None:
    offsets = torch.tensor([0, 0, 2, 6], device="cuda")
    index = torch.tensor([0, 1, 1, 2, 4, 5], device="cuda")
    src = torch.randn(6, num_channels, device="cuda", dtype=dtype)
    edge_attr = torch.randn(3, num_channels, device="cuda", dtype=dtype)
    edge_type = torch.tensor(
        [2, 0, 1, 2, 1, 0],
        device="cuda",
        dtype=edge_type_dtype,
    )
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    with torch.inference_mode():
        actual = module.segment_multi_reduce(
            src=src,
            index=index,
            edge_attr=edge_attr,
            offsets=offsets,
            edge_type=edge_type,
        )

    expected = _eager_segment_multi_reduce(
        src,
        index,
        edge_attr[edge_type],
        offsets,
    )
    torch.testing.assert_close(
        actual,
        expected,
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
    index = torch.tensor([0, 1], device="cuda")
    edge_attr = torch.zeros_like(src)
    offsets = torch.tensor([0, 2], device="cuda")
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    statistics = module.segment_multi_reduce(
        src=src,
        index=index,
        edge_attr=edge_attr,
        offsets=offsets,
    )

    expected = torch.tensor(
        [[0.0, 0.004, 0.0, 0.004, 0.0, 0.004, 0.0]],
        device="cuda",
    )
    torch.testing.assert_close(
        statistics[:, 2],
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


@onlyCUDA
def test_segment_multi_reduce_empty() -> None:
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    no_segments = module.segment_multi_reduce(
        src=torch.empty(0, 3, device="cuda"),
        index=torch.empty(0, device="cuda", dtype=torch.int64),
        edge_attr=torch.empty(0, 3, device="cuda"),
        offsets=torch.tensor([0], device="cuda"),
    )
    no_channels = module.segment_multi_reduce(
        src=torch.empty(2, 0, device="cuda"),
        index=torch.tensor([0, 1], device="cuda"),
        edge_attr=torch.empty(2, 0, device="cuda"),
        offsets=torch.tensor([0, 2], device="cuda"),
    )
    empty_segments = module.segment_multi_reduce(
        src=torch.empty(0, 7, device="cuda"),
        index=torch.empty(0, device="cuda", dtype=torch.int64),
        edge_attr=torch.empty(0, 7, device="cuda"),
        offsets=torch.zeros(4, device="cuda", dtype=torch.int64),
    )

    assert no_segments.shape == (0, 5, 3)
    assert no_channels.shape == (1, 5, 0)
    assert empty_segments.shape == (3, 5, 7)
    assert torch.count_nonzero(empty_segments) == 0


@onlyCUDA
def test_segment_multi_reduce_nonfinite() -> None:
    src = torch.tensor(
        [
            [float("nan"), 1.0, float("inf"), -float("inf")],
            [2.0, float("nan"), 4.0, 4.0],
        ],
        device="cuda",
    )
    index = torch.tensor([0, 1], device="cuda")
    edge_attr = torch.zeros_like(src)
    offsets = torch.tensor([0, 2], device="cuda")
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    actual = module.segment_multi_reduce(src, index, edge_attr, offsets)

    torch.testing.assert_close(
        actual,
        _eager_segment_multi_reduce(src, index, edge_attr, offsets),
        equal_nan=True,
    )


@onlyCUDA
def test_segment_multi_reduce_triton_requires_contiguous_inputs() -> None:
    src = torch.randn(7, 6, device="cuda")
    index = torch.tensor([0, -1, 1, -1, 2, -1], device="cuda")
    edge_attr = torch.randn(3, 6, device="cuda")
    offsets = torch.tensor([0, -1, 2, -1, 3, -1], device="cuda")
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    with pytest.raises(ValueError, match="must be contiguous"):
        module.segment_multi_reduce(
            src=src[:, ::2],
            index=index[::2].contiguous(),
            edge_attr=edge_attr[:, ::2].contiguous(),
            offsets=offsets[::2].contiguous(),
        )

    with pytest.raises(ValueError, match="must be contiguous"):
        module.segment_multi_reduce(
            src=src[:, ::2].contiguous(),
            index=index[::2],
            edge_attr=edge_attr[:, ::2].contiguous(),
            offsets=offsets[::2].contiguous(),
        )

    with pytest.raises(ValueError, match="must be contiguous"):
        module.segment_multi_reduce(
            src=src[:, ::2].contiguous(),
            index=index[::2].contiguous(),
            edge_attr=edge_attr[:, ::2],
            offsets=offsets[::2].contiguous(),
        )

    with pytest.raises(ValueError, match="must be contiguous"):
        module.segment_multi_reduce(
            src=src[:, ::2].contiguous(),
            index=index[::2].contiguous(),
            edge_attr=edge_attr[:, ::2].contiguous(),
            offsets=offsets[::2],
        )


@onlyCUDA
def test_segment_multi_reduce_requires_matching_edge_attr() -> None:
    module = importlib.import_module(
        "sdm._kernels.triton.segment_multi_reduce"
    )

    with pytest.raises(ValueError, match="edge_attr must have shape"):
        module.segment_multi_reduce(
            src=torch.empty(3, 2, device="cuda"),
            index=torch.tensor([0, 1], device="cuda"),
            edge_attr=torch.empty(2, 3, device="cuda"),
            offsets=torch.tensor([0, 2], device="cuda"),
        )


@onlyCUDA
def test_segment_multi_reduce_noncontiguous() -> None:
    src = torch.randn(7, 6, device="cuda")[:, ::2]
    index = torch.tensor([0, -1, 1, -1, 6, -1], device="cuda")[::2]
    edge_attr = torch.randn(3, 6, device="cuda")[:, ::2]
    offsets = torch.tensor([0, -1, 2, -1, 3, -1], device="cuda")[::2]

    actual = segment_multi_reduce(src, index, edge_attr, offsets)

    torch.testing.assert_close(
        actual,
        _eager_segment_multi_reduce(src, index, edge_attr, offsets),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_segment_multi_reduce_torch(dtype: torch.dtype) -> None:
    src = torch.arange(18, dtype=dtype).reshape(6, 3)
    index = torch.tensor([0, 1, 1, 2, 4, 5])
    edge_attr = torch.ones(6, 3, dtype=dtype)
    offsets = torch.tensor([0, 0, 2, 6])

    actual = segment_multi_reduce(src, index, edge_attr, offsets)

    torch.testing.assert_close(
        actual,
        _eager_segment_multi_reduce(src, index, edge_attr, offsets),
    )


def test_segment_multi_reduce_edge_type_torch() -> None:
    src = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    index = torch.tensor([0, 1, 1, 2, 4, 5])
    edge_attr = torch.tensor([[1.0, 2, 3], [4.0, 5, 6]])
    edge_type = torch.tensor([0, 1, 0, 1, 1, 0])
    offsets = torch.tensor([0, 0, 2, 6])

    actual = segment_multi_reduce(
        src,
        index,
        edge_attr,
        offsets,
        edge_type,
    )

    torch.testing.assert_close(
        actual,
        _eager_segment_multi_reduce(
            src,
            index,
            edge_attr[edge_type],
            offsets,
        ),
    )


@onlyCUDA
@pytest.mark.parametrize("with_edge_type", [False, True])
def test_segment_multi_reduce_compile(with_edge_type: bool) -> None:
    src = torch.randn(7, 8, device="cuda", dtype=torch.bfloat16)
    index = torch.arange(7, device="cuda")
    edge_attr = torch.randn(
        3 if with_edge_type else 7,
        8,
        device="cuda",
        dtype=src.dtype,
    )
    edge_type = (
        torch.arange(7, device="cuda").remainder(3) if with_edge_type else None
    )
    offsets = torch.tensor([0, 2, 2, 7], device="cuda")
    compiled = torch.compile(
        segment_multi_reduce,
        fullgraph=True,
        backend="eager",
    )

    actual = compiled(src, index, edge_attr, offsets, edge_type)

    torch.testing.assert_close(
        actual,
        _eager_segment_multi_reduce(
            src,
            index,
            edge_attr,
            offsets,
            edge_type,
        ),
    )


@onlyCUDA
def test_segment_multi_reduce_grad() -> None:
    src = torch.randn(7, 8, device="cuda", requires_grad=True)
    index = torch.arange(7, device="cuda")
    edge_attr = torch.randn(7, 8, device="cuda", requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 7], device="cuda")

    statistics = segment_multi_reduce(
        src,
        index,
        edge_attr,
        offsets,
    )
    total = statistics[:, 0]
    mean = statistics[:, 1]

    total_grads = torch.autograd.grad(
        total.sum(),
        (src, edge_attr),
        retain_graph=True,
    )
    for grad in total_grads:
        torch.testing.assert_close(grad, torch.ones_like(grad))

    # Segments hold 2, 0 and 5 rows, so each row is averaged over its degree.
    mean_grads = torch.autograd.grad(mean.sum(), (src, edge_attr))
    degree = torch.tensor([2.0, 2, 5, 5, 5, 5, 5], device="cuda")
    expected = degree.reciprocal().unsqueeze(1).expand_as(src)
    for grad in mean_grads:
        torch.testing.assert_close(grad, expected)
