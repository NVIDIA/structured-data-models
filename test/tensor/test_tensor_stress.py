import io
import math
import pickle
import subprocess
import sys
from itertools import pairwise
from typing import Any, cast

import pyarrow as pa
import pytest
import torch
from schemafm import StringTensor, VarLenTensor
from torch import Tensor


def _reshape(values: list[Any], size: tuple[int, ...]) -> Any:
    if len(size) == 0:
        return values[0]
    if len(size) == 1:
        return values

    step = math.prod(size[1:])
    return [
        _reshape(values[i * step : (i + 1) * step], size[1:])
        for i in range(size[0])
    ]


def _varlen_to_list(tensor: Tensor) -> Any:
    assert isinstance(tensor, VarLenTensor)
    tensor = cast(VarLenTensor, tensor.contiguous())
    data, offset = tensor.data_offset
    data_values = data.tolist()
    offset_values = offset.tolist()
    values = [data_values[start:end] for start, end in pairwise(offset_values)]
    return _reshape(values, tuple(tensor.size()))


def _wrap_scalars(values: Any) -> Any:
    if isinstance(values, list):
        return [_wrap_scalars(value) for value in values]
    return [values]


def _int_buffer(values: list[int], *, byte_width: int) -> pa.Buffer:
    return pa.py_buffer(
        b"".join(
            value.to_bytes(byte_width, "little", signed=True)
            for value in values
        )
    )


def _varlen_from_dense_reference(tensor: Tensor) -> Any:
    values = [[value] for value in tensor.reshape(-1).tolist()]
    return _reshape(values, tuple(tensor.size()))


def _make_tensor_view(kind: str) -> VarLenTensor:
    if kind == "varlen":
        return cast(
            VarLenTensor,
            VarLenTensor(
                data=torch.arange(12),
                offset=torch.arange(13),
                size=(3, 4),
            )[:, 1:3],
        )

    return cast(
        VarLenTensor,
        StringTensor.from_list([["a", "b", "c"], ["d", "e", "f"]])[:, 1:],
    )


def _make_base_tensor(kind: str) -> VarLenTensor:
    if kind == "varlen":
        return VarLenTensor(
            data=torch.arange(12),
            offset=torch.arange(13),
            size=(3, 4),
        )

    return StringTensor.from_list([["a", "b", "c"], ["d", "e", "f"]])


@pytest.mark.parametrize(
    ("values", "expected_size", "expected"),
    [
        ([], (0,), []),
        ([1, 2, 3], (), [1, 2, 3]),
        ([[1, 2], [], [3]], (3,), [[1, 2], [], [3]]),
        (
            [[[1], [2, 3]], [[], [4]]],
            (2, 2),
            [[[1], [2, 3]], [[], [4]]],
        ),
    ],
)
def test_varlen_from_list_shapes_and_values(
    values: list[Any],
    expected_size: tuple[int, ...],
    expected: Any,
) -> None:
    tensor = VarLenTensor.from_list(values)

    assert tensor.size() == expected_size
    assert tensor.tolist() == expected
    if tensor.numel() == 1:
        assert tensor.item() == _reshape([expected], ())


@pytest.mark.parametrize(
    "values",
    [
        [[1], 2],
        [[], 1],
        [[[1]], [[2], [3]]],
    ],
)
def test_varlen_from_list_rejects_ragged_values(values: list[Any]) -> None:
    with pytest.raises(ValueError, match="rectangular"):
        VarLenTensor.from_list(values)


@pytest.mark.parametrize(
    "values",
    [["abc"], [b"abc"], [bytearray(b"abc")]],
)
def test_varlen_from_list_rejects_string_like_values(
    values: list[Any],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        VarLenTensor.from_list(values)


@pytest.mark.parametrize(
    ("values", "expected_size"),
    [
        ([[]], (1,)),
        ([[], []], (2,)),
        ([[[], []], [[], []]], (2, 2)),
        ([[[[]]]], (1, 1, 1)),
    ],
)
def test_varlen_from_list_nested_empty_values(
    values: list[Any],
    expected_size: tuple[int, ...],
) -> None:
    tensor = VarLenTensor.from_list(values)

    assert tensor.size() == expected_size
    assert tensor.tolist() == values


@pytest.mark.parametrize(
    "offset",
    [
        torch.tensor([0, 2, 1, 3]),
        torch.tensor([0, -1, 2, 3]),
        torch.tensor([0, 2, 99]),
    ],
)
def test_varlen_constructor_rejects_invalid_offsets(offset: Tensor) -> None:
    with pytest.raises(ValueError):
        VarLenTensor(
            data=torch.arange(4), offset=offset, size=(offset.numel() - 1,)
        )


def test_varlen_constructor_accepts_nonzero_offset_start() -> None:
    tensor = VarLenTensor(
        data=torch.arange(10),
        offset=torch.tensor([5, 7, 9]),
        size=(2,),
    )

    assert _varlen_to_list(tensor) == [[5, 6], [7, 8]]
    assert tensor.to_arrow().to_pylist() == [[5, 6], [7, 8]]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "data": torch.arange(4).view(2, 2),
                "offset": torch.arange(5),
                "size": (4,),
            },
            "one-dimensional",
        ),
        (
            {
                "data": torch.arange(4),
                "offset": torch.arange(5).view(1, 5),
                "size": (4,),
            },
            "one-dimensional",
        ),
        (
            {
                "data": torch.arange(4),
                "offset": torch.arange(5, dtype=torch.float32),
                "size": (4,),
            },
            "int32",
        ),
        (
            {
                "data": torch.arange(4),
                "offset": torch.arange(5),
                "size": (2, 2),
                "stride": (1,),
            },
            "same length",
        ),
        (
            {
                "data": torch.arange(4),
                "offset": torch.arange(5),
                "size": (4,),
                "storage_offset": -1,
            },
            "non-negative",
        ),
        (
            {
                "data": torch.arange(4),
                "offset": torch.arange(5),
                "size": (-1,),
            },
            "negative",
        ),
        (
            {
                "data": torch.arange(4),
                "offset": torch.arange(5),
                "size": (2,),
                "stride": (-1,),
            },
            "negative",
        ),
    ],
)
def test_varlen_constructor_rejects_invalid_metadata(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        VarLenTensor(**kwargs)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    ],
)
def test_varlen_arrow_roundtrip_supported_dtypes(dtype: torch.dtype) -> None:
    tensor = VarLenTensor(
        data=torch.tensor([1, 2, 3, 4], dtype=dtype),
        offset=torch.tensor([0, 2, 2, 4]),
        size=(3,),
    )

    out = VarLenTensor.from_arrow(tensor.to_arrow())

    assert out.dtype == dtype
    assert _varlen_to_list(out) == [[1, 2], [], [3, 4]]


def test_varlen_arrow_roundtrip_bool_dtype() -> None:
    tensor = VarLenTensor(
        data=torch.tensor([True, False, True]),
        offset=torch.tensor([0, 2, 3]),
        size=(2,),
    )

    out = VarLenTensor.from_arrow(tensor.to_arrow())

    assert out.dtype == torch.bool
    assert _varlen_to_list(out) == [[True, False], [True]]


def test_varlen_from_arrow_bool_dtype() -> None:
    array = pa.array(
        [[True, False], [True]],
        type=pa.list_(pa.bool_()),
    )

    tensor = VarLenTensor.from_arrow(array)

    assert tensor.dtype == torch.bool
    assert tensor.tolist() == [[True, False], [True]]


@pytest.mark.parametrize(
    ("offset_dtype", "arrow_type"),
    [
        (torch.int32, pa.list_(pa.int64())),
        (torch.int64, pa.large_list(pa.int64())),
    ],
)
def test_varlen_offset_dtype_controls_arrow_type(
    offset_dtype: torch.dtype,
    arrow_type: pa.DataType,
) -> None:
    tensor = VarLenTensor.from_list(
        [[1], [2, 3]],
        offset_dtype=offset_dtype,
    )

    assert tensor._offset.dtype == offset_dtype
    assert tensor.to_arrow().type == arrow_type


def test_varlen_arrow_roundtrip_scalar_logical_element() -> None:
    tensor = VarLenTensor.from_list([1, 2, 3])
    out = VarLenTensor.from_arrow(tensor.to_arrow(), size=())

    assert tensor.size() == ()
    assert out.size() == ()
    assert out.item() == [1, 2, 3]


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.bfloat16, torch.complex64, torch.complex128],
)
def test_varlen_tolist_works_for_tensor_dtypes(dtype: torch.dtype) -> None:
    values: list[Any]
    if dtype == torch.bool:
        values = [True, False, True]
    elif dtype in (torch.complex64, torch.complex128):
        values = [1 + 2j, 3 + 4j, 5 + 6j]
    else:
        values = [1.0, 2.0, 3.0]

    tensor = VarLenTensor(
        data=torch.tensor(values, dtype=dtype),
        offset=torch.tensor([0, 2, 3]),
        size=(2,),
    )

    assert tensor.tolist() == [values[:2], values[2:]]


def test_varlen_to_arrow_rejects_requires_grad() -> None:
    tensor = VarLenTensor.from_tensor(torch.randn(2, requires_grad=True))

    with pytest.raises(RuntimeError, match="requires grad"):
        tensor.to_arrow()


def test_varlen_detach_inplace_rejects_logical_views() -> None:
    tensor = VarLenTensor(
        data=torch.arange(6, dtype=torch.float32, requires_grad=True),
        offset=torch.arange(7),
        size=(2, 3),
    )

    with pytest.raises(RuntimeError, match="detach views in-place"):
        tensor[:, ::2].detach_()


def test_varlen_from_arrow_rejects_null_values() -> None:
    array = pa.array([[1, None], [2]], type=pa.list_(pa.int64()))

    with pytest.raises(ValueError, match="null"):
        VarLenTensor.from_arrow(array)


def test_varlen_from_arrow_null_parent_does_not_expose_values() -> None:
    array = pa.ListArray.from_arrays(
        offsets=pa.array([0, 2, 3], type=pa.int32()),
        values=pa.array([99, 100, 1], type=pa.int64()),
        mask=pa.array([True, False], type=pa.bool_()),
    )

    assert array.to_pylist() == [None, [1]]
    assert VarLenTensor.from_arrow(array).tolist() == [[], [1]]


def test_varlen_from_arrow_rejects_invalid_size() -> None:
    array = pa.array([[1], [2]], type=pa.list_(pa.int64()))

    with pytest.raises(ValueError, match="size"):
        VarLenTensor.from_arrow(array, size=(3,))


def test_varlen_from_arrow_accepts_immutable_buffers() -> None:
    values = pa.Array.from_buffers(
        type=pa.int64(),
        length=3,
        buffers=[None, _int_buffer([9, 10, 11], byte_width=8)],
    )
    array = pa.Array.from_buffers(
        type=pa.list_(pa.int64()),
        length=2,
        buffers=[None, _int_buffer([0, 2, 3], byte_width=4)],
        children=[values],
    )

    assert not array.buffers()[1].is_mutable
    assert VarLenTensor.from_arrow(array).tolist() == [[9, 10], [11]]


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_varlen_offset_dtype_survives_common_ops(
    offset_dtype: torch.dtype,
) -> None:
    tensor = VarLenTensor.from_list(
        [[1], [2, 3], []],
        offset_dtype=offset_dtype,
    )

    assert tensor._offset.dtype == offset_dtype
    clone = cast(VarLenTensor, tensor.clone())
    indexed = cast(VarLenTensor, tensor[torch.tensor([True, False, True])])
    cat = cast(VarLenTensor, torch.cat([tensor, tensor]))
    assert clone._offset.dtype == offset_dtype
    assert indexed._offset.dtype == offset_dtype
    assert cat._offset.dtype == offset_dtype


@pytest.mark.parametrize("kind", ["varlen", "string"])
def test_tensor_public_data_and_offset_properties(kind: str) -> None:
    tensor = _make_base_tensor(kind)

    assert tensor.data is tensor._data
    assert getattr(tensor, "offset") is tensor._offset  # noqa: B009


def test_varlen_isnan_matches_inner_values() -> None:
    tensor = VarLenTensor(
        data=torch.tensor([0.0, float("nan"), 1.0]),
        offset=torch.tensor([0, 2, 3]),
        size=(2,),
    )
    out = torch.isnan(tensor)

    assert isinstance(out, VarLenTensor)
    assert out.tolist() == [[False, True], [False]]


def test_varlen_high_rank_empty_tolist_and_item() -> None:
    tensor = VarLenTensor(
        data=torch.empty(0, dtype=torch.int64),
        offset=torch.zeros(1, dtype=torch.int64),
        size=(2, 0, 3),
    )
    singleton = VarLenTensor.from_list([[[[1, 2]]]])

    assert tensor.tolist() == [[], []]
    assert singleton.size() == (1, 1, 1)
    assert singleton.item() == [1, 2]


def test_varlen_view_ops_preserve_logical_values() -> None:
    tensor = VarLenTensor(
        data=torch.arange(12),
        offset=torch.tensor([0, 1, 3, 6, 6, 8, 12]),
        size=(2, 3),
    )

    assert _varlen_to_list(tensor.t()) == [
        [[0], []],
        [[1, 2], [6, 7]],
        [[3, 4, 5], [8, 9, 10, 11]],
    ]
    assert _varlen_to_list(tensor[:, ::2]) == [
        [[0], [3, 4, 5]],
        [[], [8, 9, 10, 11]],
    ]
    assert _varlen_to_list(tensor.reshape(3, 2)) == [
        [[0], [1, 2]],
        [[3, 4, 5], []],
        [[6, 7], [8, 9, 10, 11]],
    ]
    assert _varlen_to_list(tensor.unsqueeze(0).squeeze(0)) == _varlen_to_list(
        tensor
    )


def test_varlen_ops_match_dense_reference_layouts() -> None:
    base = torch.arange(60, dtype=torch.float32).view(3, 4, 5)
    cases = [
        base,
        base[1:],
        base[:, ::2, 1:4],
        base.permute(2, 0, 1),
        base.transpose(0, 1)[1:3],
        base[:1, :, :2].expand(3, 4, 2),
    ]

    for dense in cases:
        tensor = VarLenTensor.from_tensor(dense)
        assert tensor.clone().tolist() == _varlen_from_dense_reference(dense)
        assert tensor.contiguous().tolist() == _varlen_from_dense_reference(
            dense.contiguous()
        )
        assert tensor.reshape(-1).tolist() == _varlen_from_dense_reference(
            dense.reshape(-1)
        )
        assert tensor.flatten().tolist() == _varlen_from_dense_reference(
            dense.flatten()
        )
        assert tensor[..., ::2].tolist() == _varlen_from_dense_reference(
            dense[..., ::2]
        )
        assert tensor.take(torch.arange(5)).tolist() == (
            _varlen_from_dense_reference(dense.take(torch.arange(5)))
        )
        mask = torch.ones(dense.size(), dtype=torch.bool)
        assert tensor.masked_select(mask).tolist() == (
            _varlen_from_dense_reference(dense.masked_select(mask))
        )


def test_varlen_indexing_masking_take_and_split() -> None:
    tensor = VarLenTensor(
        data=torch.arange(12),
        offset=torch.tensor([0, 1, 3, 6, 6, 8, 12]),
        size=(2, 3),
    )

    assert _varlen_to_list(tensor[torch.tensor([True, False])]) == [
        [[0], [1, 2], [3, 4, 5]]
    ]
    assert _varlen_to_list(tensor[:, torch.tensor([2, 0])]) == [
        [[3, 4, 5], [0]],
        [[8, 9, 10, 11], []],
    ]
    assert _varlen_to_list(tensor.take(torch.tensor([[0, 5], [2, 3]]))) == [
        [[0], [8, 9, 10, 11]],
        [[3, 4, 5], []],
    ]

    first, second = tensor.split(1, dim=0)
    assert _varlen_to_list(first) == [[[0], [1, 2], [3, 4, 5]]]
    assert _varlen_to_list(second) == [[[], [6, 7], [8, 9, 10, 11]]]


def test_varlen_common_view_decompositions_match_dense_reference() -> None:
    dense = torch.arange(24).view(2, 3, 4)[:, ::2]
    tensor = VarLenTensor.from_tensor(dense)

    assert torch.movedim(tensor, 0, -1).tolist() == (
        _varlen_from_dense_reference(torch.movedim(dense, 0, -1))
    )
    assert torch.swapdims(tensor, 0, 2).tolist() == (
        _varlen_from_dense_reference(torch.swapdims(dense, 0, 2))
    )
    assert tensor.unflatten(0, (1, 2)).tolist() == (
        _varlen_from_dense_reference(dense.unflatten(0, (1, 2)))
    )
    assert [out.tolist() for out in torch.tensor_split(tensor, 2, dim=-1)] == [
        _varlen_from_dense_reference(out)
        for out in torch.tensor_split(dense, 2, dim=-1)
    ]
    assert [out.tolist() for out in torch.chunk(tensor, 2, dim=1)] == [
        _varlen_from_dense_reference(out)
        for out in torch.chunk(dense, 2, dim=1)
    ]


def test_varlen_cat_and_stack_preserve_offsets_and_values() -> None:
    left = VarLenTensor(
        data=torch.tensor([0, 1, 2, 3]),
        offset=torch.tensor([0, 1, 3, 4]),
        size=(1, 3),
    )
    right = VarLenTensor(
        data=torch.tensor([4, 5, 6]),
        offset=torch.tensor([0, 1, 3]),
        size=(1, 2),
    )

    assert _varlen_to_list(torch.cat([left, right], dim=1)) == [
        [[0], [1, 2], [3], [4], [5, 6]]
    ]

    a = VarLenTensor(
        data=torch.tensor([0, 1]), offset=torch.arange(3), size=(2,)
    )
    b = VarLenTensor(
        data=torch.tensor([2, 3]), offset=torch.arange(3), size=(2,)
    )
    assert _varlen_to_list(torch.stack([a, b], dim=1)) == [
        [[0], [2]],
        [[1], [3]],
    ]


def test_varlen_cat_stack_mixed_dtype_and_empty_inputs() -> None:
    empty = VarLenTensor(
        data=torch.empty(0, dtype=torch.int64),
        offset=torch.zeros(1, dtype=torch.int64),
        size=(0, 2),
    )
    dense = VarLenTensor(
        data=torch.arange(4),
        offset=torch.arange(5),
        size=(2, 2),
    )
    mixed = torch.cat(
        [
            VarLenTensor(
                data=torch.tensor([1, 2]),
                offset=torch.arange(3),
                size=(2,),
            ),
            VarLenTensor(
                data=torch.tensor([3.0, 4.0]),
                offset=torch.arange(3),
                size=(2,),
            ),
        ]
    )

    assert _varlen_to_list(torch.cat([empty, dense], dim=0)) == [
        [[0], [1]],
        [[2], [3]],
    ]
    assert mixed.dtype == torch.float32
    assert _varlen_to_list(mixed) == [[1.0], [2.0], [3.0], [4.0]]


def test_varlen_equal_allclose_and_class_mismatch() -> None:
    tensor = VarLenTensor(
        data=torch.tensor([1.0, 2.0, 3.0]),
        offset=torch.tensor([0, 1, 3]),
        size=(2,),
    )
    other = VarLenTensor(
        data=torch.tensor([1.0, 2.001, 3.0]),
        offset=torch.tensor([0, 1, 3]),
        size=(2,),
    )
    string = StringTensor.from_list(["a", "bb"])

    assert tensor.equal(tensor.clone())
    assert tensor.allclose(other, atol=0.01)
    assert not tensor.allclose(other, atol=0.0001)
    assert not torch.equal(tensor, string)


def test_varlen_equal_allclose_normalize_nonzero_offsets() -> None:
    tensor = VarLenTensor(
        data=torch.tensor([10.0, 11.0, 12.0]),
        offset=torch.tensor([1, 2, 3]),
        size=(2,),
    )
    other = VarLenTensor(
        data=torch.tensor([11.0, 12.0]),
        offset=torch.tensor([0, 1, 2]),
        size=(2,),
    )

    assert tensor.tolist() == other.tolist()
    assert torch.equal(tensor, other)
    assert torch.allclose(tensor, other)


@pytest.mark.parametrize("kind", ["varlen", "string"])
def test_tensor_repr_and_str_are_stable(kind: str) -> None:
    scalar = (
        VarLenTensor.from_list([1, 2])
        if kind == "varlen"
        else (StringTensor.from_list("a"))
    )
    empty = (
        VarLenTensor.from_list([])
        if kind == "varlen"
        else (StringTensor.from_list([]))
    )

    assert scalar.__class__.__name__ in repr(scalar)
    assert empty.__class__.__name__ in str(empty)
    if kind == "varlen":
        assert "dtype=" in repr(scalar)
    else:
        assert str(scalar) == "a"


def test_varlen_from_tensor_view_clone_preserves_autograd_path() -> None:
    base = torch.arange(12, dtype=torch.float32, requires_grad=True)
    dense = base.view(3, 4).t()[:2]
    tensor = VarLenTensor.from_tensor(dense)

    out = tensor.clone(memory_format=torch.contiguous_format)
    assert _varlen_to_list(out) == _wrap_scalars(dense.tolist())

    assert isinstance(out, VarLenTensor)
    out._data.sum().backward()
    assert base.grad is not None
    assert base.grad.nonzero().numel() > 0


def test_varlen_dtype_conversion_preserves_autograd_path() -> None:
    data = torch.randn(3, requires_grad=True)
    tensor = VarLenTensor.from_tensor(data)
    out = tensor.to(torch.float64)
    assert isinstance(out, VarLenTensor)

    out._data.sum().backward()
    assert data.grad is not None
    assert data.grad.equal(torch.ones_like(data))


def test_varlen_grad_property_exposes_inner_grad() -> None:
    data = torch.randn(3, requires_grad=True)
    tensor = VarLenTensor.from_tensor(data)
    out = tensor.clone()
    assert isinstance(out, VarLenTensor)

    out._data.sum().backward()

    assert tensor.grad is data.grad


def test_varlen_grad_setter_updates_inner_grad() -> None:
    data = torch.randn(3, requires_grad=True)
    tensor = VarLenTensor.from_tensor(data)
    grad = torch.ones_like(data)

    tensor.grad = grad

    assert tensor._data.grad is grad


@pytest.mark.parametrize("class_name", ["VarLenTensor", "StringTensor"])
def test_malformed_offsets_do_not_abort_python(class_name: str) -> None:
    code = f"""
import torch
from schemafm import {class_name}

if {class_name!r} == "VarLenTensor":
    tensor = {class_name}(
        data=torch.arange(4),
        offset=torch.tensor([0, 3, 1]),
        size=(2,),
    )
else:
    tensor = {class_name}(
        data=torch.tensor(list(b"abc"), dtype=torch.uint8),
        offset=torch.tensor([0, 3, 1]),
        size=(2,),
    )
tensor.tolist()
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode >= 0, proc.stderr


@pytest.mark.parametrize("kind", ["varlen", "string"])
def test_tensor_base_pickle_and_torch_save_roundtrip(kind: str) -> None:
    tensor = _make_base_tensor(kind)

    out = pickle.loads(pickle.dumps(tensor))
    assert isinstance(out, tensor.__class__)
    assert out.tolist() == tensor.tolist()

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    out = torch.load(buffer, weights_only=False)
    assert isinstance(out, tensor.__class__)
    assert out.tolist() == tensor.tolist()


@pytest.mark.parametrize("kind", ["varlen", "string"])
def test_tensor_views_pickle_roundtrip(kind: str) -> None:
    tensor = _make_tensor_view(kind)

    out = pickle.loads(pickle.dumps(tensor))

    assert isinstance(out, tensor.__class__)
    assert out.tolist() == tensor.tolist()


@pytest.mark.parametrize("kind", ["varlen", "string"])
def test_tensor_views_torch_save_roundtrip(kind: str) -> None:
    tensor = _make_tensor_view(kind)
    buffer = io.BytesIO()

    torch.save(tensor, buffer)
    buffer.seek(0)
    out = torch.load(buffer, weights_only=False)

    assert isinstance(out, tensor.__class__)
    assert out.tolist() == tensor.tolist()


@pytest.mark.parametrize(
    ("values", "expected_size", "expected"),
    [
        ("hi", (), "hi"),
        ([], (0,), []),
        ([["a", ""], ["bb", "ccc"]], (2, 2), [["a", ""], ["bb", "ccc"]]),
        ([[["x"]]], (1, 1, 1), [[["x"]]]),
    ],
)
def test_string_from_list_tolist_and_item(
    values: str | list[Any],
    expected_size: tuple[int, ...],
    expected: Any,
) -> None:
    tensor = StringTensor.from_list(values)

    assert tensor.size() == expected_size
    assert tensor.tolist() == expected
    if tensor.numel() == 1:
        assert tensor.item() == "hi" if values == "hi" else "x"


def test_string_high_rank_empty_to_list() -> None:
    tensor = StringTensor.from_arrow(
        pa.array([], type=pa.string()),
        size=(2, 0, 3),
    )

    assert tensor.tolist() == [[], []]


def test_string_arrow_roundtrip_scalar_logical_element() -> None:
    tensor = StringTensor.from_list("hello")
    out = StringTensor.from_arrow(tensor.to_arrow(), size=())

    assert tensor.size() == ()
    assert out.size() == ()
    assert out.item() == "hello"


def test_string_isna_returns_dense_bool_mask() -> None:
    tensor = StringTensor.from_list([["", "a"], ["bb", ""]])
    out = getattr(tensor, "isna")()  # noqa: B009

    assert isinstance(out, Tensor)
    assert out.dtype == torch.bool
    assert out.tolist() == [[False, False], [False, False]]


@pytest.mark.parametrize(
    ("offset_dtype", "arrow_type"),
    [
        (torch.int32, pa.string()),
        (torch.int64, pa.large_string()),
    ],
)
def test_string_offset_dtype_controls_arrow_type(
    offset_dtype: torch.dtype,
    arrow_type: pa.DataType,
) -> None:
    tensor = StringTensor.from_list(["a", "bb"], offset_dtype=offset_dtype)

    assert tensor._offset.dtype == offset_dtype
    assert tensor.to_arrow().type == arrow_type


def test_string_to_arrow_is_full_arrow_valid() -> None:
    tensor = StringTensor(
        data=torch.tensor([255], dtype=torch.uint8),
        offset=torch.tensor([0, 1]),
        size=(1,),
    )

    tensor.to_arrow().validate(full=True)


def test_string_from_arrow_null_value_does_not_expose_bytes() -> None:
    array = pa.Array.from_buffers(
        type=pa.string(),
        length=2,
        buffers=[
            pa.py_buffer(bytearray([0b10])),
            pa.py_buffer(
                bytearray(
                    (0).to_bytes(4, "little")
                    + (2).to_bytes(4, "little")
                    + (3).to_bytes(4, "little")
                )
            ),
            pa.py_buffer(bytearray(b"zza")),
        ],
        null_count=1,
    )

    assert array.to_pylist() == [None, "a"]
    assert StringTensor.from_arrow(array).tolist() == ["", "a"]


def test_string_from_arrow_accepts_immutable_buffers() -> None:
    array = pa.Array.from_buffers(
        type=pa.string(),
        length=2,
        buffers=[
            None,
            _int_buffer([0, 2, 3], byte_width=4),
            pa.py_buffer(b"abc"),
        ],
    )

    assert not array.buffers()[1].is_mutable
    assert StringTensor.from_arrow(array).tolist() == ["ab", "c"]


@pytest.mark.parametrize(
    ("values", "expected_size"),
    [
        ([[]], (1, 0)),
        ([[], []], (2, 0)),
        ([[[], []], [[], []]], (2, 2, 0)),
    ],
)
def test_string_from_list_nested_empty_shapes(
    values: list[Any],
    expected_size: tuple[int, ...],
) -> None:
    tensor = StringTensor.from_list(values)

    assert tensor.size() == expected_size
    assert tensor.tolist() == values


def test_string_to_arrow_rejects_requires_grad_if_present() -> None:
    tensor = StringTensor.from_list("abc")
    tensor._data = tensor._data.to(torch.float32).requires_grad_()

    with pytest.raises(RuntimeError, match="requires grad"):
        tensor.to_arrow()


def test_string_views_cat_stack_and_clone() -> None:
    tensor = StringTensor.from_list([["a", "bb", ""], ["ccc", "d", "ee"]])

    assert tensor[:, ::2].tolist() == [["a", ""], ["ccc", "ee"]]
    assert tensor.t().tolist() == [["a", "ccc"], ["bb", "d"], ["", "ee"]]
    assert tensor.clone().tolist() == tensor.tolist()

    left = StringTensor.from_list([["a"], ["bb"]])
    right = StringTensor.from_list([[""], ["ccc"]])
    assert torch.cat([left, right], dim=1).tolist() == [
        ["a", ""],
        ["bb", "ccc"],
    ]
    assert torch.stack(
        [left.squeeze(1), right.squeeze(1)], dim=1
    ).tolist() == [
        ["a", ""],
        ["bb", "ccc"],
    ]
    assert torch.stack(
        [StringTensor.from_list("a"), StringTensor.from_list("b")]
    ).tolist() == ["a", "b"]


def test_string_common_view_decompositions_match_reference() -> None:
    values = [
        [[f"{i}{j}{k}" for k in range(4)] for j in range(3)] for i in range(2)
    ]
    tensor = StringTensor.from_list(values)[:, ::2]
    reference = torch.arange(24).view(2, 3, 4)[:, ::2]
    flat_values = [
        value for matrix in values for row in matrix for value in row
    ]

    def expected(layout: Tensor) -> Any:
        values = [flat_values[int(index)] for index in layout.reshape(-1)]
        return _reshape(values, tuple(layout.size()))

    assert torch.movedim(tensor, 0, -1).tolist() == (
        expected(torch.movedim(reference, 0, -1))
    )
    assert torch.swapdims(tensor, 0, 2).tolist() == (
        expected(torch.swapdims(reference, 0, 2))
    )
    assert tensor.unflatten(0, (1, 2)).tolist() == (
        expected(reference.unflatten(0, (1, 2)))
    )
    assert [out.tolist() for out in torch.tensor_split(tensor, 2, dim=-1)] == [
        expected(out) for out in torch.tensor_split(reference, 2, dim=-1)
    ]
    assert [out.tolist() for out in torch.chunk(tensor, 2, dim=1)] == [
        expected(out) for out in torch.chunk(reference, 2, dim=1)
    ]


def test_string_ops_match_reference_layouts() -> None:
    values = [
        [[f"{i}{j}{k}" for k in range(5)] for j in range(4)] for i in range(3)
    ]
    base = StringTensor.from_list(values)
    reference = torch.arange(60).view(3, 4, 5)
    flat_values = [
        value for matrix in values for row in matrix for value in row
    ]
    cases = [
        (base, reference),
        (base[1:], reference[1:]),
        (base[:, ::2, 1:4], reference[:, ::2, 1:4]),
        (base.permute(2, 0, 1), reference.permute(2, 0, 1)),
        (base.transpose(0, 1)[1:3], reference.transpose(0, 1)[1:3]),
        (
            base[:1, :, :2].expand(3, 4, 2),
            reference[:1, :, :2].expand(3, 4, 2),
        ),
    ]

    def expected(layout: Tensor) -> Any:
        values = [flat_values[int(index)] for index in layout.reshape(-1)]
        return _reshape(values, tuple(layout.size()))

    for tensor, layout in cases:
        assert tensor.clone().tolist() == expected(layout)
        assert tensor.contiguous().tolist() == expected(layout.contiguous())
        assert tensor.reshape(-1).tolist() == expected(layout.reshape(-1))
        assert tensor.flatten().tolist() == expected(layout.flatten())
        assert tensor[..., ::2].tolist() == expected(layout[..., ::2])
        assert tensor.take(torch.arange(5)).tolist() == (
            expected(layout.take(torch.arange(5)))
        )
        mask = torch.ones(layout.size(), dtype=torch.bool)
        assert tensor.masked_select(mask).tolist() == (
            expected(layout.masked_select(mask))
        )


def test_string_allclose_is_not_numeric_fuzzy_matching() -> None:
    assert not torch.equal(
        StringTensor.from_list(["a"]),
        StringTensor.from_list(["b"]),
    )
    assert not torch.allclose(
        StringTensor.from_list(["a"]),
        StringTensor.from_list(["b"]),
        atol=1,
    )


def test_string_from_arrow_chunked_and_sliced_roundtrip() -> None:
    array = pa.chunked_array(
        [
            pa.array(["zero", "one"]),
            pa.array(["two", "three"]),
        ]
    )
    tensor = StringTensor.from_arrow(array)[1:3]
    assert isinstance(tensor, StringTensor)

    assert tensor.tolist() == ["one", "two"]
    assert tensor.to_arrow().to_pylist() == ["one", "two"]


@pytest.mark.parametrize(
    "values",
    [
        ["a", 1],
        [b"a"],
        [bytearray(b"a")],
        [["a"], [1]],
        [[["x"]], [["y"], ["z"]]],
    ],
)
def test_string_from_list_rejects_non_string_or_ragged_values(
    values: list[Any],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        StringTensor.from_list(values)
