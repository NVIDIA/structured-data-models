# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from typing import cast

import pyarrow as pa
import pytest
import torch

from sdm import StringTensor
from sdm.tensor.string import _sort_native
from sdm.testing import onlyCUDA, withCUDA


def test_from_list() -> None:
    tensor = StringTensor.from_list([["hi", "é"], ["", "abc"]])
    assert repr(tensor) == "StringTensor(size=(2, 2))"
    assert tensor.size() == (2, 2)
    assert tensor.stride() == (2, 1)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105, 195, 169, 97, 98, 99]))
    assert tensor._offset.equal(torch.tensor([0, 2, 4, 4, 7]))

    tensor = StringTensor.from_list([])
    assert tensor.size() == (0,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([]))
    assert tensor._offset.equal(torch.tensor([0]))

    tensor = StringTensor.from_list("hi")
    assert tensor.size() == ()
    assert tensor.stride() == ()
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105]))
    assert tensor._offset.equal(torch.tensor([0, 2]))

    tensor = StringTensor.from_list(None)
    assert repr(tensor) == "StringTensor(size=(), null_count=1)"
    assert tensor.size() == ()
    assert tensor.valid is not None
    assert not bool(tensor.valid)
    assert tensor.item() is None


def test_arrow() -> None:
    tensor = StringTensor.from_arrow(pa.array(["hi", "é", ""]))
    assert tensor.size() == (3,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105, 195, 169]))
    assert tensor._offset.equal(torch.tensor([0, 2, 4, 4]))
    assert tensor.to_arrow().type == pa.string()

    tensor = StringTensor.from_arrow(pa.array([], type=pa.string()))
    assert tensor.size() == (0,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([]))
    assert tensor._offset.equal(torch.tensor([0]))

    tensor = StringTensor.from_list([["hi", "é"], ["", "abc"]])
    array = tensor.to_arrow()
    assert array.type == pa.large_string()
    assert array.to_pylist() == ["hi", "é", "", "abc"]

    tensor = StringTensor.from_list(["hi", "é", "", "abc"])
    array = cast(StringTensor, tensor[1:]).to_arrow()
    assert array.offset == 1
    assert array.to_pylist() == ["é", "", "abc"]
    assert (
        array.buffers()[1].address
        == tensor._offset.numpy().__array_interface__["data"][0]
    )
    assert (
        array.buffers()[2].address
        == tensor._data.numpy().__array_interface__["data"][0]
    )


@pytest.mark.parametrize("arrow_type", [pa.string(), pa.large_string()])
def test_from_arrow_chunked_strings(arrow_type: pa.DataType) -> None:
    tensor = StringTensor.from_arrow(
        pa.chunked_array(
            [
                pa.array(["hi", "é"], type=arrow_type),
                pa.array(["", "abc"], type=arrow_type),
            ]
        ),
        size=(2, 2),
    )

    assert tensor.tolist() == [["hi", "é"], ["", "abc"]]
    assert tensor.to_arrow().type == pa.large_string()
    assert tensor.to_arrow().to_pylist() == ["hi", "é", "", "abc"]


def test_from_arrow_single_string_chunk() -> None:
    tensor = StringTensor.from_arrow(
        pa.chunked_array([pa.array(["hi", "é"], type=pa.string())]),
    )

    assert tensor.to_arrow().type == pa.string()
    assert tensor.to_arrow().to_pylist() == ["hi", "é"]


@pytest.mark.parametrize("arrow_type", [pa.string(), pa.large_string()])
def test_from_arrow_zero_string_chunks(arrow_type: pa.DataType) -> None:
    tensor = StringTensor.from_arrow(
        pa.chunked_array([], type=arrow_type),
    )

    assert tensor.to_arrow().type == arrow_type
    assert tensor.to_arrow().to_pylist() == []


@pytest.mark.parametrize("arrow_type", [pa.string(), pa.large_string()])
def test_from_arrow_sliced_nullable(arrow_type: pa.DataType) -> None:
    tensor = StringTensor.from_arrow(
        pa.array(["x", "hi", None, "yo", "z"], type=arrow_type)[1:],
    )

    assert tensor.tolist() == ["hi", None, "yo", "z"]
    assert tensor[torch.tensor([0, 2])].tolist() == ["hi", "yo"]
    transposed = tensor.view(2, 2).t()
    assert not transposed.is_contiguous()
    assert transposed.contiguous().tolist() == [["hi", "yo"], [None, "z"]]


@onlyCUDA
def test_from_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(
        cudf.Series(["hi", "é", ""]),
    )

    assert tensor.is_cuda
    assert tensor.tolist() == ["hi", "é", ""]
    assert tensor._data.equal(
        torch.tensor([104, 105, 195, 169], device=tensor.device)
    )
    assert tensor._offset.equal(
        torch.tensor([0, 2, 4, 4], device=tensor.device)
    )


@onlyCUDA
def test_from_cudf_sliced_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(
        cudf.Series(["x", "hi", "é", ""])[1:],
    )

    assert tensor.is_cuda
    assert tensor.tolist() == ["hi", "é", ""]
    # The slice anchors into the base offsets instead of copying:
    assert int(tensor.storage_offset()) == 1
    assert tensor._offset.equal(
        torch.tensor([0, 1, 3, 5, 5], device=tensor.device)
    )
    # The bound covers the slice, rounded up to 2**k - 1 bytes.
    assert tensor._max_length == 3


@onlyCUDA
def test_from_cudf_sliced_nullable() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(cudf.Series(["x", "hi", None, "yo"])[1:])

    assert tensor.tolist() == ["hi", None, "yo"]
    index = torch.tensor([0, 2], device=tensor.device)
    assert tensor[index].tolist() == ["hi", "yo"]


@onlyCUDA
def test_from_cudf_all_empty_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(
        cudf.Series(["", ""]),
    )

    assert tensor.is_cuda
    assert tensor.tolist() == ["", ""]
    assert tensor._data.numel() == 0
    assert tensor._offset.equal(torch.tensor([0, 0, 0], device=tensor.device))
    assert tensor._max_length == 0


@onlyCUDA
def test_from_cudf_empty_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(
        cudf.Series([], dtype="object"),
    )

    assert tensor.is_cuda
    assert tensor.tolist() == []
    assert tensor._data.numel() == 0
    assert tensor._offset.equal(torch.tensor([0], device=tensor.device))
    assert tensor._max_length == 0


@pytest.mark.parametrize(
    "offset_dtype",
    [torch.int32, torch.int64],
    ids=["int32", "int64"],
)
@onlyCUDA
def test_to_cudf(offset_dtype: torch.dtype) -> None:
    pytest.importorskip("cudf")
    values = ["é", "東京", "🙂", ""]
    tensor = StringTensor.from_list(
        values,
        device="cuda",
        offset_dtype=offset_dtype,
    )

    series = tensor.to_cudf()

    assert tensor._offset.dtype == offset_dtype
    assert series.to_arrow().to_pylist() == values


@pytest.mark.parametrize(
    "offset_dtype",
    [torch.int32, torch.int64],
    ids=["int32", "int64"],
)
@onlyCUDA
def test_to_cudf_empty(offset_dtype: torch.dtype) -> None:
    cudf = pytest.importorskip("cudf")
    tensor = StringTensor.from_list(
        [],
        device="cuda",
        offset_dtype=offset_dtype,
    )

    series = tensor.to_cudf()

    assert tensor._data.numel() == 0
    assert tensor._offset.equal(
        torch.zeros(1, dtype=offset_dtype, device=tensor.device)
    )
    assert cudf.api.types.is_string_dtype(series.dtype)
    assert series.to_arrow().to_pylist() == []


def test_to_cudf_requires_cuda() -> None:
    tensor = StringTensor.from_list(["a"])

    with pytest.raises(RuntimeError, match="on a CUDA device"):
        tensor.to_cudf()


def test_allowed_dtype() -> None:
    tensor = StringTensor.from_list("hi")

    with pytest.raises(TypeError, match="Can't convert"):
        tensor.to(torch.float32)


@withCUDA
def test_eq(device: torch.device) -> None:
    left = StringTensor.from_list(["a", "b", "a"], device=device)
    right = StringTensor.from_list(["a", "a", "b"], device=device)

    assert (left == right).equal(
        torch.tensor([True, False, False], device=device)
    )
    assert (left == "a").equal(
        torch.tensor([True, False, True], device=device)
    )
    assert (left != right).equal(~(left == right))
    assert (left != "a").equal(~(left == "a"))

    left = StringTensor.from_list([["a", "b"]], device=device)
    right = StringTensor.from_list([["a"], ["b"]], device=device)

    assert (left == right).equal(
        torch.tensor([[True, False], [False, True]], device=device),
    )
    assert (left != right).equal(~(left == right))


def test_to_dtype_layout_copy() -> None:
    tensor = StringTensor.from_list(["hi", "é", ""])

    out = tensor.to(torch.uint8, copy=True)
    assert isinstance(out, StringTensor)
    assert out.tolist() == tensor.tolist()

    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.tolist() == tensor.tolist()


@pytest.mark.parametrize("create_in_inference_mode", [False, True])
def test_to_list_in_inference_mode(
    create_in_inference_mode: bool,
) -> None:
    values = ["hi", "é", ""]
    if create_in_inference_mode:
        with torch.inference_mode():
            tensor = StringTensor.from_list(values)
    else:
        tensor = StringTensor.from_list(values)

    with torch.inference_mode():
        assert tensor.tolist() == values


def test_item() -> None:
    assert StringTensor.from_list("é").item() == "é"
    assert StringTensor.from_list(["hi", "é"])[1].item() == "é"

    with pytest.raises(RuntimeError, match="cannot be converted"):
        StringTensor.from_list(["hi", "é"]).item()


def test_tolist() -> None:
    for strings in [
        "é",
        ["hi", "é"],
        [],
        [["a", "bb", "c"], ["dd", "e", "ff"]],
    ]:
        assert StringTensor.from_list(strings).tolist() == strings


@withCUDA
def test_sort(device: torch.device) -> None:
    tensor = StringTensor.from_list(["b", "aa", "a", "é", ""], device=device)

    out, perm = tensor.sort()
    assert out.device == device
    assert out.tolist() == ["", "a", "aa", "b", "é"]
    assert perm.equal(torch.tensor([4, 2, 1, 0, 3], device=device))

    out, perm = torch.sort(tensor, dim=-1, descending=True)
    assert out.device == device
    assert out.tolist() == ["é", "b", "aa", "a", ""]
    assert perm.equal(torch.tensor([3, 0, 1, 2, 4], device=device))

    perm = torch.argsort(tensor)
    assert perm.equal(torch.tensor([4, 2, 1, 0, 3], device=device))

    tensor = StringTensor.from_list(["b", None, "a", ""], device=device)
    out, perm = tensor.sort()
    assert out.device == device
    assert out.tolist() == ["", "a", "b", None]
    assert perm.equal(torch.tensor([3, 2, 0, 1], device=device))

    out, perm = torch.sort(tensor, dim=-1, descending=True)
    assert out.device == device
    assert out.tolist() == ["b", "a", "", None]
    assert perm.equal(torch.tensor([0, 2, 3, 1], device=device))


def test_null_handling() -> None:
    tensor = StringTensor.from_arrow(pa.array(["hi", None, "yo"]))
    assert tensor.valid is not None
    assert tensor.valid.equal(torch.tensor([True, False, True]))
    assert tensor.to_arrow().to_pylist() == ["hi", None, "yo"]
    assert tensor.tolist() == ["hi", None, "yo"]

    tensor = StringTensor.from_list(["hi", None, "yo"])
    assert tensor.valid is not None
    assert tensor.valid.equal(torch.tensor([True, False, True]))
    assert tensor.to_arrow().to_pylist() == ["hi", None, "yo"]
    assert tensor.tolist() == ["hi", None, "yo"]

    tensor = StringTensor.from_list([["hi", None], ["", "yo"]])
    assert tensor.valid is not None
    assert tensor.valid.equal(torch.tensor([[True, False], [True, True]]))
    assert tensor.tolist() == [["hi", None], ["", "yo"]]

    assert (tensor == "").equal(torch.tensor([[False, False], [True, False]]))
    assert (tensor != "").equal(torch.tensor([[True, False], [False, True]]))

    other = StringTensor.from_list([["hi", "x"], ["", None]])
    assert (tensor == other).equal(
        torch.tensor([[True, False], [True, False]])
    )
    assert (tensor != other).equal(
        torch.tensor([[False, False], [False, False]])
    )


@onlyCUDA
def test_cudf_null_handling() -> None:
    cudf = pytest.importorskip("cudf")
    tensor = StringTensor.from_cudf(cudf.Series(["hi", None, "yo"]))
    assert tensor.to_arrow().to_pylist() == ["hi", None, "yo"]

    assert (tensor == "hi").equal(
        torch.tensor([True, False, False], device="cuda")
    )
    assert (tensor != "hi").equal(
        torch.tensor([False, False, True], device="cuda")
    )


def test_registers_as_a_pytorch_buffer() -> None:
    tensor = StringTensor.from_list(["a", "b"])
    module = torch.nn.Module()
    module.register_buffer("value", tensor)

    assert tuple(module.buffers()) == (tensor,)


@pytest.mark.parametrize(
    ("length", "bound"),
    [(0, 0), (1, 1), (3, 3), (4, 7), (127, 127), (128, 255)],
)
def test_bound_rounds_up(length: int, bound: int) -> None:
    values = StringTensor.from_list(["a" * length, ""])
    assert values._max_length == bound


@withCUDA
@pytest.mark.parametrize("bound", [None, 0, 15])
def test_bound_provenance(
    compilation: None,
    device: torch.device,
    bound: int | None,
) -> None:
    texts = ["", None] if bound == 0 else ["abcdefgh", "é"]
    # A non-null Arrow slice shares storage with a longer, unreachable string.
    array = pa.array(texts)
    if bound == 15:
        array = pa.array(["x" * 40, *texts]).slice(1)
    values = StringTensor.from_arrow(array, device=device)
    if bound is None:
        values = StringTensor(
            data=values._data,
            offset=values._offset,
            valid=values._valid,
            size=values.size(),
        )
    compiled = torch.compile(lambda value: value.detach(), fullgraph=True)
    for copied, expected in (
        (values[:], texts),
        (values[::2], texts[::2]),
        (values[torch.tensor([1, 0], device=device)], texts[::-1]),
        (values.clone(), texts),
        (values.detach(), texts),
        (values.to(device, copy=True), texts),
        (pickle.loads(pickle.dumps(values)), texts),
    ):
        copied = cast(StringTensor, copied)
        assert copied._max_length == bound
        assert copied.tolist() == expected
    # Compiled outputs carry no bound, so new bounds do not recompile.
    compiled_values = cast(StringTensor, compiled(values))
    assert compiled_values._max_length is None
    assert compiled_values.tolist() == texts
    for suffix in ("", "x" * 16):
        extra = StringTensor.from_list([suffix], device=device)
        joined = cast(StringTensor, torch.cat((values, extra)))
        new_bound = None if bound is None else max(bound, 31 if suffix else 0)
        assert joined._max_length == new_bound
        assert joined.tolist() == [*texts, suffix]


SHARED = "b" * 300
SORT_VALUES = [
    [],
    [None, None],
    ["", "", None, "\0", "\0a", "a", "a", None],
    ["é", "e", "中", "😀", "é", "\x7f", "\x80"],
    ["a" * 7 + "b", "a" * 8, "a" * 7, "a" * 14, None],
    ["ba", None, "", "a\0", "é", "a", "ba", "b" * 20, None, "a" * 9],
    # A string as long as the sorted prefix ties with its extensions.
    ["p" * 21 + "q", "p" * 21, "p" * 21 + "q"],
    # Strings that tie on their prefix are ordered by their remaining bytes.
    ["é" * 511 + "a\0", "é" * 511 + "a", "é" * 511 + "b", "é" * 511 + "a\0"],
    [
        SHARED + "a",
        SHARED,
        "z" * 600,
        SHARED + "a",
        None,
        "z" * 400,
        "",
        "é" * 200,
        "a",
        "z" * 400,
        "é" * 200,
        SHARED + "\0",
    ],
]


def strided(values: list[str | None], device: torch.device) -> StringTensor:
    # Interleaved padding gives the view a storage offset and a stride.
    storage: list[str | None] = ["unused"]
    for value in values:
        storage.extend([value, "unused"])
    tensor = StringTensor.from_list(storage, device=device)[1::2]
    return cast(StringTensor, tensor)


def sorted_indices(values: list[str | None], descending: bool) -> list[int]:
    # UTF-8 byte order, stable, with nulls last in both directions.
    indices = sorted(
        (i for i, value in enumerate(values) if value is not None),
        key=lambda i: cast(str, values[i]).encode("utf-8"),
        reverse=descending,
    )
    indices.extend(i for i, value in enumerate(values) if value is None)
    return indices


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("values", SORT_VALUES)
@withCUDA
def test_sort_stable(
    values: list[str | None],
    descending: bool,
    native: bool,
    device: torch.device,
) -> None:
    tensor = strided(values, device)
    indices = sorted_indices(values, descending)

    if native:
        out, perm = _sort_native(tensor, descending=descending)
    else:
        out, perm = torch.sort(tensor, descending=descending, stable=True)
    assert out.tolist() == [values[i] for i in indices]
    assert perm.tolist() == indices
    assert out.device == device


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("values", SORT_VALUES[-3:])
@onlyCUDA
def test_sort_native_fullgraph(
    compilation: None,
    values: list[str | None],
    descending: bool,
) -> None:
    tensor = strided(values, torch.device("cuda"))

    def sort(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sort(inp, descending=descending, stable=True)

    out, perm = torch.compile(sort, fullgraph=True)(tensor)
    indices = sorted_indices(values, descending)
    assert perm.tolist() == indices
    assert out.tolist() == [values[i] for i in indices]


@withCUDA
def test_sort_storage_edge_cases(device: torch.device) -> None:
    # Null slots may contain arbitrary bytes and must retain their input order.
    null_payload = pa.Array.from_buffers(
        pa.string(),
        4,
        [
            pa.py_buffer(bytes([0b00000101])),
            pa.py_buffer(
                torch.tensor([0, 1, 4, 5, 8], dtype=torch.int32).numpy()
            ),
            pa.py_buffer(bytearray(b"bzzzazzz")),
        ],
    )
    cases = [
        (StringTensor.from_arrow(null_payload, device=device), [2, 0, 1, 3]),
        # Every row of a stride-0 view refers to the same string.
        (StringTensor.from_list(["abc"], device=device).expand(5), range(5)),
        # Empty strings with an empty byte buffer.
        (StringTensor.from_list(["", ""], device=device), range(2)),
    ]
    for tensor, indices in cases:
        out, perm = _sort_native(cast(StringTensor, tensor), descending=False)
        assert perm.tolist() == list(indices)
        assert out.tolist() == [tensor.tolist()[i] for i in indices]
