# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    NullableTensor,
    StringTensor,
    TableTensor,
    VarLenTensor,
)
from sdm.testing import onlyCUDA


@onlyCUDA
def test_record_stream() -> None:
    device = torch.device("cuda")
    tensors = (
        CategoricalTensor(
            code=torch.tensor([[0]], device=device),
            categories=(torch.tensor([1], device=device),),
        ),
        ColumnarTensor((torch.tensor([1], device=device),)),
        NullableTensor.from_list([1, None], device=device),
        StringTensor.from_list(["hi", None], device=device),
        TableTensor.from_tensor(torch.tensor([[1.0]], device=device)),
        VarLenTensor.from_list([[1], None], device=device),
    )
    stream = torch.cuda.Stream()

    for tensor in tensors:
        assert tensor.record_stream(stream) is None


def _stream_sources(kind: str) -> tuple[torch.Tensor, ...]:
    data = torch.arange(4096, device="cuda")
    valid = data.remainder(3).ne(0)
    if kind == "categorical":
        return data.remainder(4096).view(-1, 1), data
    if kind in ("var_len", "string"):
        offset = torch.arange(4097, device="cuda")
        if kind == "string":
            data = data.remainder(26).add(97).to(torch.uint8)
        return data, offset, valid
    return data, valid


def _record_stream_tensor(
    kind: str,
    sources: tuple[torch.Tensor, ...],
    stream: torch.cuda.Stream,
) -> None:
    if kind == "categorical":
        tensor = CategoricalTensor(sources[0], (sources[1],))
    elif kind == "columnar":
        tensor = ColumnarTensor((sources[0], NullableTensor(*sources)))
    elif kind == "nullable":
        tensor = NullableTensor(*sources)
    elif kind in ("var_len", "string"):
        cls = StringTensor if kind == "string" else VarLenTensor
        tensor = cls(*sources, size=(4096,))
    else:
        assert kind == "table"
        tensor = TableTensor(
            numerical=NullableTensor(
                sources[0].view(-1, 1), sources[1].view(-1, 1)
            )
        )
    tensor.record_stream(stream)


@onlyCUDA
@pytest.mark.usefixtures("compilation")
@pytest.mark.parametrize(
    "kind",
    ["categorical", "columnar", "nullable", "var_len", "string", "table"],
)
def test_record_stream_fullgraph_lifetime(kind: str) -> None:
    if not hasattr(torch.ops.streams, "record_stream"):
        pytest.skip("requires torch.ops.streams.record_stream")
    sources = _stream_sources(kind)
    expected = tuple(source.cpu() for source in sources)
    outputs = tuple(
        torch.empty_like(value, pin_memory=True) for value in expected
    )
    stream = torch.cuda.Stream()

    compiled_record = torch.compile(_record_stream_tensor, fullgraph=True)
    assert compiled_record(kind, sources, stream) is None
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Keep copies pending while the allocation stream reuses storage.
        torch.cuda._sleep(50_000_000)
        for output, source in zip(outputs, sources):
            output.copy_(source, non_blocking=True)
    del sources, source
    replacements = [
        torch.full_like(value, 1, device="cuda") for value in expected
    ]
    stream.synchronize()
    for output, value in zip(outputs, expected):
        torch.testing.assert_close(output, value)
    del replacements
