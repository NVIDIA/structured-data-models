# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103, T201

import argparse
import json
from typing import Any

import torch
from torch._dynamo.testing import CompileCounterWithBackend

from sdm import ColumnarTensor


def transform(table: ColumnarTensor) -> ColumnarTensor:
    return table.transpose(0, 1)[1::2]


parser = argparse.ArgumentParser()
parser.add_argument("--restore-outer-strides", action="store_true")
parser.add_argument("--include-empty", action="store_true")
args = parser.parse_args()
if args.restore_outer_strides:

    def old_unflatten(
        cls: type[ColumnarTensor],
        inner_tensors: dict[str, torch.Tensor],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> ColumnarTensor:
        out = torch.Tensor._make_wrapper_subclass(
            cls,
            size=outer_size,
            strides=outer_stride,
            dtype=torch.uint8,
            device=next(iter(inner_tensors.values())).device,
            requires_grad=False,
        )
        out._columns = tuple(
            value for name, value in inner_tensors.items() if name != "_empty"
        )
        for name, value in inner_tensors.items():
            setattr(out, name, value)
        return out

    ColumnarTensor.__tensor_unflatten__ = classmethod(old_unflatten)
results = []
for fullgraph in (False, True):
    kinds = ("slice", "transpose", "expand")
    if args.include_empty:
        kinds += ("empty",)
    for kind in kinds:
        torch.compiler.reset()
        backend = CompileCounterWithBackend("inductor")
        compiled = torch.compile(
            transform, backend=backend, fullgraph=fullgraph, dynamic=True
        )
        for rows in (5, 9):
            base = torch.arange((rows + 2) * 8.0).reshape(rows + 2, 8)
            leaf = base[1:-1, 1::2]
            table = ColumnarTensor((leaf,))
            if kind == "transpose":
                table = table.transpose(0, 1)
            elif kind == "expand":
                table = ColumnarTensor((base[1:2, 1::2],)).expand(rows, 4, 1)
            elif kind == "empty":
                table = ColumnarTensor((), size=(rows, 4)).transpose(0, 1)
            expected = transform(table)
            try:
                actual = compiled(table)
            except Exception as error:  # noqa: BLE001
                results.append(
                    {
                        "torch": torch.__version__,
                        "fullgraph": fullgraph,
                        "kind": kind,
                        "rows": rows,
                        "pass": False,
                        "error": str(error)[-2000:],
                    }
                )
                continue
            assert actual.shape == expected.shape
            assert actual.stride() == expected.stride()
            assert actual.storage_offset() == expected.storage_offset() == 0
            assert actual.is_contiguous() == expected.is_contiguous()
            for got, want, original in zip(
                actual._columns, expected._columns, table._columns, strict=True
            ):
                torch.testing.assert_close(got, want)
                assert got.stride() == want.stride()
                assert got.storage_offset() == want.storage_offset()
                assert (
                    got.untyped_storage().data_ptr()
                    == original.untyped_storage().data_ptr()
                )
            if kind != "empty":
                base.add_(17)
                for got, want in zip(
                    actual._columns, expected._columns, strict=True
                ):
                    torch.testing.assert_close(got, want)
            results.append(
                {
                    "torch": torch.__version__,
                    "fullgraph": fullgraph,
                    "kind": kind,
                    "rows": rows,
                    "input_stride": table.stride(),
                    "output_stride": actual.stride(),
                    "leaf_stride": [x.stride() for x in actual._columns],
                    "leaf_offset": [
                        x.storage_offset() for x in actual._columns
                    ],
                    "graphs": backend.frame_count,
                    "pass": True,
                }
            )
print(json.dumps(results, indent=2))

raise SystemExit(int(not all(result["pass"] for result in results)))
