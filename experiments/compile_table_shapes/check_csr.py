# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103, T201

import argparse
import json

import torch

from sdm.models.kumo.relational.graph import _coo_to_csr

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
results = []
for dtype in (torch.int32, torch.int64):
    for empty in (False, True):
        indices = torch.arange(10, device=args.device, dtype=dtype)[::2]
        if empty:
            indices = indices[:0]
        check = torch.library.opcheck(
            _coo_to_csr,
            (indices, 10),
            kwargs={"out_int32": dtype == torch.int32},
        )
        results.append({"dtype": str(dtype), "empty": empty, "opcheck": check})
    for fullgraph in (False, True):
        torch.compiler.reset()

        def convert(
            indices: torch.Tensor,
            nodes: torch.Tensor,
            dtype: torch.dtype = dtype,
        ) -> torch.Tensor:
            return _coo_to_csr(
                indices, nodes.size(0), out_int32=dtype == torch.int32
            )

        compiled = torch.compile(convert, fullgraph=fullgraph, dynamic=True)
        for size in (5, 9, 13, 0, 1, 5):
            indices = torch.arange(
                size, device=args.device, dtype=dtype
            ).repeat_interleave(2)[::2]
            for empty in (False, True):
                inp = indices[:0] if empty else indices
                nodes = torch.empty(size, device=args.device)
                expected = torch._convert_indices_from_coo_to_csr(
                    inp, size, out_int32=dtype == torch.int32
                )
                actual = compiled(inp, nodes)
                assert actual.dtype == expected.dtype
                assert torch.equal(actual, expected)
                results.append(
                    {
                        "dtype": str(dtype),
                        "fullgraph": fullgraph,
                        "size": size,
                        "empty": empty,
                        "pass": True,
                    }
                )
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "device": args.device,
            "results": results,
        },
        indent=2,
    )
)
