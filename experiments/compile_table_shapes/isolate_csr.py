# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: T201

import argparse
import json
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
parser.add_argument("--case")
args = parser.parse_args()
if args.case is None:
    cases = []
    for implementation in ("native", "wrapper"):
        for dtype in ("int32", "int64"):
            for layout in ("contiguous", "strided", "empty"):
                case = f"{implementation}:{dtype}:{layout}"
                process = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--device",
                        args.device,
                        "--case",
                        case,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                record = {
                    "case": case,
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }
                print(json.dumps(record), flush=True)
                cases.append(record)
    sys.exit(int(any(case["returncode"] for case in cases)))

import torch  # noqa: E402

from sdm.models.kumo.relational.graph import _coo_to_csr  # noqa: E402

implementation, dtype_name, layout = args.case.split(":")
dtype = getattr(torch, dtype_name)
base = torch.arange(10, dtype=dtype)
indices = base[::2] if layout == "strided" else base
if layout == "empty":
    indices = indices[:0]
counts = indices.long().bincount(minlength=10)
expected = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(dtype)
base = base.to(args.device)
indices = base[::2] if layout == "strided" else base
if layout == "empty":
    indices = indices[:0]
if args.device == "cuda":
    torch.cuda.synchronize()
print(
    json.dumps(
        {
            "stage": "input_ready",
            "case": args.case,
            "torch": torch.__version__,
            "stride": indices.stride(),
        }
    ),
    flush=True,
)
operator = (
    torch._convert_indices_from_coo_to_csr
    if implementation == "native"
    else _coo_to_csr
)
actual = operator(indices, 10, out_int32=dtype == torch.int32)
if args.device == "cuda":
    torch.cuda.synchronize()
assert torch.equal(actual.cpu(), expected)
print(
    json.dumps({"stage": "pass", "result": actual.cpu().tolist()}), flush=True
)
