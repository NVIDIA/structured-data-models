# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the actual SDM CUDA string and relational join interfaces."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

import sdm
from sdm.relational.join import join_index


def check(output: Path) -> None:
    """Exercise CUDA paths and compare with CPU reference join pairs."""
    # Import intentionally initializes the candidate GPU dataframe backend.
    import cudf
    import pandas as pd

    left = sdm.TableTensor.from_pandas(
        pd.DataFrame({"key": ["b", "a", "b", None], "example": [0, 0, 1, 0]}),
        {"key": "id", "example": "id"},
    )
    right = sdm.TableTensor.from_pandas(
        pd.DataFrame({"key": ["a", "b", "b", None], "example": [0, 1, 0, 0]}),
        {"key": "id", "example": "id"},
    )
    cpu = join_index(left, right, ["key", "example"], ["key", "example"])
    gpu = join_index(
        left.cuda(), right.cuda(), ["key", "example"], ["key", "example"]
    )
    expected = sorted(zip(cpu[0].tolist(), cpu[1].tolist()))
    observed = sorted(zip(gpu[0].tolist(), gpu[1].tolist()))
    assert expected == observed
    strings = sdm.StringTensor.from_list(["b", "a", "c"]).cuda()
    values, indices = torch.sort(strings)
    assert indices.cpu().tolist() == [1, 0, 2]
    assert values.to_cudf().to_pandas().tolist() == ["a", "b", "c"]
    roundtrip = sdm.StringTensor.from_cudf(strings.to_cudf())
    assert roundtrip.to_cudf().to_pandas().tolist() == ["b", "a", "c"]
    torch.cuda.synchronize()
    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudf": cudf.__version__,
        "pandas": pd.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "join_pairs": observed,
        "checks": [
            "nullable composite string join",
            "CUDA string sort",
            "cuDF string roundtrip",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args().output)
