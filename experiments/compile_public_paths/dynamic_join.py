"""Exercise the external variable-output join under fullgraph compilation."""

# ruff: noqa: T201, BLE001
import json

import torch

from sdm import ColumnarTensor, NullableTensor, StringTensor, TableTensor
from sdm.relational.join import join_index


def table(*columns):
    """Build a table containing join keys."""
    return TableTensor(
        columns={"id": tuple(f"k{i}" for i in range(len(columns)))},
        id=ColumnarTensor(columns),
    )


def matched(left, right, dtype, names):
    """Compile the join and an operation consuming its dynamic output."""
    a, b = join_index(left, right, names, names, dtype=dtype)
    return torch.stack((a, b), dim=-1)


cases = [
    (
        "duplicates",
        table(torch.tensor([2, 1, 2, 9, -1])),
        table(torch.tensor([1, 2, 2, -1])),
    ),
    (
        "empty_left",
        table(torch.empty(0, dtype=torch.int64)),
        table(torch.tensor([1, 2])),
    ),
    (
        "empty_right",
        table(torch.tensor([1, 2])),
        table(torch.empty(0, dtype=torch.int64)),
    ),
    ("no_matches", table(torch.tensor([8, 9])), table(torch.tensor([1, 2]))),
    (
        "float_nan_cast",
        table(torch.tensor([1.0, float("nan"), 3.0])),
        table(torch.tensor([1, 2, 3])),
    ),
    (
        "nullable",
        table(
            NullableTensor(
                torch.tensor([1, 2, 3]), torch.tensor([True, False, True])
            )
        ),
        table(torch.tensor([1, 2, 3])),
    ),
    (
        "strings",
        table(StringTensor.from_list(["a", None, "b", "a"])),
        table(StringTensor.from_list(["b", "a", None])),
    ),
    (
        "multikey",
        table(torch.tensor([1, 1, 2]), torch.tensor([2, 3, 4])),
        table(torch.tensor([1, 2]), torch.tensor([3, 4])),
    ),
]
rows = []
with (
    torch.inference_mode(),
    torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True),
):
    for name, left, right in cases:
        for dtype in (torch.int64, torch.int32, torch.uint8):
            torch._dynamo.reset()
            try:
                names = tuple(left.columns["id"])
                expected = matched(left, right, dtype, names)
                actual = torch.compile(matched, fullgraph=True, dynamic=True)(
                    left, right, dtype, names
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                row = {
                    "case": name,
                    "dtype": str(dtype),
                    "status": "pass",
                    "pairs": actual.tolist(),
                }
            except Exception as e:
                row = {
                    "case": name,
                    "dtype": str(dtype),
                    "status": "fail",
                    "error": str(e),
                }
            rows.append(row)
            print(json.dumps(row), flush=True)
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "passed": sum(r["status"] == "pass" for r in rows),
            "total": len(rows),
        }
    ),
    flush=True,
)
