"""Check exact string lookup parity through real CPU Inductor."""

import json

import torch

from sdm import StringTensor
from sdm.processing import AlignCategories

CASES = [
    (["", "é", "a\x00b", "red", "missing"], ["red", "a\x00b", "é", ""]),
    (["a", "a", "b", ""], ["b", "a", "a", ""]),
    ([], ["a"]),
    (["a", ""], []),
]

for fullgraph in (False, True):
    for dtype in (torch.int32, torch.int64):
        for left, right in CASES:
            torch._dynamo.reset()
            # Two groups intentionally share strings; matches stay group-local.
            inputs = (
                StringTensor.from_list(left),
                StringTensor.from_list(["red", "other"]),
            )
            fitted = (
                StringTensor.from_list(right),
                StringTensor.from_list(["other", "red"]),
            )
            codes = torch.empty(0, dtype=dtype)
            expected = AlignCategories._string_category_lookups(
                inputs, fitted, codes
            )
            compiled = torch.compile(
                AlignCategories._string_category_lookups,
                fullgraph=fullgraph,
            )
            actual = compiled(inputs, fitted, codes)
            for actual_column, expected_column in zip(
                actual, expected, strict=True
            ):
                torch.testing.assert_close(
                    actual_column, expected_column, rtol=0, atol=0
                )
    print(  # noqa: T201
        json.dumps(
            {
                "torch": torch.__version__,
                "fullgraph": fullgraph,
                "cases": 8,
                "status": "pass",
            }
        ),
        flush=True,
    )
