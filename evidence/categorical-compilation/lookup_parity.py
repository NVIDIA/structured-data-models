"""Compare numeric category lookups, including special values."""

import json

import torch

from sdm.processing import AlignCategories


def original(input_categories, fitted_categories, codes):
    """Run the original lookup for a parity reference."""
    lookup = codes.new_full((input_categories.numel(),), -1)
    if fitted_categories.numel() == 0:
        return lookup
    if input_categories.dtype in {
        torch.bool,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }:
        input_categories = input_categories.to(torch.int64)
        fitted_categories = fitted_categories.to(torch.int64)
    sorted_categories, perm = fitted_categories.sort()
    position = torch.searchsorted(sorted_categories, input_categories)
    position = position.clamp(max=sorted_categories.numel() - 1)
    match = sorted_categories[position] == input_categories
    left_index = match.nonzero().view(-1)
    right_index = perm[position[left_index]]
    lookup[left_index] = right_index.to(codes.dtype)
    return lookup


cases = [
    (
        "duplicates_order",
        torch.tensor([10, 20, 10, 99]),
        torch.tensor([20, 10, 10]),
    ),
    (
        "nan_infinity",
        torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0, 0.0]),
        torch.tensor([float("inf"), -float("inf"), 1.0, float("nan")]),
    ),
    ("bool", torch.tensor([True, False]), torch.tensor([False])),
    ("empty_fitted", torch.tensor([1, 2]), torch.empty(0, dtype=torch.int64)),
    ("empty_input", torch.empty(0, dtype=torch.int64), torch.tensor([1, 2])),
]
for dtype in [torch.uint16, torch.uint32, torch.uint64]:
    largest = torch.iinfo(dtype).max
    cases.append(
        (
            str(dtype),
            torch.tensor([largest, 0, 5], dtype=dtype),
            torch.tensor([0, largest], dtype=dtype),
        )
    )
for fullgraph in (False, True):
    for _name, categories, fitted in cases:
        for dtype in (torch.int32, torch.int64):
            torch._dynamo.reset()
            fn = torch.compile(
                AlignCategories._category_lookup, fullgraph=fullgraph
            )
            codes = torch.empty(0, dtype=dtype)
            expected = original(categories, fitted, codes)
            torch.testing.assert_close(
                AlignCategories._category_lookup(categories, fitted, codes),
                expected,
                rtol=0,
                atol=0,
            )
            actual = fn(categories, fitted, codes)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(  # noqa: T201
        json.dumps(
            {
                "version": torch.__version__,
                "fullgraph": fullgraph,
                "cases": len(cases) * 2,
                "status": "pass",
            }
        ),
        flush=True,
    )
