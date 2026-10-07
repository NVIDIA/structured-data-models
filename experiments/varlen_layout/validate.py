# ruff: noqa: D103, T201
"""CPU Inductor validation; run from the repository with PYTHONPATH=."""

import argparse
import json

import torch

from sdm import CategoricalTensor, StringTensor


def categorical_roundtrip(code, first, second):
    result = CategoricalTensor(code, (first, second))
    return result, result.code.square()


def slice_and_clone(value):
    return value[1::2].clone()


def stack_empty_strings(a, b):
    return torch.stack((a, b), dim=1)


def report(case, fullgraph):
    print(
        json.dumps(
            {
                "version": torch.__version__,
                "case": case,
                "fullgraph": fullgraph,
                "dynamic": args.dynamic,
                "status": "passed",
            }
        ),
        flush=True,
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dynamic", action=argparse.BooleanOptionalAction, default=True
)
args = parser.parse_args()

for fullgraph in (False, True):
    for strings in (False, True):
        torch._dynamo.reset()
        fn = torch.compile(
            categorical_roundtrip, fullgraph=fullgraph, dynamic=args.dynamic
        )
        sliced = torch.compile(
            slice_and_clone, fullgraph=fullgraph, dynamic=args.dynamic
        )
        for rows in (5, 9, 0, 2):
            base = torch.arange(2 * (rows + 2)).reshape(2, rows + 2) % 4 - 1
            code = base[:, 1 : rows + 1].t()
            first = (
                StringTensor.from_list(["alpha", "b", ""])
                if strings
                else torch.tensor([10, 20, 30])
            )
            second = (
                StringTensor.from_list(["1", "two", "3"])
                if strings
                else torch.tensor([7.0, 3.0, 5.0])
            )
            out, square = fn(code, first, second)
            eager, expected_square = categorical_roundtrip(code, first, second)
            torch.testing.assert_close(square, expected_square)
            assert out.tolist() == eager.tolist()
            assert out.stride() == code.stride()
            assert out.storage_offset() == code.storage_offset()
            assert torch._C._is_alias_of(out.code, code)
            if strings:
                assert torch._C._is_alias_of(
                    out.categories[0]._data, first._data
                )
                assert torch._C._is_alias_of(
                    out.categories[0]._offset, first._offset
                )
            else:
                assert torch._C._is_alias_of(out.categories[0], first)
            if not strings:
                assert sliced(out).tolist() == slice_and_clone(eager).tolist()
        report(
            f"categorical_{'string' if strings else 'numeric'}_views_aliases",
            fullgraph,
        )

    torch._dynamo.reset()
    fn = torch.compile(
        stack_empty_strings, fullgraph=fullgraph, dynamic=args.dynamic
    )
    for shape in ((4, 0), (4, 2), (0, 2), (7, 2)):
        count = shape[0] * shape[1]
        for nullable in (False, True):
            a = StringTensor(
                data=torch.empty(0, dtype=torch.uint8),
                offset=torch.zeros(count + 1, dtype=torch.int32),
                valid=torch.arange(count) % 2 == 0 if nullable else None,
                size=shape,
            )
            b = StringTensor(
                data=torch.empty(0, dtype=torch.uint8),
                offset=torch.zeros(count + 1, dtype=torch.int64),
                valid=None,
                size=shape,
            )
            actual, expected = fn(a, b), stack_empty_strings(a, b)
            assert actual.tolist() == expected.tolist()
            assert actual.shape == expected.shape
            assert actual._offset.dtype == torch.int64
    report("empty_string_stack_nullable_promoted_offsets", fullgraph)
