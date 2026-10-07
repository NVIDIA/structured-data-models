# ruff: noqa: T201
"""Reproduce the external join boundary with duplicate keys."""

import argparse

import torch

from sdm import ColumnarTensor, TableTensor
from sdm.relational.join import join_index

parser = argparse.ArgumentParser()
parser.add_argument("--fullgraph", action="store_true")
args = parser.parse_args()

left = TableTensor(
    columns={"id": ("key",)},
    id=ColumnarTensor((torch.tensor([2, 1, 2, 9, -1]),)),
)
right = TableTensor(
    columns={"id": ("key",)},
    id=ColumnarTensor((torch.tensor([1, 2, 2, -1]),)),
)


def matched_pairs(left: TableTensor, right: TableTensor) -> torch.Tensor:
    """Join externally and compile the tensor operation following the join."""
    rows, columns = join_index(left, right, ["key"], ["key"])
    return torch.stack((rows, columns), dim=-1)


with torch.inference_mode():
    expected = matched_pairs(left, right)
    compiled = torch.compile(matched_pairs, fullgraph=args.fullgraph)
    actual = compiled(left, right)
    torch.testing.assert_close(actual, expected)
print("Matching pairs:", actual.tolist())
print(
    "Captured graphs:", torch._dynamo.utils.counters["stats"]["unique_graphs"]
)
