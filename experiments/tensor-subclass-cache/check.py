"""Run once per process to check compiler cache hits and invalidation."""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cache-dir", required=True)
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--numeric-only", action="store_true")
parser.add_argument("--without-hash", action="store_true")
parser.add_argument(
    "--variant",
    choices=("base", "values", "schema", "strides", "flatten"),
    default="base",
)
args = parser.parse_args()
os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.cache_dir

import torch  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

from sdm import (  # noqa: E402
    CategoricalTensor,
    ColumnarTensor,
    NullableTensor,
    StringTensor,
    TableTensor,
    VarLenTensor,
)

if args.without_hash:
    for cls in (
        TableTensor,
        ColumnarTensor,
        CategoricalTensor,
        NullableTensor,
        VarLenTensor,
    ):
        delattr(cls, "_stable_hash_for_caching")

if args.variant == "flatten":
    # Simulate changing the wrapper protocol between two source revisions.
    original_flatten = TableTensor.__tensor_flatten__

    def flatten_with_alias(self):
        """Add one alias without changing the constructor metadata."""
        names, context = original_flatten(self)
        return [*names, "_cache_probe_alias"], context

    TableTensor.__tensor_flatten__ = flatten_with_alias
    TableTensor._cache_probe_alias = property(lambda self: self._numerical)

values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
if args.variant == "values":
    values = values + 1
if args.variant == "strides":
    values = values.T.contiguous().T
names = ["a", "b", "c", "d"]
if args.variant == "schema":
    names = ["e", "f", "g", "h"]
blocks = {}
columns = {"numerical": names}
if not args.numeric_only:
    blocks["id"] = ColumnarTensor(
        (StringTensor.from_list(["a", "bb", "cc", "d", "eee", "f"]),)
    )
    columns["id"] = ["id"]
table = TableTensor(numerical=values, columns=columns, **blocks)


def forward(inp):
    """Exercise wrapper slicing and ordinary numerical tensor work."""
    return inp[:3].numerical.sin()


compiled = torch.compile(forward, fullgraph=True, dynamic=args.dynamic)
torch.testing.assert_close(compiled(table), forward(table))
print(  # noqa: T201
    json.dumps(
        {
            "torch": torch.__version__,
            "variant": args.variant,
            "dynamic": args.dynamic,
            "numeric_only": args.numeric_only,
            "metadata_hash": not args.without_hash,
            "counters": {key: dict(value) for key, value in counters.items()},
            "parity": "PASS",
        }
    ),
    flush=True,
)
