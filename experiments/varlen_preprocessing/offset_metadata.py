# ruff: noqa: D103, T201
import sys

import torch

from sdm import StringTensor

kind = sys.argv[1]
fullgraph = sys.argv[2] == "true"
dynamic = sys.argv[3] == "true"


def fn(a):
    return a._offset[a._storage_offset : a._storage_offset + a.numel() + 1] + 1


f = torch.compile(fn, fullgraph=fullgraph, dynamic=dynamic)
for values in (
    ["a", "longer", "hello", "x", ""],
    ["b", "world", "z", "x", "long", "extra"],
):
    source = StringTensor.from_list(values)
    a = source if kind == "full" else source[1:]
    if kind == "construct":
        a = StringTensor(
            source._data,
            source._offset,
            source._valid,
            (len(values) - 1,),
            storage_offset=1,
        )
    assert a._storage_offset == a.storage_offset()
    torch.testing.assert_close(f(a), fn(a))
print(kind, fullgraph, dynamic, "PASS", flush=True)
