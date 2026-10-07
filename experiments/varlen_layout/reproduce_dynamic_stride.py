# ruff: noqa: D103, T201
import torch
from torch._subclasses.fake_tensor import is_fake

from sdm import StringTensor, VarLenTensor

old = VarLenTensor.__tensor_unflatten__


def watch(inner, ctx, size, stride):
    if not is_fake(inner["_data"]):
        print("REBUILD", ctx, size, stride, inner["_offset"].shape, flush=True)
    return old(inner, ctx, size, stride)


VarLenTensor.__tensor_unflatten__ = staticmethod(watch)


def fn(a, b):
    return torch.stack((a, b), dim=1)


f = torch.compile(fn, fullgraph=True, dynamic=True)
for shape in ((4, 2), (7, 2)):
    n = shape[0] * shape[1]
    a = StringTensor(
        torch.empty(0, dtype=torch.uint8),
        torch.zeros(n + 1, dtype=torch.int64),
        torch.arange(n) % 2 == 0,
        shape,
    )
    out = f(a, a)
    print("OUTPUT", shape, out.shape, out.stride(), flush=True)
