"""Exercise symbolic dimensions through wrapper dispatch and CPU Inductor."""

import json

import torch

from sdm import CategoricalTensor, NullableTensor, StringTensor, VarLenTensor


def _make(kind, rows):
    if kind == "StringTensor":
        return StringTensor.from_list((["a", None, "long"] * rows)[:rows])
    values = torch.arange(rows * 2).view(rows, 2)
    if kind == "NullableTensor":
        return NullableTensor(values, values.remainder(2).bool())
    if kind == "CategoricalTensor":
        return CategoricalTensor(values.remainder(3), (torch.arange(3),) * 2)
    return VarLenTensor(
        data=torch.arange(rows * 2),
        offset=torch.arange(rows * 2 + 1),
        valid=None,
        size=(rows, 2),
    )


def _compute(value):
    rows = torch.ops.aten.sym_size.int(value, 0)
    return torch.arange(rows) + rows


for kind in (
    "StringTensor",
    "VarLenTensor",
    "NullableTensor",
    "CategoricalTensor",
):
    for fullgraph in (False, True):
        torch._dynamo.reset()
        compiled = torch.compile(_compute, fullgraph=fullgraph, dynamic=True)
        for rows in (4, 7, 3, 0):
            value = _make(kind, rows)
            for dimension in range(-value.ndim, value.ndim):
                handler = type(value).HANDLED_FUNCTIONS.get(
                    torch.ops.aten.sym_size.int
                )
                if handler is not None:
                    actual = handler(value, dimension)
                else:
                    actual = type(value).__torch_dispatch__(
                        torch.ops.aten.sym_size.int,
                        (type(value),),
                        (value, dimension),
                    )
                assert actual == value.size(dimension)
                assert torch.ops.aten.sym_size.int(value, dimension) == actual
            torch.testing.assert_close(compiled(value), _compute(value))
        print(  # noqa: T201
            json.dumps(
                {
                    "torch": torch.__version__,
                    "kind": kind,
                    "fullgraph": fullgraph,
                    "status": "pass",
                }
            ),
            flush=True,
        )
