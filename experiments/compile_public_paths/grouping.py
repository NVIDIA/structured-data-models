"""Check grouping across changing rows, schemas, dtypes, and members."""

import json

import torch
from torch._dynamo.backends.registry import lookup_backend

from sdm import EnsembleTable, TableTensor


def _pack(tables):
    groups, locations = EnsembleTable._pack_tables(tables)
    return tuple(group.numerical.sin() for group in groups), locations


def _make(rows, names, dtype=torch.float32):
    values = torch.arange(rows * len(names), dtype=dtype).reshape(
        rows, len(names)
    )
    return TableTensor.from_tensor(values, columns=names)


# Order, shape-dependent grouping, changed schema, dtype, and member count.
cases = (
    ((3, ("a", "b"), torch.float32),) * 3,
    ((5, ("a", "b"), torch.float32),) * 3,
    ((7, ("a", "b"), torch.float32),) * 3,
    (
        (5, ("a",), torch.float32),
        (7, ("a",), torch.float32),
        (5, ("a",), torch.float32),
    ),
    (
        (5, ("a",), torch.float32),
        (5, ("b",), torch.float32),
        (5, ("a",), torch.float32),
    ),
    (
        (5, ("a",), torch.float32),
        (5, ("a",), torch.float64),
        (5, ("a",), torch.float32),
    ),
    ((9, ("a", "b"), torch.float32),),
    ((11, ("a", "b"), torch.float32),) * 4,
)
for fullgraph in (False, True):
    torch._dynamo.reset()
    graphs = []

    def _backend(gm, inputs, graphs=graphs):
        graphs.append(sum(n.op.startswith("call_") for n in gm.graph.nodes))
        return lookup_backend("inductor")(gm, inputs)

    compiled = torch.compile(
        _pack, backend=_backend, fullgraph=fullgraph, dynamic=True
    )
    for index, specs in enumerate(cases):
        tables = tuple(_make(*spec) for spec in specs)
        expected, locations = _pack(tables)
        actual, actual_locations = compiled(tables)
        assert locations == actual_locations
        for lhs, rhs in zip(expected, actual, strict=True):
            torch.testing.assert_close(lhs, rhs)
        print(  # noqa: T201
            json.dumps(
                {
                    "torch": torch.__version__,
                    "fullgraph": fullgraph,
                    "case": index,
                    "members": len(tables),
                    "locations": locations,
                    "graphs": len(graphs),
                    "status": "pass",
                }
            ),
            flush=True,
        )
