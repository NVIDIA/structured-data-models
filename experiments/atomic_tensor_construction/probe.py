"""Check tensor computations following traceable wrapper construction."""

import torch

from sdm import CategoricalTensor, TableTensor


def _categorical(code):
    return CategoricalTensor(code, (torch.arange(3),)).code + 1


def _table(value):
    return TableTensor.from_tensor(value).numerical.sin()


for function in (_categorical, _table):
    for fullgraph in (False, True):
        torch._dynamo.reset()
        compiled = torch.compile(function, fullgraph=fullgraph, dynamic=True)
        for rows in (3, 5):
            value = torch.arange(rows).remainder(3).reshape(rows, 1)
            if function is _table:
                value = value.float()
            torch.testing.assert_close(compiled(value), function(value))
        print(function.__name__, fullgraph, "PASS", flush=True)  # noqa: T201
