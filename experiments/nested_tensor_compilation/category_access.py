# ruff: noqa: D103, T201
import torch

from sdm import CategoricalTensor


def static(code, category):
    result = CategoricalTensor(code, (category,))
    return CategoricalTensor.category(result, 0) + 1


def bound(code, category):
    result = CategoricalTensor(code, (category,))
    return result.category(0) + 1


for f in (static, bound):
    for fullgraph in (False, True):
        torch._dynamo.reset()
        out = torch.compile(f, fullgraph=fullgraph)(
            torch.tensor([[0], [1]]), torch.arange(3)
        )
        torch.testing.assert_close(out, torch.arange(3) + 1)
        print(f.__name__, fullgraph, "PASS", flush=True)
