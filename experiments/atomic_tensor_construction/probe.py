import torch
from sdm import CategoricalTensor, TableTensor


def fn(x):
    return CategoricalTensor(x, (torch.arange(3),)).code + 1


def table(x):
    return TableTensor.from_tensor(x).numerical.sin()


for func in (fn, table):
    for fullgraph in (False, True):
        torch._dynamo.reset()
        try:
            f = torch.compile(func, fullgraph=fullgraph, dynamic=True)
            for n in (3, 5):
                x = torch.arange(n).remainder(3).reshape(n, 1)
                if func is table:
                    x = x.float()
                torch.testing.assert_close(f(x), func(x))
            print(func.__name__, fullgraph, "PASS", flush=True)
        except Exception as e:
            print(
                func.__name__, fullgraph, type(e).__name__, str(e), flush=True
            )
