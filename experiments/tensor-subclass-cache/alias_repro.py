"""Reproduce the native PyTorch 2.14 AOT cache alias/mutation failure."""

import json

import torch
from torch._dynamo.utils import counters


def forward(left, right):
    """Mutate one argument and read the other argument."""
    left.add_(1)
    return right.clone()


for shared in (True, False, True):
    torch._dynamo.reset()
    storage = torch.zeros(3)
    left = storage[:]
    right = storage[:] if shared else torch.zeros(3)
    with torch.inference_mode():
        actual = torch.compile(forward, fullgraph=True)(left, right)
    expected = torch.ones(3) if shared else torch.zeros(3)
    print(  # noqa: T201
        json.dumps(
            {
                "torch": torch.__version__,
                "shared": shared,
                "actual": actual.tolist(),
                "expected": expected.tolist(),
                "aot": dict(counters["aot_autograd"]),
                "inductor": dict(counters["inductor"]),
            }
        ),
        flush=True,
    )
    torch.testing.assert_close(actual, expected)
