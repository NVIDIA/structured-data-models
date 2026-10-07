# ruff: noqa: D103, T201, BLE001
import torch
from torch._dynamo import config


def fn(data, offsets):
    start = offsets[0].item()
    end = offsets[-1].item()
    torch._check_is_size(start)
    torch._check_is_size(end)
    torch._check(end >= start)
    torch._check(end <= data.numel())
    return data.narrow(0, start, end - start)


for capture in (False, True):
    for fullgraph in (False, True):
        torch._dynamo.reset()
        with config.patch(capture_scalar_outputs=capture):
            try:
                f = torch.compile(fn, fullgraph=fullgraph, dynamic=True)
                for offsets in (
                    torch.tensor([1, 4]),
                    torch.tensor([2, 6]),
                    torch.tensor([6, 6]),
                ):
                    data = torch.arange(10)
                    out = f(data, offsets)
                    assert torch.equal(out, fn(data, offsets))
                    assert torch._C._is_alias_of(out, data)
                print(capture, fullgraph, "PASS", flush=True)
            except Exception as e:
                print(capture, fullgraph, type(e).__name__, str(e), flush=True)
