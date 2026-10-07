# ruff: noqa: D103, T201
import torch

from sdm import StringTensor


@torch.library.custom_op("sdm_varlen_probe::cat_sum", mutates_args=())
def cat_sum(
    data1: torch.Tensor,
    offset1: torch.Tensor,
    data2: torch.Tensor,
    offset2: torch.Tensor,
) -> torch.Tensor:
    first = StringTensor(data1, offset1, None, (offset1.numel() - 1,))
    second = StringTensor(data2, offset2, None, (offset2.numel() - 1,))
    return torch.cat((first, second))._data.sum().reshape(1)


@cat_sum.register_fake
def fake(data1, offset1, data2, offset2):
    return data1.new_empty(1, dtype=torch.int64)


def fn(d1, o1, d2, o2):
    return cat_sum(d1, o1, d2, o2) + 1


for fullgraph in (False, True):
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=fullgraph)
    a = StringTensor.from_list(["a", "hello"])
    b = StringTensor.from_list(["b", ""])
    with torch.inference_mode():
        torch.testing.assert_close(
            compiled(a._data, a._offset, b._data, b._offset),
            fn(a._data, a._offset, b._data, b._offset),
        )
    print(fullgraph, "PASS", flush=True)
