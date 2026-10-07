# ruff: noqa: D101, D102, D103, T201
import torch


class Wrapper(torch.Tensor):
    def __new__(cls, data, size, stride=None):
        if stride is None:
            stride = (size[1] * size[2], size[2], 1)
        out = torch.Tensor._make_wrapper_subclass(
            cls,
            size=size,
            strides=stride,
            dtype=data.dtype,
            device=data.device,
        )
        out.data_tensor = data
        return out

    def __tensor_flatten__(self):
        return ["data_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner, ctx, size, stride):
        return Wrapper(inner["data_tensor"], size, stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(func)

    def __repr__(self):
        return f"Wrapper(size={self.size()},stride={self.stride()})"


def fn(x):
    shape = (x.shape[0], 2, x.shape[1])
    return Wrapper(x.new_zeros(x.numel() * 2 + 1), shape)


f = torch.compile(fn, fullgraph=True, dynamic=True)
for rows in (4, 7):
    out = f(torch.zeros(rows, 2))
    print(out, flush=True)
    assert out.stride() == (4, 2, 1)
