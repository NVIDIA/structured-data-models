from enum import StrEnum
import torch


class Kind(StrEnum):
    numerical = "numerical"
    categorical = "categorical"


class Wrapped(torch.Tensor):
    @staticmethod
    def __new__(cls, data):
        out = torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
            strides=data.stride(),
        )
        out.data_leaf = data
        return out

    @property
    def columns(self):
        return {Kind.numerical: ("x", "y"), Kind.categorical: ()}

    def __tensor_flatten__(self):
        return ["data_leaf"], None

    @staticmethod
    def __tensor_unflatten__(leaves, ctx, size, stride):
        return Wrapped(leaves["data_leaf"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(func)


def fn(x):
    return x.data_leaf * len(x.columns[Kind.numerical])


for full in (False, True):
    torch.compiler.reset()
    try:
        got = torch.compile(fn, fullgraph=full)(Wrapped(torch.ones(2)))
        print(torch.__version__, full, "PASS", got)
    except Exception as e:
        print(torch.__version__, full, type(e).__name__, str(e)[:1500])
