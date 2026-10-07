# ruff: noqa: D103
import torch


@torch.library.custom_op("sdm_investigation::native_nansum", mutates_args=())
def native_sum(x: torch.Tensor) -> torch.Tensor:
    return x.nansum(-2, keepdim=True)


@native_sum.register_fake
def _(x):
    return x.new_empty((*x.shape[:-2], 1, x.shape[-1]))


@torch.library.custom_op("sdm_investigation::native_nanmean", mutates_args=())
def native_mean(x: torch.Tensor) -> torch.Tensor:
    return x.nanmean(-2, keepdim=True)


@native_mean.register_fake
def _(x):
    return x.new_empty((*x.shape[:-2], 1, x.shape[-1]))


def backend(gm, inputs):
    replaced = 0
    for node in gm.graph.nodes:
        if (
            (
                (node.op == "call_method" and node.target == "nansum")
                or (
                    node.op == "call_function" and node.target is torch.nanmean
                )
            )
            and node.kwargs.get(
                "dim", node.args[1] if len(node.args) > 1 else None
            )
            == -2
            and node.kwargs.get(
                "keepdim", node.args[2] if len(node.args) > 2 else False
            )
            is True
            and node.kwargs.get("dtype") is None
        ):
            target = (
                torch.ops.sdm_investigation.native_nanmean.default
                if node.target is torch.nanmean
                else torch.ops.sdm_investigation.native_nansum.default
            )
            node.op = "call_function"
            node.target = target
            node.args = (node.args[0],)
            node.kwargs = {}
            replaced += 1
    gm.recompile()
    return torch._inductor.compile(gm, inputs)
