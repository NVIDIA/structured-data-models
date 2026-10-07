"""Reproduce invalid constant ordering in PyTorch stream dependency tracing."""

import torch
from torch._functorch._aot_autograd.streams import (
    wrap_all_sync_nodes_with_control_deps,
)

module = torch.nn.Module()
module.register_buffer("constant", torch.tensor(0.0))
graph = torch.fx.Graph()
x = graph.placeholder("x")
x.meta["val"] = torch.ones(4)
wait = graph.call_function(torch.ops.streams.wait_stream.default, (0, 1))
wait.meta["val"] = None
constant = graph.get_attr("constant")
constant.meta["val"] = module.constant
out = graph.call_function(torch.ops.aten.add.Tensor, (x, constant))
out.meta["val"] = torch.ones(4)
out.meta["custom"] = {"stream": 0}
graph.output((out,))
gm = torch.fx.GraphModule(module, graph)
gm.graph.lint()
wrap_all_sync_nodes_with_control_deps(gm)
gm.graph.lint()
print("PASS")  # noqa: T201
