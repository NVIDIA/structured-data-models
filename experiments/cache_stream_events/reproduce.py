"""Compare stream and event dependency rewriting without requiring a GPU."""

import torch
from torch._functorch._aot_autograd.streams import (
    wrap_all_sync_nodes_with_control_deps,
)


def _make_graph(use_events):
    module = torch.nn.Module()
    module.register_buffer("constant", torch.tensor(0.0))
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    value.meta["val"] = torch.ones(4)
    if use_events:
        record = graph.call_function(
            torch.ops.streams.record_event.default, (0, 1)
        )
        record.meta["val"] = None
        wait = graph.call_function(
            torch.ops.streams.wait_event.default, (0, 0)
        )
    else:
        wait = graph.call_function(
            torch.ops.streams.wait_stream.default, (0, 1)
        )
    wait.meta["val"] = None
    constant = graph.get_attr("constant")
    constant.meta["val"] = module.constant
    output = graph.call_function(torch.ops.aten.add.Tensor, (value, constant))
    output.meta["val"] = torch.ones(4)
    output.meta["custom"] = {"stream": 0}
    graph.output((output,))
    return torch.fx.GraphModule(module, graph)


for use_events in (False, True):
    module = _make_graph(use_events)
    module.graph.lint()
    wrap_all_sync_nodes_with_control_deps(module)
    try:
        module.graph.lint()
    except RuntimeError as error:
        if use_events or "used before it has been defined" not in str(error):
            raise
        print("wait_stream: reproduced invalid constant ordering")  # noqa: T201
    else:
        print(f"events={use_events}: graph ordering is valid")  # noqa: T201
