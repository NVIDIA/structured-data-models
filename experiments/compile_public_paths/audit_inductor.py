"""Save public-probe graphs and compile them with the real Inductor backend."""

import json
import os
import runpy
from pathlib import Path

from torch._dynamo.backends.registry import lookup_backend, register_backend

backend = lookup_backend("inductor")
output = Path(os.environ["SDM_GRAPH_DIRECTORY"])
output.mkdir(parents=True, exist_ok=False)


@register_backend(name="audited_inductor")
def audited_inductor(graph_module, inputs):
    """Save graph code and source metadata, then compile it with Inductor."""
    index = len(list(output.glob("graph*.py")))
    (output / f"graph{index}.py").write_text(graph_module.code)
    nodes = [
        {
            "op": node.op,
            "target": str(node.target),
            "source": node.meta.get("stack_trace", ""),
        }
        for node in graph_module.graph.nodes
        if node.op.startswith("call")
    ]
    (output / f"graph{index}.json").write_text(json.dumps(nodes, indent=2))
    return backend(graph_module, inputs)


runpy.run_path(str(Path(__file__).with_name("probe.py")), run_name="__main__")
