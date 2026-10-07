# ruff: noqa: T201, D103
import argparse
import json
import math

import torch

from sdm.nn.linear import Linear

torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True)
parser.add_argument(
    "--dtype", choices=["float32", "bfloat16"], default="float32"
)
args = parser.parse_args()
data = torch.load(args.inputs, weights_only=False)
dtype = getattr(torch, args.dtype)
weight, bias = data["weight"].to(dtype), data["bias"].to(dtype)
model = Linear(weight.size(1), weight.size(0))
model.load_state_dict({"weight": weight, "bias": bias})
results = []
with torch.inference_mode():
    for x, shape, stride in data["inputs"]:
        x = x.to(dtype)

        def make(shape=shape, stride=stride, dtype=dtype):
            return torch.empty_strided(shape, stride, dtype=dtype)

        eager = model(x, out=make())
        compiled = torch.compile(
            model, backend="eager", fullgraph=True, dynamic=True
        )(x, out=make())
        bmm = torch.bmm(
            x.view(math.prod(x.shape[:-2]), x.size(-2), x.size(-1)),
            weight.t().expand(math.prod(x.shape[:-2]), -1, -1),
        ).view(shape)
        bmm_out = make()
        bmm_out.copy_(bmm)
        bmm_out += bias
        mm_out = make()
        mm_out.copy_(torch.matmul(x, weight.t()))
        mm_out += bias
        results.append(
            {
                "shape": list(shape),
                "stride": list(stride),
                "compiled_error": (compiled - eager).abs().max().item(),
                "matmul_error": (mm_out - eager).abs().max().item(),
                "bmm_error": (bmm_out - eager).abs().max().item(),
                "compiled_equal_matmul": torch.equal(compiled, mm_out),
                "bmm_equal_eager": torch.equal(bmm_out, eager),
            }
        )
print(json.dumps(results, indent=2))
