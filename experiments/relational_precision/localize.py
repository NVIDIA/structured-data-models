"""Capture the first neural divergence without changing model arithmetic."""
# ruff: noqa: D103, T201

import argparse
import json
from pathlib import Path

import torch

from sdm.models import KumoRelational


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", type=int, default=1)
    parser.add_argument("--query", type=int, default=4)
    parser.add_argument("--backend", default="eager")
    parser.add_argument("--first-norm", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    arm = torch.load(args.data, weights_only=False)["arms"][args.arm]
    model = KumoRelational(
        task="classification", pretrained=False, device="cpu"
    )
    core = model.models["classification"]
    core.load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    )
    records = []
    linear_inputs = []

    def make_hook(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            records.append((name, value.detach().clone()))

        return hook

    def capture_linear(module, inputs, kwargs):
        out = kwargs["out"]
        linear_inputs.append((inputs[0].clone(), out.shape, out.stride()))

    query = arm["queries"][args.query]
    with torch.inference_mode():
        model.fit(
            arm["context"],
            arm["context_target"],
            arm["related_context"],
            generator=torch.Generator().manual_seed(123),
            num_estimators=1,
            num_hops=arm["num_hops"],
        )
        names = ["row_embedding.lin", "row_embedding", "gnn", "icl_block"]
        if args.first_norm:
            names.append("row_embedding.col_layers.0.output_block.query_norm")
        handles = [
            core.get_submodule(name).register_forward_hook(make_hook(name))
            for name in names
        ]
        pre = core.row_embedding.lin.register_forward_pre_hook(
            capture_linear, with_kwargs=True
        )
        expected = model.predict(
            query["x"], query["related"]
        ).numerical.clone()
        pre.remove()
        eager_records = list(records)
        records.clear()
        core.compile(backend=args.backend, fullgraph=True, dynamic=True)
        actual = model.predict(query["x"], query["related"]).numerical.clone()
        comparison = []
        for (name, before), (name2, after) in zip(
            eager_records, records, strict=True
        ):
            assert name == name2
            comparison.append(
                {
                    "name": name,
                    "shape": list(before.shape),
                    "equal": torch.equal(before, after),
                    "max_abs_error": (before - after).abs().max().item(),
                }
            )
        result = {
            "torch": torch.__version__,
            "backend": args.backend,
            "arm": args.arm,
            "query": args.query,
            "record_counts": [len(eager_records), len(records)],
            "parity": torch.allclose(actual, expected, atol=1e-5, rtol=1e-4),
            "max_abs_error": (actual - expected).abs().max().item(),
            "layers": comparison,
        }
        output = Path(args.output)
        output.write_text(json.dumps(result, indent=2) + "\n")
        torch.save(
            {
                "inputs": linear_inputs,
                "weight": core.row_embedding.lin.weight,
                "bias": core.row_embedding.lin.bias,
            },
            output.with_suffix(".pt"),
        )
        for handle in handles:
            handle.remove()
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
