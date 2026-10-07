# ruff: noqa: D103, BLE001, PLC0415
"""Record actual compiled regions and context-fitted state parity."""

import argparse
import json
import traceback
from pathlib import Path

import torch
from torch._dynamo.backends.registry import lookup_backend

from sdm import EnsembleTable, TableTensor
from sdm.models.kumo.relational.recipe import (
    default_recipe as relational_recipe,
)
from sdm.models.kumo.tabular.recipe import default_recipe as tabular_recipe
from sdm.processing import TableDispatch

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--fullgraph", action="store_true")
p.add_argument("--members", type=int, default=1)
p.add_argument("--data")
p.add_argument("--missing", action="store_true")
p.add_argument("--model", choices=["tabular", "relational"], default="tabular")
p.add_argument("--fallback-random", action="store_true")
p.add_argument("--native-sum", action="store_true")
p.add_argument("--cache-limit", type=int, default=None)
a = p.parse_args()
output = Path(a.output)
output.mkdir(parents=True, exist_ok=True)
result = {
    "torch": torch.__version__,
    "fullgraph": a.fullgraph,
    "members": a.members,
    "fallback_random": a.fallback_random,
    "native_sum": a.native_sum,
    "missing": a.missing,
    "cache_limit": a.cache_limit,
    "source": f"default Kumo{a.model} features.fit_transform_ensemble",
}
graphs = []


def backend(gm, inputs):
    record = {
        "nodes": len(list(gm.graph.nodes)),
        "calls": [
            str(n.target)
            for n in gm.graph.nodes
            if n.op in ("call_function", "call_method")
        ],
    }
    graphs.append(record)
    (output / f"graph-{len(graphs):02d}.py").write_text(gm.code)
    if a.native_sum:
        from native_sum_backend import backend as native_backend

        return native_backend(gm, inputs)
    return lookup_backend("inductor")(gm, inputs)


try:
    if a.cache_limit is not None:
        torch._dynamo.config.cache_size_limit = a.cache_limit
    if a.data:
        data = torch.load(a.data, weights_only=True)[0][:160]
        data = torch.cat((data, data.new_ones((len(data), 1))), dim=-1)
    else:
        data = torch.arange(240, dtype=torch.float32).reshape(40, 6).sin()
        data[:, 0] = 1
    if a.missing:
        data = data.clone()
        data[::7, ::3] = torch.nan
    table = EnsembleTable.from_table(
        TableTensor.from_tensor(data), num_members=a.members
    )
    factory = tabular_recipe if a.model == "tabular" else relational_recipe
    reference = factory().features
    processor = factory().features
    # RecipeExecution resolves this task-table route before fitting features.
    for pipeline in (reference, processor):
        for module in pipeline.modules():
            if isinstance(module, TableDispatch):
                module._route = "task"
    with (
        torch.inference_mode(),
        torch._inductor.config.patch(fallback_random=a.fallback_random),
    ):
        torch.manual_seed(8)
        expected = reference.fit_transform_ensemble(table)
        torch.manual_seed(8)
        torch._dynamo.utils.counters.clear()
        actual = torch.compile(
            processor.fit_transform_ensemble,
            backend=backend,
            fullgraph=a.fullgraph,
        )(table)
        errors = []
        parity_errors = []
        for idx in range(a.members):
            ref, out = expected[idx], actual[idx]
            assert ref.columns == out.columns
            errors.append(float((ref.numerical - out.numerical).abs().max()))
            try:
                torch.testing.assert_close(out.numerical, ref.numerical)
            except AssertionError as error:
                parity_errors.append(str(error))
        mismatches = []
        exact_mismatches = []
        ref_buffers = dict(reference.named_buffers())
        for name, value in processor.named_buffers():
            ref = ref_buffers[name]
            if ref.shape != value.shape or not torch.allclose(
                ref, value, rtol=0, atol=0, equal_nan=True
            ):
                exact_mismatches.append(
                    {
                        "name": name,
                        "max_abs": float((ref - value).abs().max())
                        if value.numel()
                        and ref.shape == value.shape
                        and value.dtype != torch.bool
                        else None,
                    }
                )
            if ref.shape != value.shape or not torch.allclose(
                ref, value, equal_nan=True
            ):
                mismatches.append(
                    {
                        "name": name,
                        "shape": list(value.shape),
                        "expected_shape": list(ref.shape),
                        "max_abs": float((ref - value).abs().max())
                        if value.numel()
                        and ref.shape == value.shape
                        and value.dtype != torch.bool
                        else None,
                    }
                )
        result.update(
            status="PARITY_FAIL" if parity_errors else "PASS",
            parity_errors=parity_errors,
            member_max_abs=errors,
            max_abs=max(errors),
            buffer_mismatches=mismatches,
            exact_buffer_mismatches=exact_mismatches,
            data="breast_cancer160" if a.data else "synthetic40",
            shapes=[list(actual[i].shape) for i in range(a.members)],
            stats=dict(torch._dynamo.utils.counters["stats"]),
            graph_breaks=dict(torch._dynamo.utils.counters["graph_break"]),
        )
except Exception as e:
    traceback.print_exc()
    result.update(status="FAIL", error=str(e).splitlines()[0])
result["graphs"] = graphs
(output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
