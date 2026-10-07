# ruff: noqa: D103, T201
import argparse
import contextlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models import KumoRelational, KumoTabular

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["tabular", "relational"], required=True)
p.add_argument("--output", required=True)
p.add_argument("--data", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--entry", choices=["fit", "predict", "forward"], required=True)
p.add_argument("--fullgraph", action="store_true")
p.add_argument("--dynamic", choices=["true", "false"], default="true")
p.add_argument("--backend", default="inductor")
p.add_argument(
    "--task",
    choices=["classification", "regression"],
    default="classification",
)
p.add_argument("--estimators", type=int, default=1)
p.add_argument("--query-rows", type=int, nargs="+")
p.add_argument("--query-input", choices=["view", "fresh"], default="view")
p.add_argument("--arm-index", type=int, default=0)
p.add_argument("--relational-query-indices", type=int, nargs="+")
p.add_argument("--recompile-limit", type=int)
p.add_argument("--include-predictions", action="store_true")
p.add_argument("--inner-only", action="store_true")
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(123)
if a.model == "tabular":
    data = np.load(a.data)
    cx = TableTensor.from_tensor(
        torch.tensor(data["x"][data["train_ids"][:32]])
    )
    target = torch.tensor(data["y"][data["train_ids"][:32]]).view(-1, 1)
    cy = TableTensor.from_tensor(
        CategoricalTensor.from_tensor(target.long())
        if a.task == "classification"
        else target.to(torch.float32)
    )
    qx = TableTensor.from_tensor(
        torch.tensor(
            data["x"][data["validation_ids"][: max(a.query_rows or [4])]]
        )
    )
    cr = qr = None
else:
    b = torch.load(a.data, weights_only=False)
    arm = b["arms"][a.arm_index]
    cx, cy, cr = arm["context"], arm["context_target"], arm["related_context"]
    qx, qr = arm["queries"][1]["x"], arm["queries"][1]["related"]
ckpt = torch.load(a.checkpoint, weights_only=True, map_location="cpu")


def new_model():
    torch.manual_seed(123)
    model = (
        KumoTabular(task=a.task, size="small", pretrained=False, device="cpu")
        if a.model == "tabular"
        else KumoRelational(task=a.task, pretrained=False, device="cpu")
    )
    model.models[a.task].load_state_dict(ckpt, strict=True)
    return model


def fit(model, function=None):
    kwargs = {
        "generator": torch.Generator().manual_seed(123),
        "num_estimators": a.estimators,
    }
    if a.model == "relational":
        kwargs["num_hops"] = arm["num_hops"]
    return (model.fit if function is None else function)(cx, cy, cr, **kwargs)


def predict(model, function=None):
    return (model.predict if function is None else function)(qx, qr)


def forward(model, function=None):
    kwargs = {
        "generator": torch.Generator().manual_seed(123),
        "num_estimators": a.estimators,
    }
    if a.model == "relational":
        kwargs["num_hops"] = arm["num_hops"]
    return (model if function is None else function)(
        cx, cy, qx, cr, qr, **kwargs
    )


result = {
    "torch": torch.__version__,
    "model": a.model,
    "entry": a.entry,
    "fullgraph": a.fullgraph,
    "dynamic": a.dynamic == "true",
    "backend": a.backend,
    "device": "cpu",
    "task": a.task,
    "estimators": a.estimators,
    "query_input": a.query_input,
    "arm_index": a.arm_index if a.model == "relational" else None,
    "recompile_limit": a.recompile_limit,
    "inner_only": a.inner_only,
}
compiler_config = (
    contextlib.nullcontext()
    if a.recompile_limit is None
    else torch._dynamo.config.patch(cache_size_limit=a.recompile_limit)
)
try:
    with torch.inference_mode(), compiler_config:
        oracle = new_model()
        if a.entry == "forward":
            expected = forward(oracle).numerical.clone()
        else:
            fit(oracle)
            expected = predict(oracle).numerical.clone()
        result["eager_succeeded"] = True
        model = new_model()
        kwargs = {
            "backend": a.backend,
            "fullgraph": a.fullgraph,
            "dynamic": a.dynamic == "true",
        }
        if a.entry == "fit":
            fit(model, torch.compile(model.fit, **kwargs))
            actual = predict(model).numerical
        elif a.entry == "predict":
            fit(model)
            torch._dynamo.utils.counters.clear()
            if a.inner_only:
                for inner_model in model.models.values():
                    inner_model.compile(**kwargs)
                compiled_predict = model.predict
            else:
                compiled_predict = torch.compile(model.predict, **kwargs)
            all_queries = qx
            samples = []
            cases = (
                a.relational_query_indices
                if a.model == "relational"
                else a.query_rows
            )
            for case in cases or [qx.size(-2)]:
                rows = case
                if (
                    a.model == "relational"
                    and a.relational_query_indices is not None
                ):
                    query_case = arm["queries"][case]
                    qx, qr = query_case["x"], query_case["related"]
                    rows = qx.size(-2)
                elif a.query_rows is None:
                    qx = all_queries
                elif a.query_input == "fresh" and a.model == "tabular":
                    qx = TableTensor.from_tensor(
                        all_queries.numerical[..., :rows, :]
                    )
                else:
                    qx = all_queries[..., :rows, :]
                expected = predict(oracle).numerical.clone()
                actual = predict(model, compiled_predict).numerical
                samples.append(
                    {
                        "rows": rows,
                        "case": case,
                        "max_abs_error": (actual - expected)
                        .abs()
                        .max()
                        .item(),
                        "parity": bool(
                            torch.allclose(
                                actual, expected, atol=1e-5, rtol=1e-4
                            )
                        ),
                        "graphs_captured": torch._dynamo.utils.counters[
                            "stats"
                        ]["unique_graphs"],
                        "calls_captured": torch._dynamo.utils.counters[
                            "stats"
                        ]["calls_captured"],
                    }
                )
                if a.include_predictions:
                    samples[-1]["expected"] = expected.tolist()
                    samples[-1]["actual"] = actual.tolist()
                result["samples"] = samples
            result["all_samples_parity"] = all(
                sample["parity"] for sample in samples
            )
        else:
            actual = forward(model, torch.compile(model, **kwargs)).numerical
        result.update(
            status="pass",
            max_abs_error=(actual - expected).abs().max().item(),
            parity=bool(
                torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
            ),
        )
except Exception as error:  # noqa: BLE001 - record compiler failures for the matrix
    result.update(
        status="fail",
        error_type=type(error).__name__,
        error=str(error),
        traceback=traceback.format_exc(),
    )
if result.get("samples"):
    result["max_abs_error"] = max(
        sample["max_abs_error"] for sample in result["samples"]
    )
    result["parity"] = all(sample["parity"] for sample in result["samples"])
if result["status"] == "pass" and not result["parity"]:
    result["status"] = "parity_fail"
Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in {"traceback", "error"}}
    ),
    flush=True,
)
if result["status"] == "fail":
    print(result["error"], flush=True)
