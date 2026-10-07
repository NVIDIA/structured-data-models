# ruff: noqa: D103, T201
import argparse
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
p.add_argument("--backend", default="inductor")
p.add_argument("--cache-limit", type=int, default=64)
p.add_argument("--native-sum", action="store_true")
p.add_argument("--atomic-categorical", action="store_true")
p.add_argument("--atomic-table", action="store_true")
p.add_argument("--prepared-recipe", action="store_true")
p.add_argument("--atomic-recipe", action="store_true")
p.add_argument(
    "--task",
    choices=["classification", "regression"],
    default="classification",
)
p.add_argument("--estimators", type=int, default=1)
p.add_argument("--query-rows", type=int, nargs="+")
p.add_argument("--query-input", choices=["view", "fresh"], default="view")
a = p.parse_args()
if a.atomic_categorical:
    CategoricalTensor.__new__ = staticmethod(
        torch.compiler.disable(CategoricalTensor.__new__)
    )
if a.atomic_table:
    TableTensor.__new__ = staticmethod(
        torch.compiler.disable(TableTensor.__new__)
    )
if a.atomic_recipe:
    KumoTabular.default_recipe = classmethod(
        torch.compiler.disable(KumoTabular.default_recipe.__func__)
    )
    KumoRelational.default_recipe = classmethod(
        torch.compiler.disable(KumoRelational.default_recipe.__func__)
    )
torch._dynamo.config.cache_size_limit = a.cache_limit
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
    arm = b["arms"][0]
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


fit_rng_states = {}


def fit(model, function=None):
    kwargs = {
        "generator": torch.Generator().manual_seed(123),
        "num_estimators": a.estimators,
    }
    if a.model == "relational":
        kwargs["num_hops"] = arm["num_hops"]
    if a.prepared_recipe:
        kwargs["recipe"] = model.default_recipe()
    output = (model.fit if function is None else function)(
        cx, cy, cr, **kwargs
    )
    fit_rng_states[id(model)] = kwargs["generator"].get_state().clone()
    return output


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
    "backend": a.backend,
    "device": "cpu",
    "task": a.task,
    "estimators": a.estimators,
    "query_input": a.query_input,
    "cache_limit": a.cache_limit,
    "prepared_recipe": a.prepared_recipe,
    "atomic_recipe": a.atomic_recipe,
}
try:
    with torch.inference_mode():
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
            "dynamic": True,
        }
        if a.native_sum:
            from native_sum_backend import backend

            kwargs["backend"] = backend
        torch._dynamo.utils.counters.clear()
        if a.entry == "fit":
            fit(model, torch.compile(model.fit, **kwargs))
            actual = predict(model).numerical
            result["fit_generator_state_exact"] = bool(
                torch.equal(
                    fit_rng_states[id(model)], fit_rng_states[id(oracle)]
                )
            )
            result["fitted_buffer_differences"] = []
            for role in ("features", "target", "output"):
                expected_buffers = dict(
                    getattr(
                        oracle._cache["recipe_execution"].recipe, role
                    ).named_buffers()
                )
                for name, value in getattr(
                    model._cache["recipe_execution"].recipe, role
                ).named_buffers():
                    expected_value = expected_buffers[name]
                    if not torch.equal(value, expected_value):
                        result["fitted_buffer_differences"].append(
                            {
                                "role": role,
                                "name": name,
                                "max_abs": float(
                                    (value - expected_value).abs().max()
                                )
                                if value.numel() and value.dtype != torch.bool
                                else None,
                            }
                        )
        elif a.entry == "predict":
            fit(model)
            torch._dynamo.utils.counters.clear()
            compiled_predict = torch.compile(model.predict, **kwargs)
            all_queries = qx
            samples = []
            for rows in a.query_rows or [qx.size(-2)]:
                if a.query_rows is None:
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
                result["samples"] = samples
            result["all_samples_parity"] = all(
                sample["parity"] for sample in samples
            )
        else:
            actual = forward(model, torch.compile(model, **kwargs)).numerical
        result.update(
            compile_stats=dict(torch._dynamo.utils.counters["stats"]),
            graph_breaks={
                k.splitlines()[0]: v
                for k, v in torch._dynamo.utils.counters["graph_break"].items()
            },
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
