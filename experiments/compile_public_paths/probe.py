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
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(123)
if a.model == "tabular":
    data = np.load(a.data)
    cx = TableTensor.from_tensor(
        torch.tensor(data["x"][data["train_ids"][:32]])
    )
    cy = TableTensor.from_tensor(
        CategoricalTensor.from_tensor(
            torch.tensor(data["y"][data["train_ids"][:32]]).long().view(-1, 1)
        )
    )
    qx = TableTensor.from_tensor(
        torch.tensor(data["x"][data["validation_ids"][:4]])
    )
    cr = qr = None
else:
    b = torch.load(a.data, weights_only=False)
    arm = b["arms"][0]
    cx, cy, cr = arm["context"], arm["context_target"], arm["related_context"]
    qx, qr = arm["queries"][1]["x"], arm["queries"][1]["related"]
ckpt = torch.load(a.checkpoint, weights_only=True, map_location="cpu")


def new_model():
    model = (
        KumoTabular(
            task="classification", size="small", pretrained=False, device="cpu"
        )
        if a.model == "tabular"
        else KumoRelational(
            task="classification", pretrained=False, device="cpu"
        )
    )
    model.models["classification"].load_state_dict(ckpt, strict=True)
    return model


def fit(model, function=None):
    kwargs = {
        "generator": torch.Generator().manual_seed(123),
        "num_estimators": 1,
    }
    if a.model == "relational":
        kwargs["num_hops"] = arm["num_hops"]
    return (model.fit if function is None else function)(cx, cy, cr, **kwargs)


def predict(model, function=None):
    return (model.predict if function is None else function)(qx, qr)


def forward(model, function=None):
    kwargs = {
        "generator": torch.Generator().manual_seed(123),
        "num_estimators": 1,
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
        if a.entry == "fit":
            fit(model, torch.compile(model.fit, **kwargs))
            actual = predict(model).numerical
        elif a.entry == "predict":
            fit(model)
            actual = predict(
                model, torch.compile(model.predict, **kwargs)
            ).numerical
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
Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in {"traceback", "error"}}
    ),
    flush=True,
)
if result["status"] == "fail":
    print(result["error"], flush=True)
