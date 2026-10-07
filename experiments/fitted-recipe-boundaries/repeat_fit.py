"""Reuse one compiled fit callable across different real context sizes."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models import KumoTabular

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--prepared-recipe", action="store_true")
parser.add_argument("--extra-repeats", action="store_true")
args = parser.parse_args()
torch.set_num_threads(1)
torch._dynamo.config.cache_size_limit = 64
data = np.load(args.data)
checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
query = TableTensor.from_tensor(
    torch.tensor(data["x"][data["validation_ids"][:31]])
)


def new_model():
    """Load identical pretrained weights from the local checkpoint."""
    model = KumoTabular(
        task="classification", size="small", pretrained=False, device="cpu"
    )
    model.models["classification"].load_state_dict(checkpoint, strict=True)
    return model


model = new_model()
recipe = model.default_recipe() if args.prepared_recipe else None
compiled_fit = torch.compile(model.fit, backend="inductor", dynamic=True)
torch._dynamo.utils.counters.clear()
result = {
    "torch": torch.__version__,
    "backend": "inductor",
    "device": "cpu",
    "dynamic": True,
    "fullgraph": False,
    "cache_limit": 64,
    "prepared_recipe": args.prepared_recipe,
    "same_compiled_callable": True,
    "query_rows": 31,
    "estimators": 1,
    "samples": [],
}
try:
    with torch.inference_mode():
        contexts = [(32, 0), (48, 0), (32, 0)]
        if args.extra_repeats:
            contexts.extend([(32, 32), (32, 0)])
        for rows, context_start in contexts:
            context = TableTensor.from_tensor(
                torch.tensor(
                    data["x"][
                        data["train_ids"][context_start : context_start + rows]
                    ]
                )
            )
            target = TableTensor.from_tensor(
                CategoricalTensor.from_tensor(
                    torch.tensor(
                        data["y"][
                            data["train_ids"][
                                context_start : context_start + rows
                            ]
                        ]
                    )
                    .long()
                    .reshape(-1, 1)
                )
            )
            oracle = new_model()
            reference_generator = torch.Generator().manual_seed(123)
            start = time.perf_counter()
            oracle.fit(
                context,
                target,
                num_estimators=1,
                generator=reference_generator,
                recipe=recipe,
            )
            eager_seconds = time.perf_counter() - start
            expected = oracle.predict(query).numerical.clone()
            generator = torch.Generator().manual_seed(123)
            before = dict(torch._dynamo.utils.counters["stats"])
            start = time.perf_counter()
            compiled_fit(
                context,
                target,
                num_estimators=1,
                generator=generator,
                recipe=recipe,
            )
            fit_seconds = time.perf_counter() - start
            actual = model.predict(query).numerical
            stats = dict(torch._dynamo.utils.counters["stats"])
            sample = {
                "rows": rows,
                "context_start": context_start,
                "eager_fit_seconds": eager_seconds,
                "compiled_fit_seconds": fit_seconds,
                "new_graphs": stats.get("unique_graphs", 0)
                - before.get("unique_graphs", 0),
                "new_captured_calls": stats.get("calls_captured", 0)
                - before.get("calls_captured", 0),
                "max_prediction_error": float((actual - expected).abs().max()),
                "parity": bool(
                    torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
                ),
                "labels_equal": bool(
                    torch.equal(actual.argmax(-1), expected.argmax(-1))
                ),
                "generator_state_exact": bool(
                    torch.equal(
                        generator.get_state(), reference_generator.get_state()
                    )
                ),
                "stats": stats,
            }
            result["samples"].append(sample)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            assert sample["parity"]
            assert sample["generator_state_exact"]
    result["status"] = "pass"
    result["unimplemented"] = dict(
        torch._dynamo.utils.counters["unimplemented"]
    )
    result["graph_breaks"] = dict(torch._dynamo.utils.counters["graph_break"])
except Exception as error:  # noqa: BLE001
    result.update(
        status="fail",
        error_type=type(error).__name__,
        error=str(error),
        traceback=traceback.format_exc(),
    )
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
