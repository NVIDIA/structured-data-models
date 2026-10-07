"""Exercise public numerical processors through real Inductor compilation."""

# ruff: noqa: T201, BLE001
import argparse
import copy
import json
import time

import torch

import sdm.processing as sp
from sdm import TableTensor

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="inductor")
parser.add_argument("--dynamic", action="store_true")
parser.add_argument(
    "--processors",
    nargs="+",
    default=[
        "Standardize",
        "ImputeMean",
        "PowerTransform",
        "RobustScale",
        "RankGaussian",
        "ClipSigma",
        "DropConstantColumns",
    ],
)
args = parser.parse_args()


def table(rows: int, dtype: torch.dtype) -> TableTensor:
    """Construct finite, constant and missing-value columns."""
    x = torch.arange(rows * 4, dtype=dtype).reshape(rows, 4)
    x[:, 0] = (x[:, 0] - 5).sin() * 4
    x[:, 1] = 3
    x[:, 2] = torch.nan
    x[0, 3] = torch.nan
    return TableTensor.from_tensor(x)


for name in args.processors:
    for fullgraph in [False, True]:
        for operation in ["fit_transform", "transform"]:
            for dtype in [torch.float32, torch.float64]:
                torch._dynamo.reset()
                processor = getattr(sp, name)()
                reference = copy.deepcopy(processor)
                training = table(11, dtype)
                if operation == "transform":
                    processor.fit(training)
                    reference.fit(training)
                fn = torch.compile(
                    getattr(processor, operation),
                    backend=args.backend,
                    fullgraph=fullgraph,
                    dynamic=args.dynamic,
                )
                started = time.monotonic()
                result = {
                    "version": torch.__version__,
                    "processor": name,
                    "fullgraph": fullgraph,
                    "operation": operation,
                    "dtype": str(dtype),
                    "backend": args.backend,
                    "dynamic": args.dynamic,
                }
                try:
                    with torch.inference_mode():
                        for rows in [11, 7, 15]:
                            inputs = table(rows, dtype)
                            expected = getattr(reference, operation)(inputs)
                            out = fn(inputs)
                            torch.testing.assert_close(
                                out.numerical,
                                expected.numerical,
                                equal_nan=True,
                            )
                            assert out.columns == expected.columns
                            assert out.active_stypes == expected.active_stypes
                            assert processor.is_fitted == reference.is_fitted
                        result["result"] = "pass"
                except Exception as error:
                    result["result"] = "fail"
                    result["error"] = str(error)
                    result["error_type"] = type(error).__name__
                result["seconds"] = round(time.monotonic() - started, 3)
                print(json.dumps(result), flush=True)
