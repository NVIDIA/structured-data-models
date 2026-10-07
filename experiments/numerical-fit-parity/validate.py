"""Compare actual PowerTransform fitting against the same-dtype eager path."""

# ruff: noqa: T201
import argparse
import json
import statistics
import time

import torch

from sdm import TableTensor
from sdm.processing import Clip, ImputeMean, PowerTransform, Standardize

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=("synthetic", "breast-cancer"), default="synthetic"
)
parser.add_argument(
    "--dtype", choices=("float32", "float64"), default="float32"
)
parser.add_argument("--rows", nargs="+", type=int)
parser.add_argument(
    "--data-file", help="Optional cached (features, targets) tensor tuple."
)
parser.add_argument("--fullgraph", action="store_true")
parser.add_argument("--benchmark", action="store_true")
args = parser.parse_args()
dtype = getattr(torch, args.dtype)
rows_to_test = args.rows or (
    [11, 7, 21] if args.dataset == "synthetic" else [160, 79]
)
features = None
if args.dataset == "breast-cancer":
    if args.data_file:
        features, _ = torch.load(args.data_file, weights_only=True)
        features = features.to(dtype=dtype)
    else:
        from sklearn.datasets import load_breast_cancer

        features = torch.as_tensor(load_breast_cancer().data, dtype=dtype)

reference = PowerTransform()
candidate = PowerTransform()
# The private computation excludes the separate 2.7 schema-routing blocker.
compiled = torch.compile(
    candidate._fit_transform, fullgraph=args.fullgraph, dynamic=True
)
with torch.inference_mode():
    for rows in rows_to_test:
        if features is None:
            x = torch.arange(rows * 5, dtype=dtype).reshape(rows, 5)
            x[:, 0] = (x[:, 0] - 5).sin() * 4
            x[:, 1] = 3
            x[:, 2] = torch.nan
            x[::3, 3] = torch.nan
            x[0, 4] = torch.inf
            table = TableTensor.from_tensor(x)
        else:
            x = features[:rows].clone()
            x[::7, ::3] = torch.nan
            x = torch.cat(
                [x, x.new_ones(rows, 1), x.new_full((rows, 1), torch.nan)],
                dim=1,
            )
            table = TableTensor.from_tensor(x)
            for processor in [
                ImputeMean(),
                Standardize(eps=1e-6),
                Clip(-100.0, 100.0),
            ]:
                table = processor.fit_transform(table)
        expected = reference._fit_transform(table).numerical
        actual = compiled(table).numerical
        torch.testing.assert_close(actual, expected, equal_nan=True)
        for name in ("lambdas", "mean", "scale"):
            torch.testing.assert_close(
                getattr(candidate, name),
                getattr(reference, name),
                atol=0,
                rtol=0,
            )
        result = {
            "version": torch.__version__,
            "dataset": args.dataset,
            "rows": rows,
            "dtype": args.dtype,
            "fullgraph": args.fullgraph,
            "max_abs_difference": (actual - expected)
            .nan_to_num()
            .abs()
            .max()
            .item(),
            "fitted_parameters_exact": True,
        }
        if args.benchmark:
            for _ in range(3):
                reference._fit_transform(table)
                compiled(table)
            times: dict[str, list[float]] = {"eager": [], "compiled": []}
            for repeat in range(30):
                calls = [
                    ("eager", reference._fit_transform),
                    ("compiled", compiled),
                ]
                if repeat % 2:
                    calls.reverse()
                for name, function in calls:
                    start = time.perf_counter()
                    function(table)
                    times[name].append((time.perf_counter() - start) * 1000)
            result["median_ms"] = {
                name: statistics.median(values)
                for name, values in times.items()
            }
        print(json.dumps(result), flush=True)
