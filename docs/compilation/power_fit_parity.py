"""Separate fitted-parameter drift from compiled transformation arithmetic."""

# ruff: noqa: T201
import copy
import json

import torch

from sdm import TableTensor
from sdm.processing import PowerTransform

for dtype in [torch.float32, torch.float64]:
    x = torch.arange(44, dtype=dtype).reshape(11, 4)
    x[:, 0] = (x[:, 0] - 5).sin() * 4
    x[:, 1] = 3
    x[:, 2] = torch.nan
    x[0, 3] = torch.nan
    table = TableTensor.from_tensor(x)
    reference = PowerTransform()
    with torch.inference_mode():
        expected = reference.fit_transform(table).numerical
        transform_only = copy.deepcopy(reference)
        transformed = torch.compile(transform_only.transform, fullgraph=True)(
            table
        ).numerical
        fitted = PowerTransform()
        fit_output = torch.compile(fitted.fit_transform, fullgraph=True)(
            table
        ).numerical
        state_output = fitted.transform(table).numerical

    def difference(a: torch.Tensor, b: torch.Tensor) -> float:
        """Return the largest finite absolute difference."""
        return (a - b).nan_to_num().abs().max().item()

    print(
        json.dumps(
            {
                "version": torch.__version__,
                "dtype": str(dtype),
                "compiled_transform_max_abs": difference(
                    transformed, expected
                ),
                "compiled_fit_transform_max_abs": difference(
                    fit_output, expected
                ),
                "eager_transform_compiled_stats_max_abs": difference(
                    state_output, expected
                ),
                "lambdas_eager": reference.lambdas.tolist(),
                "lambdas_compiled": fitted.lambdas.tolist(),
                "lambda_max_abs": difference(
                    fitted.lambdas, reference.lambdas
                ),
            }
        ),
        flush=True,
    )
