"""Check fitting helper contracts against native operations under autocast."""

import argparse
import json

import torch

from sdm.processing.numerical._stats import (
    _fitting_nanmean,
    _fitting_nansum,
)
from sdm.processing.numerical.power import _fitting_expm1

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
args = parser.parse_args()

operations = (
    ("nansum", _fitting_nansum, lambda x: x.nansum(-2, keepdim=True)),
    ("nanmean", _fitting_nanmean, lambda x: x.nanmean(-2, keepdim=True)),
    ("expm1", _fitting_expm1, lambda x: x.clone().expm1_()),
)
for autocast_dtype in (torch.bfloat16, torch.float16):
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.linspace(-2, 2, 35, device=args.device).reshape(7, 5)
        x = x.to(dtype)
        x[1, 2] = float("nan")
        x[:, 4] = float("nan")
        for name, operation, baseline in operations:
            with torch.autocast(args.device, dtype=autocast_dtype):
                expected = baseline(x)
                actual = operation(x)
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0, equal_nan=True
                )
                torch.library.opcheck(
                    operation,
                    (x,),
                    test_utils=("test_schema", "test_faketensor"),
                )
                for fullgraph in (False, True):
                    torch._dynamo.reset()
                    compiled = torch.compile(operation, fullgraph=fullgraph)
                    torch.testing.assert_close(
                        compiled(x), expected, rtol=0, atol=0, equal_nan=True
                    )
            print(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "device": args.device,
                        "autocast": str(autocast_dtype),
                        "input": str(dtype),
                        "operation": name,
                        "output": str(expected.dtype),
                        "eager_fake_inductor_bothflags": "PASS",
                    }
                ),
                flush=True,
            )
