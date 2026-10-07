"""Compare identical autocast modes while preserving intermediate cast rounding."""

import argparse
import runpy
import sys

import torch


original_compile = torch.compile


def compile_with_cast_rounding(*args, **kwargs):
    options = dict(kwargs.pop("options", None) or {})
    options["emulate_precision_casts"] = True
    return original_compile(*args, **kwargs, options=options)


torch.compile = compile_with_cast_rounding

parser = argparse.ArgumentParser()
parser.add_argument("--runner", required=True)
args, remaining = parser.parse_known_args()
print("Diagnostic compile option: emulate_precision_casts=True", flush=True)
sys.argv = [args.runner, *remaining]
runpy.run_path(args.runner, run_name="__main__")
