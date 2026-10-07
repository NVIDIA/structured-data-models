"""Run the shared GPU probe with the diagnostic native LayerNorm boundary."""

import argparse
import runpy
import sys

import native_layer_norm_patch  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--runner", required=True)
args, remaining = parser.parse_known_args()
sys.argv = [args.runner, *remaining]
runpy.run_path(args.runner, run_name="__main__")
