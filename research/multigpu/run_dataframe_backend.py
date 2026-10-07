# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a benchmark with explicit cuDF/Arrow selection in one environment."""

import argparse
import importlib.metadata
import importlib.util
import json
import runpy
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path


def main() -> None:
    """Select backend discovery and save its observed call sites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["arrow", "cudf"], required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("runner", type=Path)
    args, runner_args = parser.parse_known_args()
    original = importlib.util.find_spec
    calls: dict[str, int] = {}
    cudf_present = original("cudf") is not None
    if not cudf_present:
        raise RuntimeError("Paired backend runs require cuDF to be installed")

    def find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "cudf":
            caller = sys._getframe(1).f_globals.get("__name__", "unknown")
            calls[caller] = calls.get(caller, 0) + 1
            if args.backend == "arrow":
                return None
        return original(name, package)

    importlib.util.find_spec = find_spec
    sys.argv = [str(args.runner), *runner_args]
    try:
        runpy.run_path(str(args.runner), run_name="__main__")
    finally:
        importlib.util.find_spec = original
        args.receipt.write_text(
            json.dumps(
                {
                    "backend": args.backend,
                    "harness_override": "Intercept only find_spec('cudf')",
                    "cudf_installed": cudf_present,
                    "backend_checks_by_module": calls,
                    "runner": str(args.runner),
                    "runner_args": runner_args,
                    "versions": {
                        name: importlib.metadata.version(name)
                        for name in [
                            "torch",
                            "cudf-cu13",
                            "pandas",
                            "numpy",
                            "pyarrow",
                        ]
                    },
                    "scope": (
                        "Single process or thread-based ensemble; "
                        "not inherited by spawned processes"
                    ),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
