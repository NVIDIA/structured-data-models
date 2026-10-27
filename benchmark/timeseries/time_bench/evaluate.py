# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate SDM results with TIME's official leaderboard script."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from benchmark.forecasting.model import MODEL_CONFIGS

# Pinned revision of the published results. The official script adds the
# seasonal naive baseline itself:
REVISION = "1ebbe655f7fad151f5447ef94b2c371d71ba6d7e"
REFERENCES = ("TimesFM-3", "TimesFM-2.5", "Chronos-2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-path",
        type=Path,
        required=True,
        help="Path to the pinned TIME checkout.",
    )
    args = parser.parse_args()

    script = args.time_path / "scripts" / "compute_local_leaderboard.py"
    if not script.is_file():
        raise FileNotFoundError(f"TIME checkout not found at {args.time_path}")

    result_root = Path(__file__).parent.parent / "time_bench_out"
    runs = {}
    for model_config in MODEL_CONFIGS.values():
        if (result_root / model_config.name).is_dir():
            runs[f"[SDM] {model_config.name}"] = (
                result_root / model_config.name
            )

    if not runs:
        raise FileNotFoundError(f"No TIME results found under {result_root}")

    # Compare with published results on configurations run by all models:
    configs = set.intersection(
        *(
            {
                path.parent.relative_to(run)
                for path in run.glob("**/metrics.npz")
            }
            for run in runs.values()
        )
    )
    reference_root = snapshot_download(
        repo_id="Real-TSF/TIME-Output",
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[
            f"results/{name}/**/{file}"
            for name in REFERENCES
            for file in ("metrics.npz", "config.json")
        ],
    )
    for name in REFERENCES:
        runs[name] = Path(reference_root) / "results" / name

    # The official script reads all models under `output/results/`:
    with tempfile.TemporaryDirectory(prefix="sdm-time-") as directory:
        results = Path(directory) / "output" / "results"
        for name, run in runs.items():
            for config in configs:
                if (run / config / "metrics.npz").is_file():
                    (results / name / config).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    (results / name / config).symlink_to(run / config)
        subprocess.run(
            [sys.executable, str(script.resolve())],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
