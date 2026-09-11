"""Evaluate SDM results with ScoringBench's official ranking script."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).parent.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scoringbench-path",
        type=Path,
        required=True,
        help="Path to the pinned ScoringBench checkout.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=BENCHMARK_DIR / "scoringbench_out" / "univariate",
    )
    parser.add_argument(
        "--official-results",
        type=Path,
        help=(
            "Official aggregated Parquet directory; defaults to the "
            "checkout's output submodule."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIR / "evals" / "scoringbench",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main() -> None:
    args = _parser().parse_args()
    source = args.scoringbench_path.resolve()
    aggregate_script = source / "aggregate_datasets.py"
    leaderboard_script = source / "autorank_leaderboard.py"
    if not aggregate_script.is_file() or not leaderboard_script.is_file():
        raise FileNotFoundError(f"ScoringBench checkout not found at {source}")

    raw_dir = args.result_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"No ScoringBench results found under {raw_dir}"
        )
    subprocess.run(
        [
            sys.executable,
            str(aggregate_script),
            "--raw_dir",
            str(raw_dir),
            "--out_dir",
            str(args.result_dir),
        ],
        check=True,
    )

    official = (
        args.official_results
        if args.official_results is not None
        else source / "output" / "output_univariate_scoringbench_d1_n3000"
    )
    official_files = sorted(official.glob("*.parquet"))
    sdm_files = sorted(args.result_dir.glob("*.parquet"))
    if not official_files:
        raise FileNotFoundError(
            f"No official aggregated results found under {official}"
        )
    if not sdm_files:
        raise FileNotFoundError(
            f"No aggregated SDM results found under {args.result_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(path) for path in [*official_files, *sdm_files]]
    pd.concat(frames, ignore_index=True).to_csv(
        args.output_dir / "results.csv",
        index=False,
    )

    with tempfile.TemporaryDirectory(prefix="sdm-scoringbench-") as temporary:
        comparison = Path(temporary)
        for path in official_files:
            shutil.copy2(path, comparison / path.name)
        for path in sdm_files:
            shutil.copy2(path, comparison / path.name)
        subprocess.run(
            [
                sys.executable,
                str(leaderboard_script),
                "--output_dir",
                str(comparison),
                "--alpha",
                str(args.alpha),
            ],
            check=True,
        )
        generated = comparison / "figures" / "leaderboard"
        if generated.is_dir():
            shutil.copytree(
                generated,
                args.output_dir / "leaderboard",
                dirs_exist_ok=True,
            )


if __name__ == "__main__":
    main()
