"""Compare SDM runs with TALENT's published results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TALENT_REVISION = "08301d6"
OFFICIAL_URL = "https://raw.githubusercontent.com/LAMDA-Tabular/TALENT"
TABLES = {
    "binclass": "cls_bin.md",
    "multiclass": "cls_multi.md",
    "regression": "regression.md",
}
METRICS = {
    "binclass": "Accuracy",
    "multiclass": "Accuracy",
    "regression": "RMSE",
}
EXAMPLE_DIR = Path(__file__).parent.parent


def _official(root: Path | None, revision: str) -> pd.DataFrame:
    frames = []
    for task, filename in TABLES.items():
        source = (
            root / filename
            if root
            else f"{OFFICIAL_URL}/{revision}/results/{filename}"
        )
        wide = (
            pd.read_csv(source, sep=r"\s*\|\s*", engine="python", skiprows=[1])
            .dropna(axis="columns", how="all")
            .rename(columns={"Dataset": "dataset"})
        )
        long = wide.melt("dataset", var_name="method", value_name="score")
        values = (
            long.pop("score")
            .astype(str)
            .str.replace("*", "", regex=False)
            .str.split("+", n=1, expand=True)
        )
        long[["mean", "std"]] = values.apply(pd.to_numeric, errors="coerce")
        long.insert(0, "task", task)
        long["source"] = "official"
        frames.append(long.dropna(subset=["mean"]))
    return pd.concat(frames, ignore_index=True)


def _sdm(
    root: Path,
    official: pd.DataFrame,
    models: list[str] | None,
) -> pd.DataFrame:
    task_by_dataset = official.drop_duplicates("dataset").set_index("dataset")[
        "task"
    ]
    rows = []
    for path in sorted(root.glob("*/*/result.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "success" or (
            models and record["model"] not in models
        ):
            continue
        dataset, model = record["dataset"], record["model"]
        if dataset not in task_by_dataset:
            raise ValueError(
                f"{dataset!r} from {path} is not in TALENT's tables"
            )
        task = task_by_dataset[dataset]
        metric, result = METRICS[task], record["result"]
        rows.append(
            {
                "task": task,
                "dataset": dataset,
                "method": model,
                "mean": round(result["metrics_mean"][metric], 4),
                "std": round(result["metrics_std"][metric], 4),
                "seed_num": record.get("seed_num"),
                "source": "sdm",
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No successful SDM results found under {root}"
        )
    return pd.DataFrame(rows)


def _leaderboard(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, blocks = [], {}
    for task in TABLES:
        task_results = results[results["task"] == task]
        scores = task_results.pivot(
            index="dataset", columns="method", values="mean"
        )
        common = scores.dropna()
        blocks[task] = common
        ranks = common.rank(
            axis="columns", ascending=task == "regression"
        ).mean()
        available = task_results.groupby("method")["dataset"].nunique()
        total = task_results.loc[
            task_results["source"] == "official", "dataset"
        ].nunique()
        table = (
            ranks.rename("average_rank").rename_axis("method").reset_index()
        )
        table["task"] = task
        table["compared_datasets"] = len(common)
        table["available_datasets"] = table["method"].map(available)
        table["official_datasets"] = total
        table["coverage"] = table["available_datasets"] / total
        rows.append(table)
    columns = [
        "task",
        "method",
        "average_rank",
        "compared_datasets",
        "available_datasets",
        "official_datasets",
        "coverage",
    ]
    return pd.concat(rows, ignore_index=True)[columns], blocks


def _plot(output: Path, blocks: dict[str, pd.DataFrame]) -> None:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        from matplotlib import pyplot  # noqa: PLC0415
        from scikit_posthocs import (  # noqa: PLC0415
            critical_difference_diagram,
            posthoc_conover_friedman,
        )
    except ImportError as error:
        raise ImportError(
            "--plot-cd requires matplotlib and scikit-posthocs"
        ) from error

    for task, scores in blocks.items():
        if len(scores) < 2:
            continue
        ranks = (
            scores.rank(axis="columns", ascending=task == "regression")
            .mean()
            .sort_values()
        )
        figure, axis = pyplot.subplots(figsize=(12, max(6, len(ranks) * 0.2)))
        critical_difference_diagram(
            ranks, posthoc_conover_friedman(scores), ax=axis
        )
        axis.set_title(f"TALENT {task} ({len(scores)} common datasets)")
        figure.savefig(
            output / f"critical_difference_{task}.png", bbox_inches="tight"
        )
        pyplot.close(figure)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--result-dir", type=Path, default=EXAMPLE_DIR / "talent_out"
)
parser.add_argument(
    "--output-dir", type=Path, default=EXAMPLE_DIR / "evals" / "talent"
)
parser.add_argument("--official-results", type=Path)
parser.add_argument("--official-revision", default=TALENT_REVISION)
parser.add_argument("--model", action="append")
parser.add_argument("--plot-cd", action="store_true")
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
official = _official(args.official_results, args.official_revision)
sdm_results = _sdm(args.result_dir, official, args.model)
results = pd.concat([official, sdm_results], ignore_index=True)
if results.duplicated(["task", "dataset", "method"]).any():
    raise ValueError("Duplicate task/dataset/method results")
leaderboard, blocks = _leaderboard(results)
results.to_csv(args.output_dir / "results.csv", index=False)
leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
if args.plot_cd:
    _plot(args.output_dir, blocks)
ours = leaderboard[leaderboard["method"].isin(sdm_results["method"])]
print(ours.to_string(index=False))
