# ruff: noqa
"""Sweep TabICLv2 across OpenML datasets.

Runs vanilla TabICLv2 on a curated set of classification and regression
datasets, collects performance metrics, and writes results to CSV.
"""

import csv
import sys
import traceback
import warnings

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss, r2_score
from sklearn.model_selection import train_test_split

import sdm

# (data_id, task_type)
# Classification: span easy binary through hard multi-class.
# Regression: span easy through hard.
DATASETS: list[tuple[int, str]] = [
    # --- Classification ---
    # Binary
    (31, "classification"),  # credit-g
    (37, "classification"),  # diabetes
    (1462, "classification"),  # banknote-authentication
    (1510, "classification"),  # wdbc
    (1489, "classification"),  # phoneme
    (1461, "classification"),  # bank-marketing
    # Multi-class
    (23, "classification"),  # cmc
    (188, "classification"),  # eucalyptus
    (12, "classification"),  # mfeat-factors
    (14, "classification"),  # mfeat-fourier
    (40691, "classification"),  # wine-quality-white
    (181, "classification"),  # yeast
    (1466, "classification"),  # cardiotocography
    (40975, "classification"),  # car
    (40496, "classification"),  # LED-display
    (1476, "classification"),  # gas-drift
    (40668, "classification"),  # connect-4
    (6, "classification"),  # letter
    (554, "classification"),  # mnist_784
    (40685, "classification"),  # shuttle
    # --- Regression ---
    (531, "regression"),  # boston
    (507, "regression"),  # space_ga
    (422, "regression"),  # california housing
    (546, "regression"),  # pol
    (41021, "regression"),  # Moneyball
    (41540, "regression"),  # house_prices_nominal
    (42225, "regression"),  # flood
    (42570, "regression"),  # superconductor
    (42571, "regression"),  # concrete
    (41980, "regression"),  # particulate-matter-ukair
]


def run_dataset(
    data_id: int,
    task: str,
    device: torch.device,
    context_fraction: float = 0.7,
    seed: int = 42,
) -> dict[str, object]:
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = bunch.frame
    target = bunch.target_names
    if isinstance(target, list):
        target = target[0]

    df = df.dropna(subset=[target])

    # Cap dataset size to keep sweep manageable.
    max_rows = 10_000
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)

    train_df, test_df = train_test_split(
        df,
        train_size=context_fraction,
        random_state=seed,
    )

    target_stype = "categorical" if task == "classification" else "numerical"
    stypes = sdm.infer_stypes(df, overrides={target: target_stype})

    context = sdm.TableTensor.from_pandas(
        df=train_df,
        stypes=stypes,
        device=device,
    )
    query = sdm.TableTensor.from_pandas(
        df=test_df,
        stypes=stypes,
        device=device,
    )

    model = sdm.models.TabICLv2(device=device)

    with torch.amp.autocast(
        device.type, torch.bfloat16, enabled=context.is_cuda
    ):
        pred = model(
            x_context=context.drop_columns(target),
            y_context=context[:, target],
            x_query=query.drop_columns(target),
            num_estimators=1,
        )

    result: dict[str, object] = {
        "data_id": data_id,
        "task": task,
        "n_rows": len(df),
        "n_features": len(df.columns) - 1,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    if task == "classification":
        proba = pred.to_pandas().values
        y_true = test_df[target].astype(str).values
        y_pred = pred.to_pandas().idxmax(axis=1).values

        classes = pred.to_pandas().columns.tolist()
        n_classes = len(classes)
        result["n_classes"] = n_classes
        result["accuracy"] = accuracy_score(y_true, y_pred)

        try:
            result["log_loss"] = log_loss(
                y_true,
                proba,
                labels=classes,
            )
        except ValueError:
            result["log_loss"] = float("nan")

        entropy = -np.sum(
            proba * np.log(np.clip(proba, 1e-12, None)),
            axis=1,
        )
        result["mean_entropy"] = float(entropy.mean())
    else:
        y_pred = pred.numerical.float().mean(dim=-1).cpu().numpy()
        y_true = query[:, target].numerical.float().squeeze(-1).cpu().numpy()

        result["n_classes"] = None
        result["rmse"] = float(((y_pred - y_true) ** 2).mean() ** 0.5)
        result["r2"] = r2_score(y_true, y_pred)
        result["mean_entropy"] = None

    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: list[dict[str, object]] = []

    for data_id, task in DATASETS:
        name = f"openml-{data_id}"
        print(f"\n{'=' * 60}")
        print(f"Running {name} ({task})...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_dataset(data_id, task, device)
            result["name"] = name
            result["status"] = "ok"
            results.append(result)

            if task == "classification":
                print(
                    f"  acc={result['accuracy']:.3f}  "
                    f"log_loss={result['log_loss']:.3f}  "
                    f"entropy={result['mean_entropy']:.3f}  "
                    f"classes={result['n_classes']}"
                )
            else:
                print(f"  rmse={result['rmse']:.3f}  r2={result['r2']:.3f}")
        except Exception:
            print("  FAILED:")
            traceback.print_exc(file=sys.stdout)
            results.append(
                {
                    "name": name,
                    "data_id": data_id,
                    "task": task,
                    "status": "failed",
                }
            )

    # Print summary table.
    print(f"\n{'=' * 60}")
    print("CLASSIFICATION RESULTS (sorted by accuracy):")
    print(
        f"{'name':<25} {'acc':>6} {'log_loss':>9} {'entropy':>8} {'cls':>4} {'rows':>6}"
    )
    cls_results = sorted(
        [
            r
            for r in results
            if r.get("task") == "classification" and r.get("status") == "ok"
        ],
        key=lambda r: r["accuracy"],
    )
    for r in cls_results:
        print(
            f"{r['name']:<25} {r['accuracy']:>6.3f} {r['log_loss']:>9.3f} "
            f"{r['mean_entropy']:>8.3f} {r['n_classes']:>4} {r['n_rows']:>6}"
        )

    print("\nREGRESSION RESULTS (sorted by R²):")
    print(f"{'name':<25} {'rmse':>10} {'r2':>8} {'rows':>6}")
    reg_results = sorted(
        [
            r
            for r in results
            if r.get("task") == "regression" and r.get("status") == "ok"
        ],
        key=lambda r: r["r2"],
    )
    for r in reg_results:
        print(
            f"{r['name']:<25} {r['rmse']:>10.3f} {r['r2']:>8.3f} {r['n_rows']:>6}"
        )

    # Save to CSV.
    csv_path = "dataset_sweep_results.csv"
    all_keys = [
        "name",
        "data_id",
        "task",
        "status",
        "n_rows",
        "n_features",
        "n_train",
        "n_test",
        "n_classes",
        "accuracy",
        "log_loss",
        "mean_entropy",
        "rmse",
        "r2",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
