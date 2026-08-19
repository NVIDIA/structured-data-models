# ruff: noqa
# ty: ignore
"""Pseudo-label feedback experiment for TabICLv2.

Tests whether feeding back the model's own predictions as additional
pseudo-labeled context rows improves predictions on subsequent rounds.
No hooks or KV manipulation — feedback enters through the standard
x_context / y_context interface.
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
from torch import Tensor
from torch.nn.functional import cosine_similarity

import sdm

# (name, data_id, task_type)
DATASETS: list[tuple[str, int, str]] = [
    # Classification
    ("cmc", 23, "classification"),
    ("yeast", 181, "classification"),
    ("wine-quality-white", 40691, "classification"),
    ("credit-g", 31, "classification"),
    # Regression
    ("pol", 546, "regression"),
    ("superconductor", 42570, "regression"),
    ("house_prices", 41540, "regression"),
]

NUM_ROUNDS = 4
NUM_SEEDS = 3
SEEDS = [42, 123, 456]


def load_dataset(data_id, task, device, seed=42):
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = bunch.frame
    target = bunch.target_names
    if isinstance(target, list):
        target = target[0]

    df = df.dropna(subset=[target])

    train_df, test_df = train_test_split(df, train_size=0.7, random_state=seed)

    target_stype = "categorical" if task == "classification" else "numerical"
    stypes = sdm.infer_stypes(df, overrides={target: target_stype})

    context = sdm.TableTensor.from_pandas(
        df=train_df, stypes=stypes, device=device
    )
    query = sdm.TableTensor.from_pandas(
        df=test_df, stypes=stypes, device=device
    )

    return context, query, target, test_df, stypes


def get_icl_block(model, task):
    if task == "classification":
        return model.cls_model.icl_block
    return model.reg_model.icl_block


def compute_metrics(pred, y_true_df, target, task):
    metrics = {}

    if task == "classification":
        proba = pred.to_pandas().values
        y_true = y_true_df[target].astype(str).values
        y_pred = pred.to_pandas().idxmax(axis=1).values
        classes = pred.to_pandas().columns.tolist()

        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        try:
            metrics["log_loss"] = log_loss(y_true, proba, labels=classes)
        except ValueError:
            metrics["log_loss"] = float("nan")

        entropy = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1)
        metrics["entropy"] = float(entropy.mean())
    else:
        y_pred = pred.numerical.float().mean(dim=-1).cpu().numpy()
        y_true = y_true_df[target].astype(float).values

        metrics["rmse"] = float(((y_pred - y_true) ** 2).mean() ** 0.5)
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["entropy"] = float("nan")

    return metrics


def compute_embedding_drift(prev, curr):
    prev_flat = prev.reshape(-1, prev.size(-1))
    curr_flat = curr.reshape(-1, curr.size(-1))
    cos_sim = cosine_similarity(prev_flat, curr_flat, dim=-1)
    return float(1.0 - cos_sim.mean())


def build_pseudo_context(query_x, pred, task, target, stypes, device):
    """Build pseudo-labeled context rows from query features + predicted labels."""
    if task == "classification":
        pred_labels = pred.to_pandas().idxmax(axis=1)
        pseudo_df = query_x.to_pandas().copy()
        pseudo_df[target] = pred_labels.values
    else:
        pred_values = pred.numerical.float().mean(dim=-1).cpu().numpy()
        pseudo_df = query_x.to_pandas().copy()
        pseudo_df[target] = pred_values

    return sdm.TableTensor.from_pandas(
        df=pseudo_df, stypes=stypes, device=device
    )


def run_experiment(data_id, task, device, seed=42, num_rounds=NUM_ROUNDS):
    context, query, target, test_df, stypes = load_dataset(
        data_id, task, device, seed
    )
    model = sdm.models.TabICLv2(device=device)
    icl_block = get_icl_block(model, task)

    round_results = []
    prev_embeddings = None
    augmented_context = context

    for r in range(num_rounds + 1):
        captured_pre_head = []

        def capture_embedding(module, args):
            captured_pre_head.append(args[0].detach().clone())

        emb_handle = icl_block.head.register_forward_pre_hook(
            capture_embedding
        )

        # Fix seed before each forward pass for determinism.
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        with torch.amp.autocast(
            device.type, torch.bfloat16, enabled=context.is_cuda
        ):
            pred = model(
                x_context=augmented_context.drop_columns(target),
                y_context=augmented_context[:, target],
                x_query=query.drop_columns(target),
                num_estimators=1,
            )

        emb_handle.remove()

        metrics = compute_metrics(pred, test_df, target, task)

        embeddings = captured_pre_head[0] if captured_pre_head else None
        drift = float("nan")
        if prev_embeddings is not None and embeddings is not None:
            drift = compute_embedding_drift(prev_embeddings, embeddings)

        result = {
            "round": r,
            "seed": seed,
            **metrics,
            "embedding_drift": drift,
        }
        round_results.append(result)

        # Build pseudo-labeled context for next round.
        if r < num_rounds:
            with torch.no_grad():
                query_x_table = query.drop_columns(target)
                pseudo_context = build_pseudo_context(
                    query_x_table,
                    pred,
                    task,
                    target,
                    stypes,
                    device,
                )
            # Concatenate original context with pseudo-labeled rows.
            augmented_context = context.cat([context, pseudo_context])

        prev_embeddings = embeddings

    return round_results


def print_round_table(results, task):
    if task == "classification":
        print(
            f"{'round':>5}  {'accuracy':>8}  {'log_loss':>8}  {'entropy':>8}  {'emb_drift':>9}"
        )
        for r in results:
            drift = (
                f"{r['embedding_drift']:>9.4f}"
                if not np.isnan(r["embedding_drift"])
                else "        -"
            )
            print(
                f"{r['round']:>5}  {r['accuracy']:>8.3f}  {r['log_loss']:>8.3f}  {r['entropy']:>8.3f}  {drift}"
            )
    else:
        print(f"{'round':>5}  {'rmse':>10}  {'r2':>8}  {'emb_drift':>9}")
        for r in results:
            drift = (
                f"{r['embedding_drift']:>9.4f}"
                if not np.isnan(r["embedding_drift"])
                else "        -"
            )
            print(
                f"{r['round']:>5}  {r['rmse']:>10.3f}  {r['r2']:>8.3f}  {drift}"
            )


def print_seed_summary(all_seed_results, task):
    """Print mean +/- std across seeds for each round."""
    rounds = sorted(set(r["round"] for r in all_seed_results))

    if task == "classification":
        print(
            f"{'round':>5}  {'accuracy':>14}  {'log_loss':>14}  {'entropy':>14}"
        )
        for rd in rounds:
            rd_results = [r for r in all_seed_results if r["round"] == rd]
            accs = [r["accuracy"] for r in rd_results]
            lls = [r["log_loss"] for r in rd_results]
            ents = [r["entropy"] for r in rd_results]
            print(
                f"{rd:>5}  "
                f"{np.mean(accs):>6.3f} ± {np.std(accs):.3f}  "
                f"{np.mean(lls):>6.3f} ± {np.std(lls):.3f}  "
                f"{np.mean(ents):>6.3f} ± {np.std(ents):.3f}"
            )
    else:
        print(f"{'round':>5}  {'rmse':>16}  {'r2':>14}")
        for rd in rounds:
            rd_results = [r for r in all_seed_results if r["round"] == rd]
            rmses = [r["rmse"] for r in rd_results]
            r2s = [r["r2"] for r in rd_results]
            print(
                f"{rd:>5}  "
                f"{np.mean(rmses):>8.3f} ± {np.std(rmses):.3f}  "
                f"{np.mean(r2s):>6.3f} ± {np.std(r2s):.3f}"
            )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")

    all_results = []

    for name, data_id, task in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"=== {name} ({task}) ===")

        dataset_results = []

        for seed in SEEDS:
            print(f"\n  seed={seed}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = run_experiment(data_id, task, device, seed)
                for r in results:
                    r["name"] = name
                    r["data_id"] = data_id
                    r["task"] = task
                dataset_results.extend(results)
                all_results.extend(results)
                print_round_table(results, task)
            except Exception:
                print("  FAILED:")
                traceback.print_exc(file=sys.stdout)

        if dataset_results:
            print(f"\n  Mean ± std across seeds:")
            print_seed_summary(dataset_results, task)

    # Print summary table.
    print(f"\n{'=' * 70}")
    print("SUMMARY: Mean accuracy/R² per round (averaged across seeds)")
    print()

    for name, _, task in DATASETS:
        ds_results = [r for r in all_results if r["name"] == name]
        if not ds_results:
            continue

        rounds = sorted(set(r["round"] for r in ds_results))
        metric_key = "accuracy" if task == "classification" else "r2"

        round_strs = []
        for rd in rounds:
            vals = [r[metric_key] for r in ds_results if r["round"] == rd]
            round_strs.append(f"R{rd}: {np.mean(vals):.3f}±{np.std(vals):.3f}")

        base_vals = [r[metric_key] for r in ds_results if r["round"] == 0]
        final_vals = [
            r[metric_key] for r in ds_results if r["round"] == NUM_ROUNDS
        ]
        if base_vals and final_vals:
            gain = np.mean(final_vals) - np.mean(base_vals)
            gain_str = f"  gain: {gain:+.3f}"
        else:
            gain_str = ""

        print(f"  {name:<22} {' | '.join(round_strs)}{gain_str}")

    # Save to CSV.
    csv_path = "pseudo_label_results.csv"
    all_keys = [
        "name",
        "data_id",
        "task",
        "seed",
        "round",
        "accuracy",
        "log_loss",
        "entropy",
        "rmse",
        "r2",
        "embedding_drift",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
