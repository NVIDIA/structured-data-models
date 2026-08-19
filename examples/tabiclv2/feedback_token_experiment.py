# ruff: noqa
# ty: ignore
"""Feedback token experiment for TabICLv2.

Tests whether feeding back the model's own hidden states as additional
key/value tokens in the ICL attention layers improves predictions.
Compares three compression strategies across multiple rounds.
"""

import csv
import sys
import traceback
import warnings
from typing import Literal

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

COMPRESSION_METHODS = ["mean_pool", "top_k", "no_compress"]
NUM_ROUNDS = 4
TOP_K = 10


def load_dataset(
    data_id: int, task: str, device: torch.device, seed: int = 42
):
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

    return context, query, target, test_df


def get_icl_block(model: sdm.models.TabICLv2, task: str):
    if task == "classification":
        return model.cls_model.icl_block
    return model.reg_model.icl_block


def compress_embeddings(
    embeddings: Tensor,
    predictions: Tensor,
    method: str,
    task: str,
) -> Tensor:
    if method == "mean_pool":
        return embeddings.mean(dim=-2, keepdim=True)

    if method == "top_k":
        if task == "classification":
            proba = predictions.float().softmax(dim=-1)
            entropy = -(proba * proba.clamp(min=1e-12).log()).sum(dim=-1)
        else:
            std = predictions.float().std(dim=-1)
            entropy = std

        k = min(TOP_K, embeddings.size(-2))
        _, indices = entropy.topk(k, dim=-1)
        # Gather top-k embeddings along the row dimension.
        indices_expanded = indices.unsqueeze(-1).expand(
            *indices.shape, embeddings.size(-1)
        )
        return embeddings.gather(-2, indices_expanded)

    # no_compress
    return embeddings


def compute_metrics(
    pred, y_true_df, target: str, task: str
) -> dict[str, float]:
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


def compute_embedding_drift(prev: Tensor, curr: Tensor) -> float:
    prev_flat = prev.reshape(-1, prev.size(-1))
    curr_flat = curr.reshape(-1, curr.size(-1))
    cos_sim = cosine_similarity(prev_flat, curr_flat, dim=-1)
    return float(1.0 - cos_sim.mean())


def make_inject_hook(feedback_tokens: Tensor):
    def hook(module, args, kwargs):
        kv = kwargs.get("key_value")
        if kv is not None and isinstance(kv, Tensor):
            kwargs["key_value"] = torch.cat([kv, feedback_tokens], dim=-2)
        return args, kwargs

    return hook


def run_experiment(
    data_id: int,
    task: str,
    device: torch.device,
    compression: str,
    num_rounds: int = NUM_ROUNDS,
) -> list[dict[str, object]]:
    context, query, target, test_df = load_dataset(data_id, task, device)
    model = sdm.models.TabICLv2(device=device)
    icl_block = get_icl_block(model, task)

    round_results = []
    feedback_tokens = None
    prev_embeddings = None

    for r in range(num_rounds + 1):
        captured_embeddings: list[Tensor] = []
        captured_pre_head: list[Tensor] = []

        def capture_embedding(module, args):
            captured_pre_head.append(args[0].detach().clone())

        # Set up embedding capture hook.
        emb_handle = icl_block.head.register_forward_pre_hook(
            capture_embedding
        )

        # Set up feedback injection hooks for rounds > 0.
        inject_handles = []
        if feedback_tokens is not None:
            for layer in icl_block.layers:
                h = layer.register_forward_pre_hook(
                    make_inject_hook(feedback_tokens),
                    with_kwargs=True,
                )
                inject_handles.append(h)

        # Run forward pass.
        with torch.amp.autocast(
            device.type, torch.bfloat16, enabled=context.is_cuda
        ):
            pred = model(
                x_context=context.drop_columns(target),
                y_context=context[:, target],
                x_query=query.drop_columns(target),
                num_estimators=1,
            )

        # Clean up hooks.
        emb_handle.remove()
        for h in inject_handles:
            h.remove()

        # Compute metrics.
        metrics = compute_metrics(pred, test_df, target, task)

        # Compute embedding drift.
        embeddings = captured_pre_head[0] if captured_pre_head else None
        drift = float("nan")
        if prev_embeddings is not None and embeddings is not None:
            drift = compute_embedding_drift(prev_embeddings, embeddings)

        result = {
            "round": r,
            "compression": compression,
            **metrics,
            "embedding_drift": drift,
        }
        round_results.append(result)

        # Prepare feedback tokens for next round.
        if embeddings is not None:
            with torch.no_grad():
                # Get raw logits/predictions for entropy-based compression.
                raw_pred = captured_pre_head[0]
                feedback_tokens = compress_embeddings(
                    embeddings,
                    raw_pred,
                    compression,
                    task,
                )
            prev_embeddings = embeddings

    return round_results


def print_round_table(results: list[dict], task: str) -> None:
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results: list[dict[str, object]] = []

    for name, data_id, task in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"=== {name} ({task}) ===")

        for compression in COMPRESSION_METHODS:
            print(f"\nCompression: {compression}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = run_experiment(
                        data_id, task, device, compression
                    )
                for r in results:
                    r["name"] = name
                    r["data_id"] = data_id
                    r["task"] = task
                all_results.extend(results)
                print_round_table(results, task)
            except Exception:
                print("  FAILED:")
                traceback.print_exc(file=sys.stdout)

    # Print summary table.
    print(f"\n{'=' * 70}")
    print("SUMMARY: Round-4 performance vs baseline (round 0)")
    print()

    cls_datasets = sorted(
        set(
            r["name"]
            for r in all_results
            if r.get("task") == "classification" and r.get("round") == 0
        )
    )
    reg_datasets = sorted(
        set(
            r["name"]
            for r in all_results
            if r.get("task") == "regression" and r.get("round") == 0
        )
    )

    if cls_datasets:
        print("Classification (accuracy gain over baseline):")
        print(
            f"{'dataset':<25} {'baseline':>8} {'mean_pool':>10} {'top_k':>10} {'no_compress':>12}"
        )
        for name in cls_datasets:
            baseline = [
                r
                for r in all_results
                if r["name"] == name
                and r["round"] == 0
                and r["compression"] == "mean_pool"
            ]
            if not baseline:
                continue
            base_acc = baseline[0]["accuracy"]
            cols = [f"{base_acc:>8.3f}"]
            for comp in COMPRESSION_METHODS:
                r4 = [
                    r
                    for r in all_results
                    if r["name"] == name
                    and r["round"] == NUM_ROUNDS
                    and r["compression"] == comp
                ]
                if r4:
                    gain = r4[0]["accuracy"] - base_acc
                    cols.append(f"{gain:>+10.3f}")
                else:
                    cols.append(f"{'N/A':>10}")
            print(f"{name:<25} {'  '.join(cols)}")

    if reg_datasets:
        print(f"\nRegression (R² gain over baseline):")
        print(
            f"{'dataset':<25} {'baseline':>8} {'mean_pool':>10} {'top_k':>10} {'no_compress':>12}"
        )
        for name in reg_datasets:
            baseline = [
                r
                for r in all_results
                if r["name"] == name
                and r["round"] == 0
                and r["compression"] == "mean_pool"
            ]
            if not baseline:
                continue
            base_r2 = baseline[0]["r2"]
            cols = [f"{base_r2:>8.3f}"]
            for comp in COMPRESSION_METHODS:
                r4 = [
                    r
                    for r in all_results
                    if r["name"] == name
                    and r["round"] == NUM_ROUNDS
                    and r["compression"] == comp
                ]
                if r4:
                    gain = r4[0]["r2"] - base_r2
                    cols.append(f"{gain:>+10.3f}")
                else:
                    cols.append(f"{'N/A':>10}")
            print(f"{name:<25} {'  '.join(cols)}")

    # Save to CSV.
    csv_path = "feedback_token_results.csv"
    all_keys = [
        "name",
        "data_id",
        "task",
        "compression",
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
