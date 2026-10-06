# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot KumoTabular/TabICLv2 on Kaggle's "ITK Fall 2018" competition.

The competition's data (``aps_failure_training_set.arff``,
``aps_failure_test_set.arff``) is the APS Failure at Scania Trucks dataset:
170 numerical sensor features, a binary ``class`` target ({false, true})
with heavy missingness and heavy class imbalance (~1.7% positive), and
submissions keyed by 1-based row position in the test file rather than an
explicit id column. Missing values are left as NaN rather than imputed:
KumoTabular's default recipe (``Standardize``) fits statistics ignoring NaN
and preserves it through the transform, and the model itself embeds
missingness as a native signal rather than requiring pre-imputed inputs.

The dataset is the same one behind the IDA 2016 industrial challenge:
``Total_cost = 10*FP + 500*FN`` (a missed failure costs 50x an unnecessary
check), per the dataset's own UCI documentation. This Kaggle mirror's score
is confirmed (by reconciling its unlabeled, row-shuffled test set against
UCI's canonically labeled release and matching a real submission's score
exactly) to be ``-Total_cost / 16000`` -- the per-instance cost, negated so
higher is better. A 0.5 probability threshold is a poor fit for this 50:1
cost asymmetry; predictions are thresholded at the Bayes-optimal cutoff
``FP_cost / (FP_cost + FN_cost)`` instead, which minimizes expected cost.
"""

import argparse
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import sdm
from sdm.models.kumo.tabular.ckpt import remap_ckpt
from sdm.models.kumo.tabular.model import MODEL_KWARGS

# pandas infers the ARFF "class" column (values "false"/"true") as a
# native bool dtype, so sdm exposes it as a boolean-categorical target
# rather than a string one, keyed by the Python bool below.
TARGET_COLUMN = "class"
POSITIVE_CLASS = True
FALSE_POSITIVE_COST = 10.0
FALSE_NEGATIVE_COST = 500.0
# Bayes-optimal decision threshold minimizing expected cost under this
# asymmetric cost matrix: predict positive when p(positive) exceeds
# FP_cost / (FP_cost + FN_cost), not the usual 0.5.
DECISION_THRESHOLD = FALSE_POSITIVE_COST / (
    FALSE_POSITIVE_COST + FALSE_NEGATIVE_COST
)


def _load_arff(path: Path) -> pd.DataFrame:
    """Parse a dense ARFF file into a DataFrame, treating "?" as missing."""
    lines = path.read_text().splitlines(keepends=True)
    columns = [
        line.split()[1]
        for line in lines
        if line.lower().startswith("@attribute")
    ]
    data_start = (
        next(
            i
            for i, line in enumerate(lines)
            if line.lower().startswith("@data")
        )
        + 1
    )
    data = "".join(lines[data_start:])
    return pd.read_csv(
        io.StringIO(data), header=None, names=columns, na_values="?"
    )


def _build_model(
    name: str,
    device: torch.device,
    local_checkpoint: Path | None = None,
) -> sdm.models.ICLModel:
    if name == "tabiclv2":
        return sdm.models.TabICLv2(task="classification", device=device)
    size = "small" if name == "kumo-small" else "large"
    if local_checkpoint is None:
        return sdm.models.KumoTabular(
            task="classification", size=size, device=device
        )
    # Loads a checkpoint file directly, bypassing the Hugging Face Hub
    # fetch in KumoTabular(pretrained=True), for environments without Hub
    # access to nvidia/Kumo-Tabular. Mirrors KumoTabular._load_from_pretrained.
    model = sdm.models.KumoTabular(
        task="classification", size=size, pretrained=False, device=device
    )
    ckpt = torch.load(local_checkpoint, map_location=device, weights_only=True)
    state_dict = remap_ckpt(
        ckpt=ckpt["model"],
        is_classifier=True,
        num_layers=MODEL_KWARGS[size]["num_embedding_layers"],
    )
    model.models["classification"].load_state_dict(state_dict, assign=True)
    return model


def _predict(
    model: sdm.models.ICLModel,
    x_context: sdm.TableTensor,
    y_context: sdm.TableTensor,
    x_query: sdm.TableTensor,
    *,
    num_estimators: int,
    max_context_size: int | None,
    generator: torch.Generator,
) -> sdm.TableTensor:
    """Zero-shot in-context fit + predict, reusing one cached context.

    If ``max_context_size`` is set and the context exceeds it, falls back to
    per-estimator random context subsampling (`benchmark/tabular/model.py`'s
    ``SDMModel._fit``) instead of attending over the full context at once.
    """
    device = x_context.device
    effective_num_estimators: int | None = num_estimators
    if max_context_size is not None and x_context.size(0) > max_context_size:
        num_repeats = math.ceil(
            num_estimators * max_context_size / x_context.size(0)
        )
        perm = torch.cat(
            [
                torch.randperm(
                    x_context.size(0), generator=generator, device=device
                )
                for _ in range(num_repeats)
            ]
        )[: num_estimators * max_context_size]
        shape = (num_estimators, max_context_size)
        x_context = x_context[perm].unflatten(0, shape)
        y_context = y_context[perm].unflatten(0, shape)
        x_query = x_query.expand(num_estimators, *x_query.size())
        effective_num_estimators = None

    with (
        torch.inference_mode(),
        torch.amp.autocast(
            device.type, torch.float16, enabled=x_context.is_cuda
        ),
    ):
        model.fit(
            x=x_context,
            y=y_context,
            num_estimators=effective_num_estimators,
            generator=generator,
        )
        out = model.predict(x_query)
    model.clear()
    return out


def _cost(
    predicted_positive: np.ndarray, target_positive: np.ndarray
) -> float:
    false_positives = (predicted_positive & ~target_positive).sum()
    false_negatives = (~predicted_positive & target_positive).sum()
    return (
        FALSE_POSITIVE_COST * false_positives
        + FALSE_NEGATIVE_COST * false_negatives
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing aps_failure_training_set.arff and "
            "aps_failure_test_set.arff, from `kaggle competitions download "
            "-c itk-fall-2018`."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("kumo-small", "kumo-large", "tabiclv2"),
        default="kumo-large",
    )
    parser.add_argument(
        "--local-checkpoint",
        type=Path,
        default=None,
        help=(
            "KumoTabular checkpoint file to load directly, bypassing the "
            "Hugging Face Hub download (for --model kumo-small/kumo-large)."
        ),
    )
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-context-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = _load_arff(args.data_dir / "aps_failure_training_set.arff")
    test_frame = _load_arff(args.data_dir / "aps_failure_test_set.arff").drop(
        columns=[TARGET_COLUMN]
    )

    stypes = sdm.infer_stypes(
        train_frame, overrides={TARGET_COLUMN: "categorical"}
    )
    feature_stypes = {k: v for k, v in stypes.items() if k != TARGET_COLUMN}
    model = _build_model(args.model, device, args.local_checkpoint)

    splitter = StratifiedShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    aucs: list[float] = []
    kaggle_scores: list[float] = []
    for split_train_index, split_val_index in splitter.split(
        train_frame, train_frame[TARGET_COLUMN]
    ):
        split_train = sdm.TableTensor.from_pandas(
            df=train_frame.iloc[split_train_index],
            stypes=stypes,
            device=device,
        )
        split_val = sdm.TableTensor.from_pandas(
            df=train_frame.iloc[split_val_index],
            stypes=stypes,
            device=device,
        )
        out = _predict(
            model,
            x_context=split_train.drop_columns(TARGET_COLUMN),
            y_context=split_train[:, TARGET_COLUMN],
            x_query=split_val.drop_columns(TARGET_COLUMN),
            num_estimators=args.num_estimators,
            max_context_size=args.max_context_size,
            generator=generator,
        )
        scores, target = sdm.evaluation.to_binary_class(
            out,
            split_val[:, TARGET_COLUMN],
            positive_class=POSITIVE_CLASS,
        )
        scores_np = scores.float().cpu().numpy()
        target_np = target.cpu().numpy()
        predicted_positive = scores_np >= DECISION_THRESHOLD
        aucs.append(roc_auc_score(target_np, scores_np))
        cost = _cost(predicted_positive, target_np)
        # Matches the competition's own per-instance-normalized score (see
        # module docstring); normalizing by the fold size rather than a
        # fixed constant keeps this comparable across --val-fraction values.
        kaggle_scores.append(-cost / len(target_np))

    auc = np.array(aucs)
    score = np.array(kaggle_scores)
    print(
        f"{args.model}  val_auc={auc.mean():.4f}+-{auc.std():.4f}  "
        f"val_score={score.mean():.4f}+-{score.std():.4f}  "
        f"({args.num_splits} splits of {args.val_fraction:.0%}; "
        "val_score approximates the competition's own metric "
        "(-(10*FP + 500*FN)/N), higher is better; decision threshold="
        f"{DECISION_THRESHOLD:.4f})"
    )

    full_train = sdm.TableTensor.from_pandas(
        df=train_frame,
        stypes=stypes,
        device=device,
    )
    query = sdm.TableTensor.from_pandas(
        df=test_frame,
        stypes=feature_stypes,
        device=device,
    )
    out = _predict(
        model,
        x_context=full_train.drop_columns(TARGET_COLUMN),
        y_context=full_train[:, TARGET_COLUMN],
        x_query=query,
        num_estimators=args.num_estimators,
        max_context_size=args.max_context_size,
        generator=generator,
    )
    positive_scores = (
        out[str(POSITIVE_CLASS)].numerical.squeeze(-1).float().cpu().numpy()
    )
    predicted_class = np.where(
        positive_scores >= DECISION_THRESHOLD, "true", "false"
    )

    submission = pd.DataFrame(
        {
            "ID": np.arange(1, len(predicted_class) + 1),
            TARGET_COLUMN: predicted_class,
        }
    )
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
