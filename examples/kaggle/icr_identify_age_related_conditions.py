# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot KumoTabular/TabICLv2 cross-validation on Kaggle's ICR -
Identify Age-Related Conditions competition.

Single flat table: 56 anonymized feature columns (55 numerical, one
categorical column ``EJ``) and a binary ``Class`` target, 617 rows, with
scattered missing values in a handful of feature columns (largest: ``BQ``
and ``EL`` at 60 NaNs each). Four of the raw column headers (``BD ``,
``CD ``, ``CW ``, ``FD ``) have a trailing space in the source CSV; stripped
on load to avoid silent column-lookup bugs. Evaluated by "balanced
logarithmic loss": per-class log loss weighted by inverse class frequency,
so the ~17.5% positive rate doesn't dominate the score. Computed here via
``sklearn.metrics.log_loss``'s ``sample_weight``, with per-sample weight
``1 / count(sample's true class)`` -- summing to 2 across the two classes,
which is exactly the competition's own normalization.

This is a Kaggle *code* competition: ``test.csv`` ships with only 5 dummy
rows (the real hidden test set only exists inside Kaggle's own scored
notebook environment), so there is no local CSV to submit via the usual
`kaggle competitions submit` upload. This script only reports
cross-validated local metrics; it does not produce a submission file.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import sdm
from sdm.models.kumo.tabular.ckpt import remap_ckpt
from sdm.models.kumo.tabular.model import MODEL_KWARGS

ID_COLUMN = "Id"
TARGET_COLUMN = "Class"
# to_binary_class compares this against the target's raw (int) category
# values, not their stringified column-name form, so this must be an int.
POSITIVE_CLASS = 1


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing train.csv, from `kaggle competitions "
            "download -c icr-identify-age-related-conditions`."
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = pd.read_csv(args.data_dir / "train.csv")
    train_frame.columns = train_frame.columns.str.strip()
    train_frame = train_frame.drop(columns=[ID_COLUMN])

    stypes = sdm.infer_stypes(
        train_frame, overrides={TARGET_COLUMN: "categorical"}
    )
    model = _build_model(args.model, device, args.local_checkpoint)

    splitter = StratifiedShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    balanced_log_losses: list[float] = []
    aucs: list[float] = []
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
        class_counts = np.bincount(target_np.astype(np.int64))
        sample_weight = np.where(
            target_np, 1.0 / class_counts[1], 1.0 / class_counts[0]
        )
        balanced_log_losses.append(
            log_loss(target_np, scores_np, sample_weight=sample_weight)
        )
        aucs.append(roc_auc_score(target_np, scores_np))

    loss = np.array(balanced_log_losses)
    auc = np.array(aucs)
    print(
        f"{args.model}  val_balanced_log_loss={loss.mean():.4f}"
        f"+-{loss.std():.4f}  val_auc={auc.mean():.4f}+-{auc.std():.4f}  "
        f"({args.num_splits} splits of {args.val_fraction:.0%})"
    )


if __name__ == "__main__":
    main()
