# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot KumoTabular/TabICLv2 on Kaggle's Africa Soil Property
Prediction (AFSIS) competition.

Single flat table: a ``PIDN`` row id, ~3578 infrared-spectrum absorbance
features plus a handful of spatial/geographic features (one of which,
``Depth``, is categorical), and **five** simultaneous regression targets
(``Ca``, ``P``, ``pH``, ``SOC``, ``Sand`` -- already-scaled soil property
measurements, not raw units, so predictions can be negative). Far more
feature columns (~3578) than rows (1157), similar in shape to
`santander_value_prediction_challenge.py`, so the default recipe's column
cap is raised the same way (see ``_wide_column_recipe``, ``--max-columns``).

``KumoTabular.supports_multi_target`` is ``False``, so the five targets
can't be fit jointly in one call: this script fits and predicts each target
independently (same features, one target column at a time), reusing the
same underlying feature columns and recipe across all five. Evaluated by
"Mean Columnwise Root Mean Squared Error" -- plain (non-log) RMSE per
target, averaged across the five targets.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit

import sdm
import sdm.processing as sp
from sdm.models.kumo.tabular.ckpt import remap_ckpt
from sdm.models.kumo.tabular.model import MODEL_KWARGS

ID_COLUMN = "PIDN"
TARGET_COLUMNS = ("Ca", "P", "pH", "SOC", "Sand")


def _build_model(
    name: str,
    device: torch.device,
    local_checkpoint: Path | None = None,
) -> sdm.models.ICLModel:
    if name == "tabiclv2":
        return sdm.models.TabICLv2(task="regression", device=device)
    size = "small" if name == "kumo-small" else "large"
    if local_checkpoint is None:
        return sdm.models.KumoTabular(
            task="regression", size=size, device=device
        )
    # Loads a checkpoint file directly, bypassing the Hugging Face Hub
    # fetch in KumoTabular(pretrained=True), for environments without Hub
    # access to nvidia/Kumo-Tabular. Mirrors KumoTabular._load_from_pretrained.
    model = sdm.models.KumoTabular(
        task="regression", size=size, pretrained=False, device=device
    )
    ckpt = torch.load(local_checkpoint, map_location=device, weights_only=True)
    state_dict = remap_ckpt(
        ckpt=ckpt["model"],
        is_classifier=False,
        num_layers=MODEL_KWARGS[size]["num_embedding_layers"],
    )
    model.models["regression"].load_state_dict(state_dict, assign=True)
    return model


def _wide_column_recipe(
    model: sdm.models.ICLModel, max_columns: int
) -> sp.Recipe:
    """Model's default recipe, with its ``SelectColumns`` feature cap raised.

    With ~3578 spectral features and only 1157 training rows, the default
    cap (500, "first") arbitrarily drops most columns with no informative
    ordering to justify it -- raised here (see
    `santander_value_prediction_challenge.py` for the analogous case), also
    switching to "round_robin" so each estimator sees a different chunk of
    columns instead of the identical first ``max_columns``.
    """
    recipe = model.default_recipe()
    for processor in recipe.features.modules():
        if isinstance(processor, sp.SelectColumns):
            processor.method = "round_robin"
            processor.max_columns = max_columns
    return recipe


def _predict(
    model: sdm.models.ICLModel,
    x_context: sdm.TableTensor,
    y_context: sdm.TableTensor,
    x_query: sdm.TableTensor,
    *,
    recipe: sp.Recipe,
    num_estimators: int,
    max_context_size: int | None,
    query_batch_size: int,
    generator: torch.Generator,
) -> sdm.TableTensor:
    """Zero-shot in-context fit + predict, reusing one cached context.

    If ``max_context_size`` is set and the context exceeds it, falls back to
    per-estimator random context subsampling (`benchmark/tabular/model.py`'s
    ``SDMModel._fit``) instead of attending over the full context at once.
    Predicts in ``query_batch_size``-row chunks, reusing the context
    ``fit()`` cached rather than recomputing it per chunk.
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
            recipe=recipe,
            num_estimators=effective_num_estimators,
            generator=generator,
        )
        # Row dim is always -2 (there may be a leading ensemble dim from
        # the max_context_size branch above).
        out = torch.cat(
            [
                model.predict(batch)
                for batch in x_query.split(query_batch_size, dim=-2)
            ],
            dim=-2,
        )
    model.clear()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing training.csv and sorted_test.csv (from "
            "the unzipped train.zip/test.zip), and sample_submission.csv, "
            "from `kaggle competitions download -c afsis-soil-properties`."
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
    parser.add_argument(
        "--max-columns",
        type=int,
        default=2000,
        help=(
            "Feature column cap (see _wide_column_recipe): raising this "
            "improves RMSE but costs memory, especially for kumo-large."
        ),
    )
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-context-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = pd.read_csv(args.data_dir / "training.csv").drop(
        columns=[ID_COLUMN]
    )
    test_ids = pd.read_csv(args.data_dir / "sorted_test.csv")[ID_COLUMN]
    test_frame = pd.read_csv(args.data_dir / "sorted_test.csv").drop(
        columns=[ID_COLUMN]
    )

    feature_columns = [
        c for c in train_frame.columns if c not in TARGET_COLUMNS
    ]
    stypes = sdm.infer_stypes(
        train_frame, overrides={t: "numerical" for t in TARGET_COLUMNS}
    )
    feature_stypes = {c: stypes[c] for c in feature_columns}
    model = _build_model(args.model, device, args.local_checkpoint)
    recipe = _wide_column_recipe(
        model, max_columns=min(args.max_columns, len(feature_columns))
    )

    splitter = ShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    rmses: dict[str, list[float]] = {t: [] for t in TARGET_COLUMNS}
    for target_column in TARGET_COLUMNS:
        frame = train_frame[[*feature_columns, target_column]]
        target_stypes = {**feature_stypes, target_column: stypes[target_column]}
        for split_train_index, split_val_index in splitter.split(frame):
            split_train = sdm.TableTensor.from_pandas(
                df=frame.iloc[split_train_index],
                stypes=target_stypes,
                device=device,
            )
            split_val = sdm.TableTensor.from_pandas(
                df=frame.iloc[split_val_index],
                stypes=target_stypes,
                device=device,
            )
            out = _predict(
                model,
                x_context=split_train.drop_columns(target_column),
                y_context=split_train[:, target_column],
                x_query=split_val.drop_columns(target_column),
                recipe=recipe,
                num_estimators=args.num_estimators,
                max_context_size=args.max_context_size,
                query_batch_size=args.query_batch_size,
                generator=generator,
            )
            # out.numerical holds 999 quantile columns; their mean is the
            # point estimate (matches examples/finetune/full_finetune.py).
            pred = out.numerical.mean(dim=-1).float().cpu().numpy()
            actual = (
                split_val[:, target_column]
                .numerical.squeeze(-1)
                .float()
                .cpu()
                .numpy()
            )
            rmses[target_column].append(
                float(np.sqrt(np.mean((pred - actual) ** 2)))
            )

    for target_column in TARGET_COLUMNS:
        values = np.array(rmses[target_column])
        print(
            f"{args.model}  {target_column}  val_rmse={values.mean():.4f}"
            f"+-{values.std():.4f}"
        )
    mean_columnwise_rmse = np.mean(
        [np.mean(rmses[t]) for t in TARGET_COLUMNS]
    )
    print(
        f"{args.model}  val_mean_columnwise_rmse={mean_columnwise_rmse:.4f}  "
        f"({args.num_splits} splits of {args.val_fraction:.0%})"
    )

    predictions: dict[str, np.ndarray] = {}
    for target_column in TARGET_COLUMNS:
        frame = train_frame[[*feature_columns, target_column]]
        target_stypes = {**feature_stypes, target_column: stypes[target_column]}
        full_train = sdm.TableTensor.from_pandas(
            df=frame, stypes=target_stypes, device=device
        )
        query = sdm.TableTensor.from_pandas(
            df=test_frame[feature_columns], stypes=feature_stypes, device=device
        )
        out = _predict(
            model,
            x_context=full_train.drop_columns(target_column),
            y_context=full_train[:, target_column],
            x_query=query,
            recipe=recipe,
            num_estimators=args.num_estimators,
            max_context_size=args.max_context_size,
            query_batch_size=args.query_batch_size,
            generator=generator,
        )
        predictions[target_column] = out.numerical.mean(dim=-1).float().cpu().numpy()

    submission = pd.DataFrame({ID_COLUMN: test_ids, **predictions})
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
