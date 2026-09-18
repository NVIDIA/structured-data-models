# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot KumoTabular/TabICLv2 on Santander Value Prediction Challenge.

Single flat table: an ``ID`` row id, a continuous ``target`` (a customer
transaction value), and ~4991 anonymized numerical features -- far more
columns than rows (4991 features, 4459 training rows), unlike the
classification competitions in this directory. Evaluated by RMSLE. The
target is log1p-transformed once up front so the model's own recipe
(standardize/inverse-standardize) operates directly in log space, making
plain RMSE between log1p-transformed predictions and targets equal to
RMSLE; predictions are exponentiated back (``expm1``) only for the
submission file. The default recipe's column cap (500, "first") is also
raised (see ``_wide_column_recipe``, ``--max-columns``): cross-validation
shows RMSLE improving monotonically as more of the ~4991 columns are kept,
though kumo-large OOMs well before the full column count is reached.
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

ID_COLUMN = "ID"
TARGET_COLUMN = "target"


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

    With ~4991 anonymized features and only ~4459 training rows, the
    default cap (500, "first") arbitrarily drops ~90% of columns -- with no
    informative ordering to justify keeping the first 500 over any other
    500. Cross-validation shows RMSLE improving monotonically as the cap is
    raised (best at no cap at all), so it's raised well above the default
    here, also switching to "round_robin" so each estimator sees a
    different chunk of columns instead of the identical first ``max_columns``.
    ``max_columns`` is still capped well below the full column count by
    default: kumo-large OOMs at the full ~4991 columns with
    ``--num-estimators 8``.
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

    Predicts in ``query_batch_size``-row chunks rather than one call over
    the full query: per `docs/source/icl.md`'s documented fit-once/
    predict-per-batch pattern, ``predict()`` reuses the context cached by
    ``fit()`` rather than recomputing it, so the per-call buffer scales
    with the query chunk, not the full query size -- this is what lets a
    much larger ``--max-columns`` fit in memory for a large query (e.g.
    this competition's 49342-row test set) even though the context itself
    is small. A chunk size at or above the query size is one no-op chunk.
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
        # the max_context_size branch above), unlike the plain dim=0
        # split shown in docs/source/icl.md's simpler example.
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
            "Directory containing train.csv, test.csv, and "
            "sample_submission.csv, from `kaggle competitions download -c "
            "santander-value-prediction-challenge`."
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
            "improves RMSLE but costs memory, especially for kumo-large, "
            "which OOMs at the full ~4991 columns with --num-estimators 8."
        ),
    )
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-context-size", type=int, default=None)
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=5000,
        help=(
            "Predict in query-row chunks of this size, reusing the same "
            "fit()-cached context (see _predict), instead of one call over "
            "the full query -- keeps memory bounded for the 49342-row "
            "test set at higher --max-columns. Set to a value >= the "
            "query size (or pass a very large number) to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = pd.read_csv(args.data_dir / "train.csv").drop(
        columns=[ID_COLUMN]
    )
    train_frame[TARGET_COLUMN] = np.log1p(train_frame[TARGET_COLUMN])
    test_ids = pd.read_csv(args.data_dir / "test.csv")[ID_COLUMN]
    test_frame = pd.read_csv(args.data_dir / "test.csv").drop(
        columns=[ID_COLUMN]
    )

    stypes = sdm.infer_stypes(
        train_frame, overrides={TARGET_COLUMN: "numerical"}
    )
    feature_stypes = {k: v for k, v in stypes.items() if k != TARGET_COLUMN}
    model = _build_model(args.model, device, args.local_checkpoint)
    recipe = _wide_column_recipe(
        model, max_columns=min(args.max_columns, len(feature_stypes))
    )

    splitter = ShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    rmsles: list[float] = []
    for split_train_index, split_val_index in splitter.split(train_frame):
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
            recipe=recipe,
            num_estimators=args.num_estimators,
            max_context_size=args.max_context_size,
            query_batch_size=args.query_batch_size,
            generator=generator,
        )
        # out.numerical holds 999 quantile columns; their mean is the
        # point estimate (matches examples/finetune/full_finetune.py).
        pred_log = out.numerical.mean(dim=-1).float().cpu().numpy()
        actual_log = (
            split_val[:, TARGET_COLUMN]
            .numerical.squeeze(-1)
            .float()
            .cpu()
            .numpy()
        )
        rmsles.append(float(np.sqrt(np.mean((pred_log - actual_log) ** 2))))

    rmsle = np.array(rmsles)
    print(
        f"{args.model}  val_rmsle={rmsle.mean():.4f}+-{rmsle.std():.4f}  "
        f"({args.num_splits} splits of {args.val_fraction:.0%})"
    )

    full_train = sdm.TableTensor.from_pandas(
        df=train_frame, stypes=stypes, device=device
    )
    query = sdm.TableTensor.from_pandas(
        df=test_frame, stypes=feature_stypes, device=device
    )
    out = _predict(
        model,
        x_context=full_train.drop_columns(TARGET_COLUMN),
        y_context=full_train[:, TARGET_COLUMN],
        x_query=query,
        recipe=recipe,
        num_estimators=args.num_estimators,
        max_context_size=args.max_context_size,
        query_batch_size=args.query_batch_size,
        generator=generator,
    )
    pred_log = out.numerical.mean(dim=-1).float().cpu().numpy()
    predicted_target = np.clip(np.expm1(pred_log), a_min=0.0, a_max=None)

    submission = pd.DataFrame(
        {ID_COLUMN: test_ids, TARGET_COLUMN: predicted_target}
    )
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
