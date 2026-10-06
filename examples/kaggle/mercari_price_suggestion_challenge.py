# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot KumoTabular/TabICLv2 on Mercari Price Suggestion Challenge.

Single flat table: free-text ``name``/``item_description``, categorical
``category_name``/``brand_name``, numerical ``item_condition_id``/
``shipping``, and a continuous ``price`` target. Unlike every other
competition here, training data is enormous (1482535 rows) and the test
set even larger (3460725 rows -- this is the competition's "stage 2" test
set; only a submission scored against it is accepted post-competition,
per this competition's two-stage rules).

Full in-context attention over the whole training set is computationally
infeasible at this scale, so ``--max-context-size`` (default 8000)
subsamples context per estimator (`benchmark/tabular/model.py`'s pattern,
an opt-in fallback in this directory's other scripts; here it's the
default, not a fallback). Local validation similarly uses a small
absolute-size held-out sample (``--val-size``) rather than a fraction of
1482535 rows.

Text columns are embedded (sentence embedding + PCA, ``sp.SentenceTransformer``
+ ``sp.PCA``) and prepended to the model's default recipe
(`examples/tabiclv2/strable.py`'s "embed" option), since KumoTabular has no
native text support -- not ``sp.TFIDF`` (that module's other option),
since its character n-gram path unconditionally imports cuDF with no CPU
fallback, unavailable in this environment. Evaluated by RMSLE, so price is
log1p-transformed for fitting, matching
`santander_value_prediction_challenge.py`.
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

TARGET_COLUMN = "price"
TEXT_COLUMNS = ("name", "item_description")


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


def _text_recipe(
    model: sdm.models.ICLModel, text_model: str, embedding_dim: int
) -> sp.Recipe:
    """Model's default recipe, with text columns embedded first.

    KumoTabular's ``supported_feature_stypes`` is numerical-only and its
    default recipe has no ``StypeDispatch(text=...)`` branch, so text
    columns are dropped unless converted to numerical features upstream.
    Uses a sentence embedding + PCA (`examples/tabiclv2/strable.py`'s
    "embed" option) rather than ``sp.TFIDF``: TFIDF's character n-gram path
    unconditionally imports cuDF with no CPU fallback, which isn't
    installed here, while ``sp.SentenceTransformer`` degrades gracefully to
    a CPU tokenizer when cuDF is unavailable.
    """
    recipe = model.default_recipe()
    recipe.prepend_features(
        sp.StypeDispatch(
            text=[sp.SentenceTransformer(text_model), sp.PCA(embedding_dim)]
        )
    )
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
    ``fit()`` cached rather than recomputing it per chunk (see
    `docs/source/icl.md`) -- keeps memory bounded for the 3.46M-row test set.
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


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].fillna("")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing train.tsv, test_stg2.tsv, and "
            "sample_submission_stg2.csv (extracted from the .7z/.zip "
            "archives from `kaggle competitions download -c "
            "mercari-price-suggestion-challenge`)."
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
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--text-embedding-dim", type=int, default=64)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="Absolute held-out row count (not a fraction of 1.48M rows).",
    )
    parser.add_argument(
        "--max-context-size",
        type=int,
        default=8000,
        help=(
            "Per-estimator context row cap; unlike this directory's other "
            "scripts, this defaults to a real cap (not None/full-context), "
            "since train.tsv has 1482535 rows."
        ),
    )
    parser.add_argument("--query-batch-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = _load(args.data_dir / "train.tsv").drop(columns=["train_id"])
    train_frame[TARGET_COLUMN] = np.log1p(train_frame[TARGET_COLUMN])
    test_frame_full = _load(args.data_dir / "test_stg2.tsv")
    test_ids = test_frame_full["test_id"]
    test_frame = test_frame_full.drop(columns=["test_id"])

    stypes = sdm.infer_stypes(
        train_frame,
        overrides={
            TARGET_COLUMN: "numerical",
            **dict.fromkeys(TEXT_COLUMNS, "text"),
        },
    )
    feature_stypes = {k: v for k, v in stypes.items() if k != TARGET_COLUMN}
    model = _build_model(args.model, device, args.local_checkpoint)
    recipe = _text_recipe(
        model,
        text_model=args.text_model,
        embedding_dim=args.text_embedding_dim,
    )

    splitter = ShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_size,
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
        # point estimate (matches examples/finetune/full_finetune.py). Both
        # sides are still log1p-transformed here, so plain RMSE == RMSLE.
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
        f"({args.num_splits} splits, val_size={args.val_size}, "
        f"max_context_size={args.max_context_size})"
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
    predicted_price = np.clip(np.expm1(pred_log), a_min=0.0, a_max=None)

    submission = pd.DataFrame(
        {"test_id": test_ids, TARGET_COLUMN: predicted_price}
    )
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
