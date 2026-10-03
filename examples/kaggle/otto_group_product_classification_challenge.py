# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot KumoTabular/TabICLv2 on Otto Group Product Classification.

Single flat table: an ``id`` row id, a 9-class ``target`` (``Class_1``..
``Class_9``), and 93 numerical count features (``feat_1``..``feat_93``).
Evaluated by multi-class log loss. KumoTabular supports at most 10 classes
(`sdm/models/kumo/tabular/model.py`), so this dataset's 9 classes fit
directly with no special handling, and unlike
`santander_value_prediction_challenge.py`, 93 features is already under the
default recipe's column cap, so no column-selection override is needed
either. The test set is large (144368 rows), so predictions are still
batched via ``--query-batch-size``.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit

import sdm
from sdm.models.kumo.tabular.ckpt import remap_ckpt
from sdm.models.kumo.tabular.model import MODEL_KWARGS

ID_COLUMN = "id"
TARGET_COLUMN = "target"
CLASSES = tuple(f"Class_{i}" for i in range(1, 10))


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
    query_batch_size: int,
    generator: torch.Generator,
) -> sdm.TableTensor:
    """Zero-shot in-context fit + predict, reusing one cached context.

    If ``max_context_size`` is set and the context exceeds it, falls back to
    per-estimator random context subsampling (`benchmark/tabular/model.py`'s
    ``SDMModel._fit``) instead of attending over the full context at once.
    Predicts in ``query_batch_size``-row chunks, reusing the context
    ``fit()`` cached rather than recomputing it per chunk (see
    `docs/source/icl.md`) -- keeps memory bounded for the 144368-row test set.
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
            "Directory containing train.csv, test.csv, and "
            "sampleSubmission.csv, from `kaggle competitions download -c "
            "otto-group-product-classification-challenge`."
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
    parser.add_argument("--query-batch-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = pd.read_csv(args.data_dir / "train.csv").drop(
        columns=[ID_COLUMN]
    )
    test_ids = pd.read_csv(args.data_dir / "test.csv")[ID_COLUMN]
    test_frame = pd.read_csv(args.data_dir / "test.csv").drop(
        columns=[ID_COLUMN]
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
    log_losses: list[float] = []
    accuracies: list[float] = []
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
            query_batch_size=args.query_batch_size,
            generator=generator,
        )
        scores, indices = sdm.evaluation.to_class_indices(
            out, split_val[:, TARGET_COLUMN], missing_score=0.0
        )
        scores_np = scores.float().cpu().numpy()
        indices_np = indices.cpu().numpy()
        log_losses.append(
            log_loss(
                indices_np, scores_np, labels=list(range(scores_np.shape[-1]))
            )
        )
        accuracies.append(float((scores_np.argmax(-1) == indices_np).mean()))

    loss = np.array(log_losses)
    acc = np.array(accuracies)
    print(
        f"{args.model}  val_log_loss={loss.mean():.4f}+-{loss.std():.4f}  "
        f"val_accuracy={acc.mean():.4f}+-{acc.std():.4f}  "
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
        num_estimators=args.num_estimators,
        max_context_size=args.max_context_size,
        query_batch_size=args.query_batch_size,
        generator=generator,
    )
    columns = out.columns[sdm.Stype.numerical]
    indices = [columns.index(c) for c in CLASSES]
    probabilities = out.numerical[..., indices].float().cpu().numpy()

    submission = pd.DataFrame(probabilities, columns=pd.Index(CLASSES))
    submission.insert(0, ID_COLUMN, test_ids.to_numpy())
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
