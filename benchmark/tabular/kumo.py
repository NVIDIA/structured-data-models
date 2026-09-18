# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared KumoTabular construction for the tabular benchmark harness."""

import os
from pathlib import Path
from typing import Literal

import huggingface_hub.errors
import torch

import sdm
from sdm.models.kumo.tabular.ckpt import remap_ckpt
from sdm.models.kumo.tabular.model import MODEL_KWARGS

Size = Literal["small", "large"]
Task = Literal["classification", "regression"]

_CHECKPOINT_ROOT = Path(
    os.environ.get("SDM_KUMO_CHECKPOINT_DIR")
    or (Path.home() / "KumoTFM-Checkpoints")
)
_LOCAL_CHECKPOINTS: dict[tuple[Size, Task], Path] = {
    ("small", "classification"): (
        _CHECKPOINT_ROOT / "kumo-tabular-cls-s3-27m-priorrefine-12k.pt"
    ),
    ("small", "regression"): (
        _CHECKPOINT_ROOT / "kumo-tabular-reg-s3-28m-priorrefine-12k.pt"
    ),
    ("large", "classification"): (
        _CHECKPOINT_ROOT / "kumo-tabular-cls-s3-61m-priorrefine-12k.pt"
    ),
    ("large", "regression"): (
        _CHECKPOINT_ROOT / "kumo-tabular-reg-s3-62m-priorrefine-12k.pt"
    ),
}


def load_kumo_tabular(
    task: Task,
    size: Size,
    device: torch.device,
) -> sdm.models.KumoTabular:
    """Construct a pretrained ``KumoTabular``.

    Falls back to a local checkpoint under ``~/KumoTFM-Checkpoints`` if the
    Hugging Face Hub fetch of ``nvidia/Kumo-Tabular`` fails (e.g.
    gated/missing access).
    """
    try:
        return sdm.models.KumoTabular(task=task, size=size, device=device)
    except huggingface_hub.errors.HfHubHTTPError:
        print(
            f"load_kumo_tabular: HF fetch failed for size={size!r} "
            f"task={task!r}, falling back to local checkpoint"
        )
        model = sdm.models.KumoTabular(
            task=task,
            size=size,
            pretrained=False,
            device=device,
        )
        ckpt = torch.load(
            _LOCAL_CHECKPOINTS[(size, task)],
            map_location=device,
            weights_only=True,
        )
        state_dict = remap_ckpt(
            ckpt=ckpt["model"],
            is_classifier=task == "classification",
            num_layers=MODEL_KWARGS[size]["num_embedding_layers"],
        )
        model.models[task].load_state_dict(state_dict, assign=True)
        return model
