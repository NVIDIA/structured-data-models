"""Load Google TabFM v1.0.0 checkpoints.

Released weights use ``tabfm-non-commercial-v1.0`` and may not be used for
commercial or production purposes or redistributed. See the pinned
`license <https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83/LICENSE>`_.
No weights are bundled with this package.
Local files are schema-validated, not authenticated as released artifacts.
"""

from pathlib import Path
from typing import Any, Literal, cast

import torch
from huggingface_hub import load_state_dict_from_file
from huggingface_hub.utils import is_safetensors_available
from torch import Tensor

from sdm.models._huggingface import download_checkpoint
from sdm.models.tabfm.core import _TabFM

_Task = Literal["classification", "regression"]

_HF_REPO_ID = "google/tabfm-1.0.0-pytorch"
_HF_REVISION = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
_V1_0_0_TASKS: dict[_Task, tuple[bool, str]] = {
    "classification": (True, "classification/model.safetensors"),
    "regression": (False, "regression/model.safetensors"),
}
_V1_0_0_MODEL_KWARGS: dict[str, Any] = {
    "embed_dim": 256,
    "max_classes": 10,
    "col_num_blocks": 3,
    "col_num_heads": 4,
    "col_num_inducing_points": 256,
    "row_num_blocks": 3,
    "row_num_heads": 8,
    "row_num_cls": 8,
    "icl_num_blocks": 24,
    "icl_num_heads": 8,
    "feedforward_factor": 4,
    "feature_group_size": 3,
    "num_frequencies": 32,
    "decoder_hidden_channels": None,
}


def _target_device(device: torch.device | str | None) -> torch.device:
    target = torch.device("cpu" if device is None else device)
    if target.type == "meta":
        raise ValueError(
            "TabFM checkpoints cannot be loaded onto a meta device"
        )
    return target


def _remap_rope_state_dict(state_dict: dict[str, Tensor]) -> None:
    for source_key in tuple(state_dict):
        if source_key.endswith(".rope.inv_freq"):
            official_key = source_key.removesuffix("inv_freq") + "freqs"
            if official_key in state_dict:
                raise RuntimeError(
                    f"TabFM checkpoint keys {official_key!r} and "
                    f"{source_key!r} both map to {source_key!r}"
                )
            raise RuntimeError(
                f"TabFM checkpoints must use the official key {official_key!r}"
            )
        if not source_key.endswith(".rope.freqs"):
            continue
        target_key = source_key.removesuffix("freqs") + "inv_freq"
        if target_key in state_dict:
            raise RuntimeError(
                f"TabFM checkpoint keys {source_key!r} and {target_key!r} "
                f"both map to {target_key!r}"
            )
        state_dict[target_key] = state_dict.pop(source_key)


def _model_from_state_dict(
    state_dict: dict[str, Tensor],
    is_classifier: bool,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> _TabFM:
    target_device = _target_device(device)
    if any(value.dtype != torch.float32 for value in state_dict.values()):
        raise RuntimeError("Official TabFM checkpoint tensors must be float32")
    model_dtype = torch.float32 if dtype is None else dtype
    model = _TabFM(
        **_V1_0_0_MODEL_KWARGS,
        is_classifier=is_classifier,
        device="meta",
        dtype=model_dtype,
    )
    expected_state_dict = model.state_dict()
    _remap_rope_state_dict(state_dict)
    model.load_state_dict(
        {key: value.to(device="meta") for key, value in state_dict.items()},
        strict=True,
    )
    for key, value in state_dict.items():
        state_dict[key] = value.to(
            device=target_device,
            dtype=expected_state_dict[key].dtype,
        )
    model.load_state_dict(state_dict, strict=True, assign=True)
    state_dict.clear()
    model.eval()
    return model


def _load_checkpoint_file(
    checkpoint_path: str | Path,
    *,
    is_classifier: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
) -> _TabFM:
    _target_device(device)
    path = Path(checkpoint_path)
    if path.suffix != ".safetensors":
        raise ValueError("TabFM checkpoints must use the SafeTensor format")
    state_dict = cast(
        dict[str, Tensor],
        load_state_dict_from_file(path, map_location="cpu"),
    )
    return _model_from_state_dict(
        state_dict,
        is_classifier,
        device=device,
        dtype=dtype,
    )


def _load_tabfm_v1_0_0(
    checkpoint_path: str | Path,
    *,
    task: _Task,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
) -> _TabFM:
    is_classifier, _ = _V1_0_0_TASKS[task]
    return _load_checkpoint_file(
        checkpoint_path,
        is_classifier=is_classifier,
        device=device,
        dtype=dtype,
    )


def _load_tabfm_v1_0_0_from_huggingface(
    *,
    task: _Task,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
) -> _TabFM:
    is_classifier, filename = _V1_0_0_TASKS[task]
    _target_device(device)
    # Fail before a multi-gigabyte download if the dependency is unavailable.
    if not is_safetensors_available():
        raise ImportError(
            "Loading TabFM weights requires the optional 'safetensors' package"
        )
    checkpoint_path = download_checkpoint(
        repo_id=_HF_REPO_ID,
        filename=filename,
        revision=_HF_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return _load_checkpoint_file(
        checkpoint_path,
        is_classifier=is_classifier,
        device=device,
        dtype=dtype,
    )
