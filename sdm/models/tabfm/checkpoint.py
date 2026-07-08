import json
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file
from torch import Tensor

from sdm.models.tabfm.model import TabFMCore

_CONFIG_FILENAME = "config.json"
_SAFETENSORS_FILENAME = "model.safetensors"
_PYTORCH_FILENAME = "pytorch_model.bin"
_VARIANTS = ("classification", "regression")
_CONFIG_KEYS = {
    "embed_dim",
    "max_classes",
    "col_num_blocks",
    "col_nhead",
    "col_num_inds",
    "row_num_blocks",
    "row_nhead",
    "row_num_cls",
    "icl_num_blocks",
    "icl_nhead",
    "ff_factor",
    "feature_group_size",
    "num_freq",
    "decoder_hidden",
    "is_classifier",
}
_METADATA_KEYS = {"framework", "model_type", "version"}


def _load_local_cores(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> tuple[TabFMCore, TabFMCore]:
    root = Path(checkpoint_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"TabFM checkpoint directory does not exist: {root}"
        )

    models = {
        variant: _load_variant(
            root / variant,
            variant=variant,
            device=device,
            dtype=dtype,
        )
        for variant in _VARIANTS
    }
    return models["classification"], models["regression"]


def _load_variant(
    directory: Path,
    *,
    variant: str,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> TabFMCore:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing TabFM {variant} checkpoint directory: {directory}"
        )

    config = _load_config(directory / _CONFIG_FILENAME, variant=variant)
    state_dict = _load_state_dict(directory)
    model_dtype = dtype or _state_dict_dtype(state_dict)
    model = TabFMCore(
        **config,
        device="cpu",
        dtype=model_dtype,
    )
    model.load_state_dict(state_dict, strict=True)
    if device is not None:
        model = model.to(device=device)
    model.eval()
    return model


def _load_config(path: Path, *, variant: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing TabFM config file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"Unable to read TabFM config file: {path}"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"TabFM config must contain a JSON object: {path}")

    config = cast(dict[str, Any], raw.copy())
    task = config.pop("task", None)
    expected_classifier = variant == "classification"
    if task is not None:
        if task not in _VARIANTS:
            raise ValueError(f"Unsupported TabFM task in {path}: {task!r}")
        task_classifier = task == "classification"
        if task_classifier != expected_classifier:
            raise ValueError(
                f"TabFM config task {task!r} does not match its {variant} "
                "directory"
            )
        configured_classifier = config.get("is_classifier")
        if (
            configured_classifier is not None
            and configured_classifier != task_classifier
        ):
            raise ValueError(
                f"Conflicting task and is_classifier values in {path}"
            )
        config["is_classifier"] = task_classifier

    configured_classifier = config.get(
        "is_classifier",
        expected_classifier,
    )
    if not isinstance(configured_classifier, bool):
        raise ValueError(f"is_classifier must be boolean in {path}")
    if configured_classifier != expected_classifier:
        raise ValueError(
            f"TabFM is_classifier value does not match its {variant} directory"
        )
    config["is_classifier"] = expected_classifier

    for key in _METADATA_KEYS:
        config.pop(key, None)
    unexpected = sorted(set(config) - _CONFIG_KEYS)
    if unexpected:
        raise ValueError(
            f"Unsupported TabFM config keys in {path}: {', '.join(unexpected)}"
        )
    return config


def _load_state_dict(directory: Path) -> dict[str, Tensor]:
    safetensors_path = directory / _SAFETENSORS_FILENAME
    pytorch_path = directory / _PYTORCH_FILENAME
    if safetensors_path.is_file():
        return load_file(safetensors_path, device="cpu")
    if pytorch_path.is_file():
        state_dict = torch.load(
            pytorch_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state_dict, dict) or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in state_dict.items()
        ):
            raise ValueError(
                f"TabFM checkpoint must contain a tensor state dictionary: "
                f"{pytorch_path}"
            )
        return cast(dict[str, Tensor], state_dict)
    raise FileNotFoundError(
        f"Missing {_SAFETENSORS_FILENAME} or {_PYTORCH_FILENAME} in "
        f"{directory}"
    )


def _state_dict_dtype(state_dict: dict[str, Tensor]) -> torch.dtype:
    for value in state_dict.values():
        if value.is_floating_point():
            return value.dtype
    raise ValueError("TabFM checkpoint contains no floating-point tensors")
