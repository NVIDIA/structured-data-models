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


def _official_block_name(key: str) -> str:
    key = key.replace(".inducing_block.", ".mab1.")
    key = key.replace(".output_block.", ".mab2.")
    for root in ("col_embedder", "col_embedder_2"):
        key = key.replace(f"{root}.tf_col.", f"{root}.tf_col.blocks.", 1)
    return key


def _official_key(key: str, target_keys: set[str]) -> str:
    if key.endswith(".inv_freq"):
        return key.split(".blocks.", 1)[0] + ".rope.freqs"

    if ".attn.query_transform." in key:
        prefix, suffix = key.split(".attn.query_transform.", 1)
        index = int(suffix.partition(".")[0])
        has_rope = f"{prefix}.attn.query_transform.0.inv_freq" in target_keys
        key = prefix + (
            ".attn.query_ln.weight"
            if index == int(has_rope)
            else ".attn.per_dim_scale"
        )
    elif ".attn.key_transform." in key:
        prefix, suffix = key.split(".attn.key_transform.", 1)
        index = int(suffix.partition(".")[0])
        has_rope = f"{prefix}.attn.key_transform.0.inv_freq" in target_keys
        if index == int(has_rope):
            key = prefix + ".attn.key_ln.weight"

    key = key.replace(
        "cell_embedder.num_freq", "cell_embedder.fourier_frequencies"
    )
    key = key.replace(
        "cell_embedder.cat_freq", "cell_embedder.fourier_frequencies_cat"
    )
    key = key.replace("cell_embedder.num_lin.", "cell_embedder.in_linear.")
    key = key.replace("cell_embedder.cat_lin.", "cell_embedder.in_linear_cat.")
    key = key.replace(
        "target_embedder.lookup.", "cell_embedder.y_embedder_lookup."
    )
    key = _official_block_name(key)
    key = key.replace(".query_norm.", ".pre_attn_ln.")
    key = key.replace(".key_value_norm.", ".pre_attn_ln.")
    key = key.replace(".post_attn_norm.", ".post_attn_ln.")
    key = key.replace(".attn.out_lin.", ".attn.out_proj.")
    key = key.replace(".mlp.0.", ".pre_ff_ln.")
    key = key.replace(".mlp.1.up_lin.", ".linear1.")
    key = key.replace(".mlp.1.gate_lin.", ".linear1_gate.")
    key = key.replace(".mlp.1.down_lin.", ".linear2.")
    key = key.replace(".mlp.2.", ".post_ff_ln.")
    return key.replace(".inducing_points", ".ind_vectors")


def _remap_state_dict(
    state_dict: dict[str, Tensor],
    expected_state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Translate the official checkpoint layout into SDM's module layout."""
    target_keys = set(expected_state_dict)
    remapped: dict[str, Tensor] = {}
    used: set[str] = set()
    for target_key, target in expected_state_dict.items():
        if target_key.endswith("attn.qkv_lin.weight"):
            prefix = _official_block_name(
                target_key.removesuffix("qkv_lin.weight")
            )
            source_keys = [f"{prefix}{name}_proj.weight" for name in "qkv"]
            if all(key in state_dict for key in source_keys):
                remapped[target_key] = torch.cat(
                    [state_dict[key] for key in source_keys]
                )
                used.update(source_keys)
            continue
        if target_key.endswith("attn.qkv_lin.bias"):
            prefix = _official_block_name(
                target_key.removesuffix("qkv_lin.bias")
            )
            source_keys = [f"{prefix}{name}_proj.bias" for name in "qkv"]
            if all(key in state_dict for key in source_keys):
                remapped[target_key] = torch.cat(
                    [state_dict[key] for key in source_keys]
                )
                used.update(source_keys)
            continue

        source_key = _official_key(target_key, target_keys)
        if source_key in state_dict:
            value = state_dict[source_key]
            if value.shape == target.shape:
                remapped[target_key] = value
                used.add(source_key)

    unexpected = set(state_dict) - used
    missing = target_keys - set(remapped)
    if missing or unexpected:
        raise RuntimeError(
            "TabFM checkpoint schema mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return remapped


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
    official_state_dict = state_dict
    state_dict = _remap_state_dict(official_state_dict, expected_state_dict)
    official_state_dict.clear()
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
