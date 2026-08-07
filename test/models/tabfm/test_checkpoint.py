import hashlib
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from sdm.models.tabfm import checkpoint
from sdm.models.tabfm.core import _TabFM
from sdm.testing import withCUDA

_Task = Literal["classification", "regression"]
_TINY_KWARGS: dict[str, Any] = {
    "embed_dim": 4,
    "max_classes": 3,
    "col_num_blocks": 1,
    "col_num_heads": 2,
    "col_num_inducing_points": 2,
    "row_num_blocks": 1,
    "row_num_heads": 2,
    "row_num_cls": 2,
    "icl_num_blocks": 1,
    "icl_num_heads": 2,
    "feedforward_factor": 2,
    "feature_group_size": 2,
    "num_frequencies": 2,
    "decoder_hidden_channels": 5,
}

# SafeTensor metadata from google/tabfm-1.0.0-pytorch at revision
# 77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83.
_RELEASE_SCHEMA_FINGERPRINTS = {
    "classification": (
        "eef3a61c6d85e5c71303f5977c3b86344a0ceaae30df936f731411ea1f1a59a3"
    ),
    "regression": (
        "923a5b2d475f47dab8239aeeb92d77fd63ae07834518b38cca6a0eff7953d9a8"
    ),
}


def _official_key(key: str) -> str:
    if key.endswith(".rope.inv_freq"):
        return key.removesuffix(".rope.inv_freq") + ".rope.freqs"
    return key


def _tiny_model(
    task: _Task,
    *,
    dtype: torch.dtype = torch.float32,
) -> _TabFM:
    return _TabFM(
        **_TINY_KWARGS,
        is_classifier=task == "classification",
        device="meta",
        dtype=dtype,
    )


def _official_state_dict(model: _TabFM) -> dict[str, Tensor]:
    return {
        _official_key(key): torch.empty(tensor.shape, dtype=torch.float32)
        for key, tensor in model.state_dict().items()
    }


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_release_config_matches_official_state_schema(task: _Task) -> None:
    model = _TabFM(
        **checkpoint._V1_0_0_MODEL_KWARGS,
        is_classifier=task == "classification",
        device="meta",
        dtype=torch.float32,
    )
    state_dict = {
        _official_key(key): tensor
        for key, tensor in model.state_dict().items()
    }
    schema = "".join(
        f"{key}\tF32\t{','.join(map(str, state_dict[key].shape))}\n"
        for key in sorted(state_dict)
    )
    assert {tensor.dtype for tensor in state_dict.values()} == {torch.float32}
    assert (
        hashlib.sha256(schema.encode()).hexdigest()
        == _RELEASE_SCHEMA_FINGERPRINTS[task]
    )


def test_rope_remap_is_exact_and_collision_safe() -> None:
    model = _tiny_model("classification")
    state_dict = _official_state_dict(model)
    rope_key = next(key for key in state_dict if key.endswith(".rope.freqs"))
    rope_tensor = state_dict[rope_key]

    checkpoint._remap_rope_state_dict(state_dict)

    assert set(state_dict) == set(model.state_dict())
    target_key = rope_key.removesuffix("freqs") + "inv_freq"
    assert state_dict[target_key] is rope_tensor

    state_dict = _official_state_dict(model)
    state_dict[target_key] = state_dict[rope_key]
    with pytest.raises(RuntimeError, match="both map"):
        checkpoint._remap_rope_state_dict(state_dict)

    del state_dict[rope_key]
    with pytest.raises(RuntimeError, match="official key"):
        checkpoint._remap_rope_state_dict(state_dict)


@pytest.mark.parametrize("corruption", ["keys", "shape", "task", "dtype"])
def test_model_loading_is_strict(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    monkeypatch.setattr(checkpoint, "_V1_0_0_MODEL_KWARGS", _TINY_KWARGS)
    state_dict = _official_state_dict(_tiny_model("classification"))
    is_classifier = True
    if corruption == "keys":
        del state_dict["cls_tokens"]
        state_dict["unexpected"] = torch.empty(1)
    elif corruption == "shape":
        state_dict["cls_tokens"] = torch.empty(1)
    elif corruption == "task":
        is_classifier = False
    else:
        state_dict["cls_tokens"] = state_dict["cls_tokens"].to(torch.bfloat16)

    message = "must be float32" if corruption == "dtype" else None
    with pytest.raises(RuntimeError, match=message):
        checkpoint._model_from_state_dict(
            state_dict,
            is_classifier,
            device="cpu",
            dtype=torch.float32,
        )


@withCUDA
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_meta_assignment_preserves_target_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    task: _Task,
) -> None:
    monkeypatch.setattr(checkpoint, "_V1_0_0_MODEL_KWARGS", _TINY_KWARGS)
    source = _official_state_dict(_tiny_model(task, dtype=torch.bfloat16))

    model = checkpoint._model_from_state_dict(
        source,
        task == "classification",
        device=device,
        dtype=torch.bfloat16,
    )

    assert source == {}
    assert not model.training
    assert model.is_classifier == (task == "classification")
    assert {parameter.dtype for parameter in model.parameters()} == {
        torch.bfloat16
    }
    assert {tensor.device for tensor in model.state_dict().values()} == {
        device
    }
    assert model.cell_embedder.fourier_frequencies.dtype == torch.float32
    assert model.cell_embedder.fourier_frequencies_cat.dtype == torch.float32


def test_none_dtype_keeps_float32_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoint, "_V1_0_0_MODEL_KWARGS", _TINY_KWARGS)
    source = _official_state_dict(_tiny_model("classification"))
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = checkpoint._model_from_state_dict(
            source,
            True,
            device="cpu",
            dtype=None,
        )
    finally:
        torch.set_default_dtype(default_dtype)

    assert {tensor.dtype for tensor in model.state_dict().values()} == {
        torch.float32
    }


def test_local_loader_is_safetensor_only_and_uses_huggingface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dict = {"weight": torch.ones(1)}
    load = Mock(return_value=state_dict)
    build = Mock(return_value=object())
    monkeypatch.setattr(checkpoint, "load_state_dict_from_file", load)
    monkeypatch.setattr(checkpoint, "_model_from_state_dict", build)

    with pytest.raises(ValueError, match="SafeTensor"):
        checkpoint._load_tabfm_v1_0_0(
            tmp_path / "model.pt",
            task="regression",
        )
    path = tmp_path / "model.safetensors"
    result = checkpoint._load_tabfm_v1_0_0(
        path,
        task="regression",
        dtype=torch.float32,
    )
    assert result is build.return_value
    load.assert_called_once_with(path, map_location="cpu")
    build.assert_called_once_with(
        state_dict,
        False,
        device=None,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_huggingface_loader_is_pinned_and_task_specific(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: _Task,
) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    monkeypatch.setattr(checkpoint, "is_safetensors_available", lambda: True)

    download = Mock(return_value=str(checkpoint_path))
    load = Mock(return_value=object())
    monkeypatch.setattr(checkpoint, "download_checkpoint", download)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_file", load)

    result = checkpoint._load_tabfm_v1_0_0_from_huggingface(
        task=task,
        cache_dir=tmp_path,
        local_files_only=True,
        device="cpu",
        dtype=torch.float32,
    )

    assert result is load.return_value
    download.assert_called_once_with(
        repo_id="google/tabfm-1.0.0-pytorch",
        filename=f"{task}/model.safetensors",
        revision="77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83",
        cache_dir=tmp_path,
        local_files_only=True,
    )
    load.assert_called_once_with(
        str(checkpoint_path),
        is_classifier=task == "classification",
        device="cpu",
        dtype=torch.float32,
    )


def test_invalid_device_and_missing_dependency_fail_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = Mock(return_value=False)
    download = Mock()
    monkeypatch.setattr(checkpoint, "is_safetensors_available", available)
    monkeypatch.setattr(checkpoint, "download_checkpoint", download)

    with pytest.raises(ValueError, match="meta device"):
        checkpoint._load_tabfm_v1_0_0_from_huggingface(
            task="classification",
            device="meta",
        )
    available.assert_not_called()

    with pytest.raises(ImportError, match="optional 'safetensors'"):
        checkpoint._load_tabfm_v1_0_0_from_huggingface(task="classification")
    available.assert_called_once_with()
    download.assert_not_called()
