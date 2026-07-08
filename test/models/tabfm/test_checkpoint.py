import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from sdm.models import TabFM
from sdm.models.tabfm.model import TabFMCore

_CONFIG: dict[str, Any] = {
    "embed_dim": 8,
    "max_classes": 4,
    "col_num_blocks": 1,
    "col_nhead": 2,
    "col_num_inds": 4,
    "row_num_blocks": 1,
    "row_nhead": 2,
    "row_num_cls": 2,
    "icl_num_blocks": 1,
    "icl_nhead": 2,
    "ff_factor": 2,
    "feature_group_size": 3,
    "num_freq": 4,
    "decoder_hidden": 16,
}


def _write_checkpoint(
    root: Path,
    *,
    use_safetensors: bool = True,
) -> tuple[TabFMCore, TabFMCore]:
    models = []
    for variant in ("classification", "regression"):
        is_classifier = variant == "classification"
        model = TabFMCore(
            **_CONFIG,
            is_classifier=is_classifier,
        ).eval()
        with torch.no_grad():
            model.cell_embedder.fourier_frequencies.normal_()
            model.cell_embedder.fourier_frequencies_cat.normal_()
        directory = root / variant
        directory.mkdir(parents=True)
        config = {
            **_CONFIG,
            "task": variant,
            "framework": "pytorch",
            "model_type": "tabfm",
            "version": "1.0.0",
        }
        (directory / "config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        if use_safetensors:
            save_file(model.state_dict(), directory / "model.safetensors")
        else:
            torch.save(model.state_dict(), directory / "pytorch_model.bin")
        models.append(model)
    return models[0], models[1]


@pytest.mark.parametrize("use_safetensors", [False, True])
def test_tabfm_loads_local_official_checkpoint_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_safetensors: bool,
) -> None:
    expected_cls, expected_reg = _write_checkpoint(
        tmp_path,
        use_safetensors=use_safetensors,
    )

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("local checkpoint loading attempted a download")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        fail_download,
    )
    model = TabFM(pretrained=True, checkpoint_path=tmp_path)

    for loaded, expected in (
        (model.cls_model, expected_cls),
        (model.reg_model, expected_reg),
    ):
        assert loaded.training is False
        assert loaded.state_dict().keys() == expected.state_dict().keys()
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[key])

    input = torch.randn(2, 7, 5)
    class_target = torch.randint(0, 4, (2, 4))
    regression_target = torch.randn(2, 4)
    torch.testing.assert_close(
        model(input, class_target),
        TabFM(
            cls_model=expected_cls,
            reg_model=expected_reg,
        )(input, class_target),
    )
    torch.testing.assert_close(
        model(input, regression_target),
        TabFM(
            cls_model=expected_cls,
            reg_model=expected_reg,
        )(input, regression_target),
    )


def test_tabfm_checkpoint_honors_dtype(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    model = TabFM(
        pretrained=True,
        checkpoint_path=tmp_path,
        dtype=torch.bfloat16,
    )

    assert next(model.cls_model.parameters()).dtype == torch.bfloat16
    assert next(model.reg_model.parameters()).dtype == torch.bfloat16
    output = model(
        torch.randn(7, 5),
        torch.randint(0, 4, (4,)),
    )
    assert output.dtype == torch.bfloat16


def test_tabfm_checkpoint_rejects_missing_variant(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    regression = tmp_path / "regression"
    for path in regression.iterdir():
        path.unlink()
    regression.rmdir()

    with pytest.raises(FileNotFoundError, match="regression"):
        TabFM(pretrained=True, checkpoint_path=tmp_path)


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"task": "regression"}, "does not match"),
        ({"unknown_key": 1}, "Unsupported TabFM config keys"),
        ({"is_classifier": "yes"}, "must be boolean"),
    ],
)
def test_tabfm_checkpoint_rejects_invalid_config(
    tmp_path: Path,
    update: dict[str, object],
    match: str,
) -> None:
    _write_checkpoint(tmp_path)
    path = tmp_path / "classification/config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if "is_classifier" in update:
        config.pop("task")
    config.update(update)
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        TabFM(pretrained=True, checkpoint_path=tmp_path)


def test_tabfm_checkpoint_load_is_strict(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    path = tmp_path / "classification/model.safetensors"
    state_dict = load_file(path)
    state_dict.pop("cls_tokens")
    save_file(state_dict, path)

    with pytest.raises(RuntimeError, match="Missing key"):
        TabFM(pretrained=True, checkpoint_path=tmp_path)
