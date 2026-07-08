import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch
from sdm.cache import KVCacheEntry
from sdm.models import TabFM


@pytest.fixture(scope="module")
def official_checkpoint_dir() -> Path:
    configured = os.environ.get("TABFM_CHECKPOINT_DIR")
    if configured is None:
        pytest.skip(
            "Official TabFM checkpoints are not configured. Set "
            "TABFM_CHECKPOINT_DIR to an appropriately licensed local "
            "checkpoint directory."
        )
    checkpoint_dir = Path(configured).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        pytest.fail(
            f"TABFM_CHECKPOINT_DIR does not name a directory: {checkpoint_dir}"
        )
    return checkpoint_dir


def _upstream_config(root: Path, *, variant: str) -> dict[str, Any]:
    path = root / variant / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = cast(dict[str, Any], raw)
    task = config.pop("task", variant)
    if task != variant:
        raise ValueError(
            f"Official config task {task!r} does not match {variant!r}"
        )
    for key in ("framework", "model_type", "version"):
        config.pop(key, None)
    config["is_classifier"] = variant == "classification"
    return config


@pytest.mark.parametrize("variant", ["classification", "regression"])
def test_official_checkpoint_matches_upstream_and_cached_replay(
    official_checkpoint_dir: Path,
    upstream_tabfm_module: ModuleType,
    variant: str,
) -> None:
    model = TabFM(
        pretrained=True,
        checkpoint_path=official_checkpoint_dir,
    )
    is_classifier = variant == "classification"
    core = model.cls_model if is_classifier else model.reg_model
    input = torch.randn(1, 5, 4, dtype=next(core.parameters()).dtype)
    if is_classifier:
        context_target = torch.randint(0, core.max_classes, (1, 3))
    else:
        context_target = torch.randn(1, 3, dtype=input.dtype)

    expected = model(input, context_target)
    repeated = model(input, context_target)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)

    model.fit(input[:, :3], context_target)
    cached = model.predict(input[:, 3:])
    torch.testing.assert_close(cached, expected)
    assert model._cache is not None
    assert "x" not in model._cache
    assert "y" not in model._cache
    assert any(
        isinstance(value, KVCacheEntry) for value in model._cache.values()
    )

    if is_classifier:
        del model.reg_model
    else:
        del model.cls_model
    del model

    upstream = upstream_tabfm_module.TabFM(
        **_upstream_config(official_checkpoint_dir, variant=variant)
    ).to(dtype=input.dtype)
    upstream.load_state_dict(core.state_dict(), strict=True)
    upstream.eval()
    padded_target = context_target.new_full((1, input.size(1)), -100)
    padded_target[:, : context_target.size(1)] = context_target
    upstream_output = upstream(
        input,
        padded_target,
        torch.tensor([context_target.size(1)]),
    )

    torch.testing.assert_close(upstream_output[:, 3:], expected)
