from pathlib import Path

import pytest
from huggingface_hub.utils import LocalEntryNotFoundError
from sdm.models import _huggingface


def test_download_checkpoint_prefers_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/cache/model.ckpt"

    monkeypatch.setattr(_huggingface, "hf_hub_download", download)

    path = _huggingface.download_checkpoint(
        repo_id="org/model",
        filename="model.ckpt",
        revision="v1",
        cache_dir=Path("cache"),
    )

    assert path == "/cache/model.ckpt"
    assert calls == [
        {
            "repo_id": "org/model",
            "filename": "model.ckpt",
            "revision": "v1",
            "cache_dir": Path("cache"),
            "local_files_only": True,
        }
    ]


def test_download_checkpoint_fetches_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            raise LocalEntryNotFoundError("not cached")
        return "/cache/model.ckpt"

    monkeypatch.setattr(_huggingface, "hf_hub_download", download)

    path = _huggingface.download_checkpoint(
        repo_id="org/model",
        filename="model.ckpt",
    )

    assert path == "/cache/model.ckpt"
    assert [call["local_files_only"] for call in calls] == [True, False]


def test_download_checkpoint_honors_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(_huggingface, "hf_hub_download", download)

    with pytest.raises(LocalEntryNotFoundError, match="not cached"):
        _huggingface.download_checkpoint(
            repo_id="org/model",
            filename="model.ckpt",
            local_files_only=True,
        )

    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
