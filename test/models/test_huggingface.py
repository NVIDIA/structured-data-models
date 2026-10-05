# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from sdm.models import _huggingface
from sdm.models._huggingface import download_checkpoint


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    def download(
        repo_id: str,
        filename: str,
        *,
        revision: str | None,
        cache_dir: str | Path | None,
        local_files_only: bool,
    ) -> str:
        assert repo_id == "nvidia/Kumo-Relational"
        assert revision == "release-with-config"
        assert cache_dir == tmp_path
        path = tmp_path / filename
        if path.exists():
            return str(path)
        if local_files_only:
            raise LocalEntryNotFoundError("Not cached")
        if filename == "classifier.pt":
            assert (tmp_path / "config.json").read_text() == "{}"
        path.write_text("{}" if filename == "config.json" else "weights")
        return str(path)

    monkeypatch.setattr(_huggingface, "hf_hub_download", download)
    return tmp_path


def test_download_checkpoint_with_config(hub: Path) -> None:
    path = download_checkpoint(
        repo_id="nvidia/Kumo-Relational",
        filename="classifier.pt",
        revision="release-with-config",
        cache_dir=hub,
        config_filename="config.json",
    )
    assert Path(path).read_text() == "weights"
    assert (hub / "config.json").read_text() == "{}"


@pytest.mark.parametrize("local_files_only", [False, True])
def test_cached_checkpoint_without_config(
    hub: Path,
    local_files_only: bool,
) -> None:
    (hub / "classifier.pt").write_text("cached weights")
    path = download_checkpoint(
        repo_id="nvidia/Kumo-Relational",
        filename="classifier.pt",
        revision="release-with-config",
        cache_dir=hub,
        local_files_only=local_files_only,
        config_filename="config.json",
    )
    assert Path(path).read_text() == "cached weights"
    assert not (hub / "config.json").exists()


def test_local_only_checkpoint_missing(hub: Path) -> None:
    with pytest.raises(LocalEntryNotFoundError):
        download_checkpoint(
            repo_id="nvidia/Kumo-Relational",
            filename="classifier.pt",
            revision="release-with-config",
            cache_dir=hub,
            local_files_only=True,
            config_filename="config.json",
        )
    assert not list(hub.iterdir())


def test_download_checkpoint_without_config(hub: Path) -> None:
    path = download_checkpoint(
        repo_id="nvidia/Kumo-Relational",
        filename="regressor.pt",
        revision="release-with-config",
        cache_dir=hub,
    )
    assert Path(path).read_text() == "weights"
    assert not (hub / "config.json").exists()
