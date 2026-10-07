# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError


def download_checkpoint(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    license_prompt: str | None = None,
    config_filename: str | None = None,
) -> str:
    r"""Resolve a checkpoint from the Hugging Face cache or Hub.

    Authentication is delegated to :mod:`huggingface_hub`, which uses its
    configured login or the ``HF_TOKEN`` environment variable for private
    repositories.
    """
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        if local_files_only:
            raise
        if license_prompt is not None:
            print(f"{license_prompt}\n")  # noqa: T201
            answer = input("Accept license terms? [y/N] ")
            if answer.lower() not in {"y", "yes"}:
                raise RuntimeError(
                    "Checkpoint download requires license acceptance"
                ) from None
        if config_filename is not None:
            hf_hub_download(
                repo_id=repo_id,
                filename=config_filename,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=False,
            )
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
        )
