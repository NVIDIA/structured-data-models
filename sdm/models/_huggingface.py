# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from contextlib import suppress

from huggingface_hub import hf_hub_download
from huggingface_hub import logging as hf_logging
from huggingface_hub.utils import LocalEntryNotFoundError


def download_checkpoint(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    license_prompt: str | None = None,
    config_filename: str | None = None,
) -> str:
    r"""Resolve a checkpoint from the Hugging Face cache or Hub.

    Authentication is delegated to :mod:`huggingface_hub`, which uses its
    configured login or the ``HF_TOKEN`` environment variable for private
    repositories.
    """
    if not local_files_only and config_filename is not None:
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        with suppress(Exception):
            hf_hub_download(
                repo_id=repo_id,
                filename=config_filename,
                revision=revision,
                local_files_only=False,
            )
        hf_logging.set_verbosity(verbosity)
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
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
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=False,
        )
