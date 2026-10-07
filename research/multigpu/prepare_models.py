# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a portable HF cache containing exact SDM Kumo checkpoint revisions."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def prepare(root: Path, family: str) -> None:
    """Stage checkpoint bytes, hashes and an offline-compatible HF cache."""
    models = {
        "nvidia/Kumo-Tabular": "3c3e10bbdb590ace29e7026847f92db3c603096d",
        "nvidia/Kumo-Relational": "2bd603d3d8f25f67a7aaa8579908595e20567e22",
    }
    if family != "all":
        models = {k: v for k, v in models.items() if k.endswith(family)}
    records = {}
    for repo, revision in models.items():
        source = Path(
            snapshot_download(
                repo,
                revision=revision,
                allow_patterns=[
                    "small/*.pt",
                    "large/*.pt",
                    "*.pt",
                    "config.json",
                    "README.md",
                    "LICENSE*",
                    "THIRD-PARTY*",
                ],
            )
        )
        cache_repo = root / "hub" / ("models--" + repo.replace("/", "--"))
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            dest = (
                cache_repo / "snapshots" / revision / path.relative_to(source)
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            records[str(dest.relative_to(root))] = {
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "bytes": dest.stat().st_size,
            }
        refs = cache_repo / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "v1.0.1").write_text(revision)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "revisions": models,
                "model_revision_in_sdm": "v1.0.1",
                "license": (
                    "OpenMDW 1.1; see model licenses and third-party notices"
                ),
                "usage": (
                    "Set HF_HUB_CACHE to absolute models/hub path and "
                    "HF_HUB_OFFLINE=1 before importing SDM"
                ),
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Portable models ready: {root}", flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--family", choices=["all", "Tabular", "Relational"], default="all"
    )
    args = parser.parse_args()
    prepare(args.root, args.family)
