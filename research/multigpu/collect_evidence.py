"""Retain small benchmark records and index large local evidence by hash.

Each input is an explicit result directory, never a dataset/model cache.
Collection creates a fresh directory. Verification checks archived records
without requiring the original machine; --external also checks large files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_role(name: str) -> str | None:
    """Recognize benchmark outputs; exclude model and dataset cache files."""
    if name in {
        "result.json",
        "results.json",
        "workload.json",
        "config.json",
        "parity.json",
        "failure.json",
        "attempt.json",
        "backend.json",
        "quality-independent-audit.json",
        "runtime.txt",
        "pip-freeze.txt",
        "topology.txt",
        "gpu.csv",
        "model-verification.json",
        "provenance.json",
        "stdout.jsonl",
        "junit.xml",
        "resume-graphdp-gpu-contracts.xml",
    }:
        return "record"
    if re.fullmatch(r"(?:rank|error-rank)\d+\.json", name):
        return "record"
    if re.fullmatch(
        r"(?:cuda-)?(?:classifier|regressor)-(?:fp32|bf16|autocast_bf16)"
        r"-n\d+-b\d+\.json",
        name,
    ):
        return "record"
    if name.endswith("-analysis.json") or re.fullmatch(
        r"quality-[A-Za-z0-9_-]+-audit\.json", name
    ):
        return "record"
    if name in {"command.txt", "probe.py"}:
        return "command"
    if name.startswith("predictions") and Path(name).suffix in {".npy", ".pt"}:
        return "predictions"
    if name in {"context_parallel.pt", "native_sdpa.pt", "single_rank_lse.pt"}:
        return "predictions"
    if name in {"query_ids.npy", "targets.npy", "context-row-indices.npy"}:
        return "row-or-target-identity"
    if name.startswith(("profile", "trace", "nsys")) or Path(name).suffix in {
        ".nsys-rep",
        ".sqlite",
    }:
        return "profile"
    if name == "nvidia-smi.csv":
        return "telemetry"
    if name.endswith(".log"):
        return "log"
    return None


def collect(
    runs: list[str], output: Path, *, only_names: set[str] | None = None
) -> Path:
    """Copy small records and index all recognized outputs in explicit runs."""
    selected: dict[str, Path] = {}
    for item in runs:
        label, separator, directory = item.partition("=")
        if not separator or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", label
        ):
            raise ValueError("Use --run label=/absolute/result/directory")
        if label in selected:
            raise ValueError(f"Duplicate run label: {label}")
        path = Path(directory).resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"Expected a result directory: {path}")
        selected[label] = path
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for label, directory in sorted(selected.items()):
        artifacts = []
        for path in sorted(directory.iterdir()):
            if only_names is not None and path.name not in only_names:
                continue
            role = artifact_role(path.name)
            if not path.is_file() or role is None:
                continue
            entry = {
                "name": path.name,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_path": str(path),
            }
            if (
                role in {"record", "command"}
                and entry["bytes"] <= 2 * 1024 * 1024
            ):
                archived = output / label / path.name
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, archived)
                if sha256(archived) != entry["sha256"]:
                    raise RuntimeError(
                        f"Input changed during collection: {path}"
                    )
                entry["archived_path"] = str(archived.relative_to(output))
            artifacts.append(entry)
        if not any(item["role"] == "record" for item in artifacts):
            raise ValueError(
                f"No result/config/rank/error records found: {directory}"
            )
        records.append(
            {
                "label": label,
                "source_directory": str(directory),
                "artifacts": artifacts,
            }
        )
    index = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "Small raw benchmark records are archived; model weights and "
            "raw datasets are excluded. Larger outputs remain at recorded "
            "local paths and are bound by content hashes."
        ),
        "runs": records,
    }
    index_path = output / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    (output / "index.sha256").write_text(sha256(index_path) + "  index.json\n")
    return index_path


def verify(index_path: Path, *, external: bool = False) -> dict[str, int]:
    """Check archived records and optionally original larger artifacts."""
    expected_index = (
        index_path.with_name("index.sha256").read_text().split()[0]
    )
    if sha256(index_path) != expected_index:
        raise ValueError("Evidence index checksum mismatch")
    index = json.loads(index_path.read_text())
    checked = {
        "archived_files": 0,
        "external_files": 0,
        "external_unchecked": 0,
    }
    for run in index["runs"]:
        for item in run["artifacts"]:
            candidates = []
            if "archived_path" in item:
                candidates.append(
                    (
                        "archived_files",
                        index_path.parent / item["archived_path"],
                    )
                )
            if external:
                candidates.append(
                    ("external_files", Path(item["source_path"]))
                )
            elif "archived_path" not in item:
                checked["external_unchecked"] += 1
            for kind, path in candidates:
                if (
                    not path.is_file()
                    or path.stat().st_size != item["bytes"]
                    or sha256(path) != item["sha256"]
                ):
                    raise ValueError(f"Missing or mismatched evidence: {path}")
                checked[kind] += 1
    return checked


def main() -> None:
    """Collect selected result directories or verify a previous collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--run", action="append", required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument(
        "--include-name",
        action="append",
        help="Restrict collection to these exact recognized file names",
    )
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--index", type=Path, required=True)
    verify_parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    if args.command == "collect":
        result = {
            "index": str(
                collect(
                    args.run,
                    args.output,
                    only_names=set(args.include_name)
                    if args.include_name
                    else None,
                )
            )
        }
    else:
        result = verify(args.index, external=args.external)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
