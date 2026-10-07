"""Evidence retention must detect changed records and missing large outputs."""

import json
from pathlib import Path

import pytest
from research.multigpu.collect_evidence import collect, verify


def test_retains_records_without_copying_predictions(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text('{"status": "complete"}\n')
    (run / "predictions.npy").write_bytes(b"prediction bytes")
    (run / "classifier.pt").write_bytes(b"excluded model weights")
    index_path = collect([f"baseline={run}"], tmp_path / "evidence")
    index = json.loads(index_path.read_text())
    assert {entry["name"] for entry in index["runs"][0]["artifacts"]} == {
        "result.json",
        "predictions.npy",
    }
    assert not (index_path.parent / "baseline" / "predictions.npy").exists()
    assert verify(index_path) == {
        "archived_files": 1,
        "external_files": 0,
        "external_unchecked": 1,
    }
    assert verify(index_path, external=True)["external_files"] == 2
    (run / "predictions.npy").unlink()
    assert verify(index_path)["archived_files"] == 1
    with pytest.raises(ValueError, match="Missing or mismatched"):
        verify(index_path, external=True)


def test_detects_modified_records_and_index(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "error-rank0.json").write_text('{"status": "failed"}\n')
    index_path = collect([f"failed={run}"], tmp_path / "evidence")
    archived = index_path.parent / "failed" / "error-rank0.json"
    archived.write_text('{"status": "complete"}\n')
    with pytest.raises(ValueError, match="Missing or mismatched"):
        verify(index_path)
    index_path.write_text("{}\n")
    with pytest.raises(ValueError, match="index checksum"):
        verify(index_path)


def test_retains_failures_and_additive_audits(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    for name in ("failure.json", "quality-full-cohort-audit.json"):
        (run / name).write_text("{}\n")
    index_path = collect([f"attempt={run}"], tmp_path / "evidence")
    assert verify(index_path, external=True) == {
        "archived_files": 2,
        "external_files": 2,
        "external_unchecked": 0,
    }
