import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXAMPLES_README = EXAMPLES / "README.md"
DOCS_EXAMPLES = ROOT / "docs" / "source" / "examples.md"


def _top_level_example_entries() -> list[str]:
    entries = []
    for path in sorted(EXAMPLES.iterdir()):
        if path.name.startswith(".") or path.name in {
            "README.md",
            "__pycache__",
        }:
            continue
        if path.is_dir():
            entries.append(f"examples/{path.name}/")
            continue
        entries.append(f"examples/{path.name}")
    return entries


def test_top_level_examples_are_indexed() -> None:
    index_texts = {
        "examples README": EXAMPLES_README.read_text(),
        "docs examples page": DOCS_EXAMPLES.read_text(),
    }

    missing = {
        entry: [
            index_name
            for index_name, index_text in index_texts.items()
            if entry not in index_text
        ]
        for entry in _top_level_example_entries()
    }
    missing = {entry: indexes for entry, indexes in missing.items() if indexes}

    assert not missing


def test_example_notebooks_are_stripped() -> None:
    for path in sorted(EXAMPLES.rglob("*.ipynb")):
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, (path, index)
            assert cell.get("outputs") == [], (path, index)
