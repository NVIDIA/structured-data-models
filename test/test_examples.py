import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXAMPLES_README = EXAMPLES / "README.md"


def _top_level_example_entries() -> list[str]:
    entries = []
    for path in sorted(EXAMPLES.iterdir()):
        if path.name.startswith(".") or path.name in {
            "README.md",
            "__pycache__",
        }:
            continue
        if path.is_dir():
            entries.append(f"{path.name}/")
            continue
        entries.append(path.name)
    return entries


def test_top_level_examples_are_indexed() -> None:
    index_text = EXAMPLES_README.read_text()

    missing = [
        entry
        for entry in _top_level_example_entries()
        if entry not in index_text
    ]

    assert not missing


def test_example_notebooks_are_stripped() -> None:
    for path in sorted(EXAMPLES.rglob("*.ipynb")):
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, (path, index)
            assert cell.get("outputs") == [], (path, index)
