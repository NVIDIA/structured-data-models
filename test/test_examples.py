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


def test_notebooks_are_not_checked_into_examples() -> None:
    notebooks = sorted(EXAMPLES.rglob("*.ipynb"))

    assert not notebooks
