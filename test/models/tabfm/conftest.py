import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")

UPSTREAM_COMMIT = "633cd265f498e1d20c9625be0639f6305d8e2541"
DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parents[4] / "tabfm-reference"
REFERENCE_DIR = Path(
    os.environ.get("TABFM_REFERENCE_DIR", DEFAULT_REFERENCE_DIR)
).resolve()
UPSTREAM_MODEL = REFERENCE_DIR / "tabfm/src/pytorch/model.py"


@pytest.fixture(scope="session")
def upstream_tabfm_module() -> ModuleType:
    """Load the standalone upstream PyTorch module used for parity tests."""
    if not UPSTREAM_MODEL.is_file():
        pytest.skip(
            "Upstream TabFM checkout not found. Set TABFM_REFERENCE_DIR to "
            f"the checkout pinned at {UPSTREAM_COMMIT}."
        )

    spec = importlib.util.spec_from_file_location(
        "tabfm_upstream_pytorch_model",
        UPSTREAM_MODEL,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load upstream model from {UPSTREAM_MODEL}"
        )
    upstream_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream_module)
    return upstream_module
