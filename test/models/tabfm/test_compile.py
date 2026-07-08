import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).with_name("_compile_worker.py")


def test_tabfm_compile_paths_have_no_graph_breaks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "all",
            "false",
            "false",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            "isolated torch.compile check failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
