# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path
from typing import Any


def test_nanotabicl() -> None:
    path = Path(__file__).parents[2] / "docs/source/nanotabicl.md"
    text = path.read_text()
    namespace: dict[str, Any] = {}

    for match in re.finditer(r"^```python\n(.*?)^```", text, re.M | re.S):
        source = "\n" * text.count("\n", 0, match.start(1)) + match[1]
        exec(compile(source, str(path), "exec"), namespace)

    out = namespace["out"]
    assert out.size() == (64, 10)
    assert out.isfinite().all()
