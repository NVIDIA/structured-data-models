# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from benchmark.tabular.scoringbench.main import main


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--finetune-epochs", "1"),
        ("--finetune_epochs", "1"),
        ("--finetune-iters-per-epoch", "1"),
        ("--finetune-lr", "0.01"),
        ("--finetune-train-size", "10"),
        ("--finetune-context-frac", "0.5"),
        ("--finetune-val-frac", "0.2"),
    ],
)
def test_finetune_options_require_finetune(
    option: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["scoringbench", "--scoringbench-path", "/missing", option, value],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--finetune-* options require --finetune" in capsys.readouterr().err
