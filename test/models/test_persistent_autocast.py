# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from research.multigpu.persistent_autocast import (
    PersistentAutocastEnsembleParallel,
)
from test.models.test_ensemble_parallel import _RandomCacheModel

from sdm.models import EnsembleParallel


def test_persistent_context_parity_and_precision_contract() -> None:
    x, y = torch.randn(16, 3), torch.randn(16, 1)
    with (
        EnsembleParallel([_RandomCacheModel()]) as reference,
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        reference.fit(
            x,
            y,
            num_estimators=4,
            generator=torch.Generator().manual_seed(7),
        )
        expected = reference.predict(x)
    with PersistentAutocastEnsembleParallel(
        [_RandomCacheModel(), _RandomCacheModel()], dtype=torch.bfloat16
    ) as model:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model.fit(
                x,
                y,
                num_estimators=4,
                generator=torch.Generator().manual_seed(7),
            )
            for _ in range(3):
                torch.testing.assert_close(
                    model.predict(x).numerical,
                    expected.numerical,
                    rtol=0,
                    atol=0,
                )
        with pytest.raises(ValueError, match="fixed worker precision"):
            model.predict(x)
    model.close()  # Closing twice must not exit another thread's state.
    assert not torch.is_autocast_enabled("cpu")
