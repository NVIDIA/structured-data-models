# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from sdm.models import KumoAnomaly
from sdm.testing import withCUDA


@withCUDA
def test_forward(device: torch.device) -> None:
    model = KumoAnomaly(pretrained=False, device=device)
    x = torch.randn(2, 130, 3, device=device)

    out = model(x, num_samples=1, num_steps=1)
    assert out.size() == (2, 130, 1)
    assert out.columns["numerical"] == ("mae",)
    assert out.device == device
    assert out.numerical.isfinite().all()
    assert (out.numerical >= 0).all()


def test_load_from_pretrained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = KumoAnomaly(pretrained=False)
    ckpt = {}
    for key, value in reference.model.state_dict().items():
        if key.startswith("denoiser."):
            key = "diffmodel." + key.removeprefix("denoiser.")
            key = key.replace(".time_layer.", ".time_layer.layers.0.")
            key = key.replace(".feature_layer.", ".feature_layer.layers.0.")
            if key in (
                "diffmodel.input_projection.weight",
                "diffmodel.output_projection1.weight",
                "diffmodel.output_projection2.weight",
            ):
                value = value.unsqueeze(-1)
        else:
            key = key.removeprefix("side_embedding.")
        ckpt[key] = value

    path = tmp_path / "final_model.pth"
    torch.save({"model": ckpt}, path)
    monkeypatch.setattr(
        "sdm.models.kumo.timeseries.anomaly.model.download_checkpoint",
        lambda **kwargs: path,
    )
    model = KumoAnomaly(pretrained=True)
    x = torch.randn(17, 3)
    expected = reference(
        x=x,
        num_samples=1,
        num_steps=2,
        generator=torch.Generator().manual_seed(42),
    )
    out = model(
        x=x,
        num_samples=1,
        num_steps=2,
        generator=torch.Generator().manual_seed(42),
    )
    torch.testing.assert_close(out.numerical, expected.numerical)
