from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sdm.models import TabFM
from sdm.models.tabfm import checkpoint


def test_remap_packs_attention_and_folds_class_bias() -> None:
    class_weight = torch.tensor([[1.003, 2.005], [3.007, 4.009]])
    class_bias = torch.tensor([0.003, 0.005])
    source = {
        "icl_predictor.tf_icl.blocks.0.attn.q_proj.weight": torch.full(
            size=(2, 2),
            fill_value=1.0,
        ),
        "icl_predictor.tf_icl.blocks.0.attn.k_proj.weight": torch.full(
            size=(2, 2),
            fill_value=2.0,
        ),
        "icl_predictor.tf_icl.blocks.0.attn.v_proj.weight": torch.full(
            size=(2, 2),
            fill_value=3.0,
        ),
        "icl_predictor.y_encoder.projection.weight": class_weight,
        "icl_predictor.y_encoder.projection.bias": class_bias,
    }
    expected = {
        "icl_block.layers.0.attn.qkv_lin.weight": torch.empty(6, 2),
        "icl_block.y_emb.weight": torch.empty(2, 2),
    }

    output = checkpoint._remap_state_dict(
        source=source,
        expected=expected,
    )

    assert source == {}
    assert output["icl_block.layers.0.attn.qkv_lin.weight"][:, 0].tolist() == [
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        3.0,
    ]
    expected_embedding = F.linear(
        input=torch.eye(2),
        weight=class_weight,
        bias=class_bias,
    )
    torch.testing.assert_close(
        output["icl_block.y_emb.weight"], expected_embedding
    )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_model_loads_the_requested_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: Literal["classification", "regression"],
) -> None:
    core = torch.nn.Identity()
    load = Mock(return_value=core)
    monkeypatch.setattr(checkpoint, "_load_tabfm_v1_0_0", load)
    path = tmp_path / "model.safetensors"

    model = TabFM(
        task=task,
        checkpoint_path=path,
        device="cpu",
    )

    assert model.model is core
    load.assert_called_once_with(
        path,
        task=task,
        device="cpu",
    )
