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


@pytest.mark.parametrize(
    ("source", "expected", "match"),
    [
        (
            {},
            {"cell_embedding.num_freq": torch.empty(1)},
            "Checkpoint is missing",
        ),
        (
            {"unexpected": torch.empty(1)},
            {},
            "Unmapped checkpoint parameters",
        ),
    ],
)
def test_remap_rejects_checkpoint_key_mismatches(
    source: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    match: str,
) -> None:
    with pytest.raises((KeyError, ValueError), match=match):
        checkpoint._remap_state_dict(source=source, expected=expected)


def test_remap_consumes_shared_row_rope() -> None:
    rope = torch.tensor([1.0, 2.0])
    source = {
        "row_interactor.tf_row.rope.freqs": rope,
        "row_interactor_2.tf_row.rope.freqs": rope.clone(),
    }
    expected = {
        "row_embedding.row_blocks.0.0.attn.query_transform.0.inv_freq": (
            torch.empty(2)
        ),
    }

    output = checkpoint._remap_state_dict(
        source=source,
        expected=expected,
    )

    assert source == {}
    assert output[next(iter(expected))] is rope


@pytest.mark.parametrize(
    "target_key",
    [
        "unknown.weight",
        "icl_block.layers.0.attn.query_transform.7.weight",
        "icl_block.layers.0.attn.key_transform.0.bias",
    ],
)
def test_source_rejects_unknown_layouts(target_key: str) -> None:
    with pytest.raises((KeyError, ValueError)):
        checkpoint._source(target_key)


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
