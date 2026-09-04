from pathlib import Path

import pytest
import torch

from sdm.models.kumo.tabular import KumoTabular
from sdm.models.kumo.tabular.checkpoint import (
    _EXPECTED_ARGS_BY_TASK,
    _remap_checkpoint,
    _validate_args,
)
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.tabfm.cell_embedding import FourierNanIndicatorCellEmbedding


def test_checkpoint_requires_one_task() -> None:
    with pytest.raises(ValueError, match="requires exactly one"):
        KumoTabular(checkpoint_path=Path("checkpoint.pt"))


def test_checkpoint_requires_pretrained() -> None:
    with pytest.raises(ValueError, match="pretrained=False"):
        KumoTabular(
            task="classification",
            pretrained=False,
            checkpoint_path=Path("checkpoint.pt"),
        )


def test_checkpoint_arguments_require_consistent_missing_architecture() -> None:
    args = {
        **_EXPECTED_ARGS_BY_TASK["classification"],
        "rope_frac": 0.25,
    }

    with pytest.raises(ValueError, match="rope_frac"):
        _validate_args(args, task="classification")


def test_checkpoint_arguments_support_standard_architecture() -> None:
    args = {
        **_EXPECTED_ARGS_BY_TASK["regression"],
        "cell_embedding": "fourier",
        "row_stage_logn_scale": False,
        "rope_frac": 0.25,
        "icl_num_kv_heads_test": 0,
    }

    architecture = _validate_args(args, task="regression")

    assert architecture.cell_embedding == "fourier"
    assert architecture.row_log_scale is False
    assert architecture.rope_fraction == 0.25


@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_checkpoint_arguments_allow_training_checkpointing(
    grad_checkpoint: bool,
) -> None:
    args = {
        **_EXPECTED_ARGS_BY_TASK["regression"],
        "grad_checkpoint": grad_checkpoint,
    }

    _validate_args(args, task="regression")


def test_checkpoint_remaps_classification_labels_and_rope() -> None:
    weight = torch.arange(30, dtype=torch.float32).view(3, 10)
    bias = torch.arange(3, dtype=torch.float32)
    rope = torch.arange(16, dtype=torch.float32)
    state = {
        "y_encoder.lin.weight": weight,
        "y_encoder.lin.bias": bias,
        "icl_y_encoder.lin.weight": weight,
        "icl_y_encoder.lin.bias": bias,
        "encoder.rope.inv_freq": rope,
    }

    remapped = _remap_checkpoint(state, task="classification")

    torch.testing.assert_close(
        remapped["row_embedding.y_emb.weight"],
        weight.T + bias,
    )
    torch.testing.assert_close(
        remapped["icl_block.y_emb.weight"],
        weight.T + bias,
    )
    rope_keys = [key for key in remapped if key.endswith("inv_freq")]
    assert len(rope_keys) == 8
    assert all(torch.equal(remapped[key], rope) for key in rope_keys)


def test_checkpoint_architecture_uses_missing_embedding_and_full_rope() -> (
    None
):
    checkpoint_model = _KumoTabular(
        num_classes=10,
        num_quantiles=0,
        cell_embedding="fourier_nan_indicator",
        row_log_scale=True,
        rope_fraction=1.0,
    )

    assert isinstance(
        checkpoint_model.row_embedding.cell_embedding,
        FourierNanIndicatorCellEmbedding,
    )
    rope = checkpoint_model.state_dict()[
        "row_embedding.row_blocks.0.attn.query_transform.0.inv_freq"
    ]
    assert rope.numel() == 16
