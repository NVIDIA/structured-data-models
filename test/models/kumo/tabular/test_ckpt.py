import os
from pathlib import Path
from typing import cast

import pytest
import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular import ckpt as ckpt_module
from sdm.models.kumo.tabular.ckpt import (
    _EXPECTED_ARGS,
    _insert,
    _map_key,
    _remap_checkpoint,
    _validate_args,
    _validate_state,
    load_checkpoint,
)


def test_args_are_validated_before_model_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = {"args": {**_EXPECTED_ARGS, "channels": 64}, "model": {}}
    load_args: tuple[object, ...] = ()
    load_kwargs: dict[str, object] = {}

    def fake_load(*args: object, **kwargs: object) -> object:
        nonlocal load_args, load_kwargs
        load_args = args
        load_kwargs = kwargs
        return checkpoint

    monkeypatch.setattr(torch, "load", fake_load)

    def fail() -> None:
        pytest.fail("model was allocated before checkpoint validation")

    monkeypatch.setattr(ckpt_module, "_KumoTabular", fail)

    with pytest.raises(ValueError, match=r"changed=\['channels'\]"):
        load_checkpoint("checkpoint.pt")
    assert load_args == ("checkpoint.pt",)
    assert load_kwargs == {"map_location": "cpu", "weights_only": True}


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({**_EXPECTED_ARGS, "channels": True}, "changed"),
        ({**_EXPECTED_ARGS, "channels": torch.ones(2)}, "changed"),
        ({**_EXPECTED_ARGS, 1: "unexpected"}, "unexpected"),
    ],
)
def test_args_reject_hostile_values(
    args: dict[object, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_args(args)


def test_transformer_and_rope_mapping() -> None:
    assert _map_key("icl.blocks.3.q_norm.weight") == [
        "icl_block.layers.3.query_norm.weight"
    ]
    assert _map_key("encoder.stages.5.blocks.0.attn.q_norm.weight") == [
        "table_encoder.row_blocks.2.attn.query_transform.1.weight"
    ]
    assert _map_key("encoder.rope.inv_freq") == [
        f"table_encoder.row_blocks.{stage}.attn.{side}_transform.0.inv_freq"
        for stage in range(4)
        for side in ("query", "key")
    ]


def test_mapping_rejects_unknown_and_colliding_entries() -> None:
    with pytest.raises(KeyError, match="Unsupported checkpoint entry"):
        _map_key("unused.weight")

    with pytest.raises(TypeError, match="string keys to tensors"):
        _remap_checkpoint(
            cast(
                dict[str, torch.Tensor],
                {"y_encoder.lin.weight": "not a tensor"},
            )
        )

    colliding = {
        "encoder.stages.1.blocks.0.q_norm.weight": torch.empty(128),
        "encoder.stages.01.blocks.0.q_norm.weight": torch.empty(128),
    }
    with pytest.raises(ValueError, match="collide"):
        _remap_checkpoint(colliding)

    out = {"weight": torch.tensor(1.0)}
    with pytest.raises(ValueError, match="collide"):
        _insert(out, "weight", torch.tensor(2.0), "second.weight")


@pytest.mark.parametrize("defect", ["key", "shape", "dtype"])
def test_state_validation(defect: str) -> None:
    expected = {"weight": torch.empty(2, dtype=torch.float32)}
    if defect == "key":
        state = {"other": torch.empty(2)}
        match = "state keys"
    elif defect == "shape":
        state = {"weight": torch.empty(3)}
        match = "shape"
    else:
        state = {"weight": torch.empty(2, dtype=torch.float64)}
        match = "dtype"

    with pytest.raises(ValueError, match=match):
        _validate_state(state, expected)


def test_local_checkpoint() -> None:
    path = os.environ.get("KUMO_TABULAR_CHECKPOINT")
    if path is None:
        pytest.skip("KUMO_TABULAR_CHECKPOINT is not set")

    model = load_checkpoint(Path(path)).eval()
    x = (torch.arange(70, dtype=torch.float32).reshape(2, 7, 5) - 17) / 11
    y = torch.tensor([[0, 1, 2, 1], [2, 0, 1, 2]])
    categorical_mask = torch.tensor(
        [
            [False, True, False, True, False],
            [True, False, False, True, True],
        ]
    )

    with torch.inference_mode():
        out = model(x, y, categorical_mask)

    expected = torch.tensor(
        [
            [
                0.4692101777,
                1.4048469067,
                0.7584285736,
                -5.6648416519,
                -9.1197347641,
            ],
            [
                0.7624204755,
                0.7110487819,
                1.0508751869,
                -3.2481987476,
                -7.3946108818,
            ],
        ]
    )
    actual = torch.stack((out[0, 0, :5], out[1, -1, :5]))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        34_188_428
    )
    assert not model.training

    cache = Cache()
    with torch.inference_mode():
        context_out = model(
            x[..., :4, :],
            y,
            categorical_mask,
            cache=cache,
        )
        replayed = model(
            x[..., 4:, :],
            y[..., :0],
            categorical_mask,
            cache=cache.freeze(),
        )
    assert context_out.size(-2) == 0
    torch.testing.assert_close(replayed, out, atol=0, rtol=0)
