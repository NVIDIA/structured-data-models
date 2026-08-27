import os
from pathlib import Path
from typing import cast

import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models.kumo.tabular.ckpt import (
    _CLASSIFICATION_ARGS,
    _REGRESSION_ARGS,
    _insert,
    _map_key,
    _remap_checkpoint,
    _validate_args,
    _validate_state,
    load_checkpoint,
)
from sdm.models.kumo.tabular.model import _KumoTabular


def test_checkpoint_args_are_validated_before_state_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = {
        "args": {**_CLASSIFICATION_ARGS, "channels": 64},
        "model": {},
        "step": 1,
        "config": {},
    }
    load_args: tuple[object, ...] = ()
    load_kwargs: dict[str, object] = {}

    def fake_load(*args: object, **kwargs: object) -> object:
        nonlocal load_args, load_kwargs
        load_args = args
        load_kwargs = kwargs
        return checkpoint

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(ValueError, match=r"changed=\['channels'\]"):
        load_checkpoint(
            _KumoTabular(10, 0, device="meta"),
            "checkpoint.pt",
            task="classification",
        )
    assert load_args == ("checkpoint.pt",)
    assert load_kwargs == {"map_location": "cpu", "weights_only": True}


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({**_CLASSIFICATION_ARGS, "channels": True}, "changed"),
        ({**_CLASSIFICATION_ARGS, "channels": torch.ones(2)}, "changed"),
        ({**_CLASSIFICATION_ARGS, 1: "unexpected"}, "unexpected"),
    ],
)
def test_args_reject_hostile_values(
    args: dict[object, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_args(args)


def test_task_and_core_configuration_must_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = {
        "args": _REGRESSION_ARGS,
        "model": {},
        "step": 1,
        "config": {},
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)

    with pytest.raises(ValueError, match="does not match requested task"):
        load_checkpoint(
            _KumoTabular(10, 0, device="meta"),
            "regression.pt",
            task="classification",
        )
    with pytest.raises(ValueError, match="Model configuration"):
        load_checkpoint(
            _KumoTabular(10, 0, device="meta"),
            "regression.pt",
            task="regression",
        )


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"model": {}, "args": {}, "step": 1},
        {
            "model": {},
            "args": {},
            "step": 1,
            "config": {},
            "optimizer": {},
        },
        {"model": {}, "args": {}, "step": 1, "config": {}, 1: None},
    ],
)
def test_checkpoint_schema_is_exact(
    checkpoint: dict[object, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    with pytest.raises(ValueError, match="checkpoint schema"):
        load_checkpoint(
            _KumoTabular(10, 0, device="meta"),
            "checkpoint.pt",
            task="classification",
        )


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
            ),
            task="classification",
        )
    with pytest.raises(ValueError, match="expected cpu"):
        _remap_checkpoint(
            {"y_encoder.lin.weight": torch.empty(10, 128, device="meta")},
            task="classification",
        )
    with pytest.raises(ValueError, match=r"expected torch\.strided"):
        _remap_checkpoint(
            {
                "y_encoder.lin.weight": torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.int64),
                    torch.empty(0),
                    (10, 128),
                )
            },
            task="classification",
        )

    colliding = {
        "encoder.stages.1.blocks.0.q_norm.weight": torch.empty(128),
        "encoder.stages.01.blocks.0.q_norm.weight": torch.empty(128),
    }
    with pytest.raises(ValueError, match="collide"):
        _remap_checkpoint(colliding, task="classification")

    out = {"weight": torch.tensor(1.0)}
    with pytest.raises(ValueError, match="collide"):
        _insert(out, "weight", torch.tensor(2.0), "second.weight")


@pytest.mark.parametrize("defect", ["missing", "shape", "dtype", "value"])
def test_regression_alphas_are_consumed_and_validated(defect: str) -> None:
    alphas = torch.linspace(0.001, 0.999, 999)
    state: dict[str, torch.Tensor] = {"prediction_head.alphas": alphas}
    if defect == "missing":
        state.clear()
        match = "missing"
    elif defect == "shape":
        state["prediction_head.alphas"] = alphas[:-1]
        match = "shape"
    elif defect == "dtype":
        state["prediction_head.alphas"] = alphas.double()
        match = "dtype"
    else:
        state["prediction_head.alphas"] = alphas.roll(1)
        match = "do not match"

    with pytest.raises(ValueError, match=match):
        _remap_checkpoint(state, task="regression")

    assert (
        _remap_checkpoint(
            {"prediction_head.alphas": alphas},
            task="regression",
        )
        == {}
    )
    with pytest.raises(ValueError, match="Classification"):
        _remap_checkpoint(
            {"prediction_head.alphas": alphas},
            task="classification",
        )


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


def test_local_classification_checkpoint() -> None:
    path = os.environ.get("KUMO_TABULAR_CHECKPOINT")
    if path is None:
        pytest.skip("KUMO_TABULAR_CHECKPOINT is not set")

    public_model = KumoTabular(checkpoint_path=Path(path))
    model = public_model.model
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
    assert not public_model.training
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

    context = TableTensor.from_tensor(x[0, :4])
    query = TableTensor.from_tensor(x[0, 4:])
    target = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=y[0, :, None],
            categories=(torch.arange(3),),
        ),
    )
    expected_public = public_model(context, target, query)
    public_model.fit(context, target)
    actual_public = public_model.predict(query)
    assert actual_public.size() == (3, 3)
    torch.testing.assert_close(
        actual_public.numerical,
        expected_public.numerical,
    )
    torch.testing.assert_close(
        actual_public.numerical.sum(dim=-1),
        torch.ones(3),
    )


def test_local_regression_checkpoint() -> None:
    path = os.environ.get("KUMO_TABULAR_REGRESSION_CHECKPOINT")
    if path is None:
        pytest.skip("KUMO_TABULAR_REGRESSION_CHECKPOINT is not set")

    public_model = KumoTabular(
        task="regression",
        checkpoint_path=Path(path),
    )
    model = public_model.model
    x = (torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) - 9) / 7
    y = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    categorical_mask = torch.tensor([[False, True, False, True]])

    with torch.inference_mode():
        out = model(x, y, categorical_mask)

    assert out.size() == (1, 2, 999)
    indices = torch.tensor([0, 249, 499, 749, 998])
    expected_quantiles = torch.tensor(
        [
            [
                -7.4459395409,
                -1.9083330631,
                0.2014052421,
                2.2037322521,
                7.3432683945,
            ],
            [
                -7.5432763100,
                -1.7959007025,
                0.2442419678,
                2.8374941349,
                7.5136070251,
            ],
        ]
    )
    expected_means = torch.tensor([0.0590415597, 0.1823115200])
    torch.testing.assert_close(
        out[0].index_select(-1, indices),
        expected_quantiles,
        atol=1e-5,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        out.mean(dim=-1)[0],
        expected_means,
        atol=1e-5,
        rtol=1e-6,
    )
    assert not public_model.training
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
    assert context_out.size() == (1, 0, 999)
    torch.testing.assert_close(replayed, out, atol=0, rtol=0)

    context = TableTensor.from_tensor(x[0, :4])
    query = TableTensor.from_tensor(x[0, 4:])
    target = TableTensor.from_tensor(y[0, :, None])
    expected_public = public_model(context, target, query)
    public_model.fit(context, target)
    actual_public = public_model.predict(query)
    assert actual_public.size() == (2, 1)
    assert actual_public.numerical.isfinite().all()
    torch.testing.assert_close(
        actual_public.numerical,
        expected_public.numerical,
    )
