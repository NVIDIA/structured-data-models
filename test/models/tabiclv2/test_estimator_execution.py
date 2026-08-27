from typing import Any, Literal

import pytest
import torch

import sdm.processing as sp
from sdm.models import TabICLv2
from sdm.models.tabiclv2 import row_embedding as row_embedding_module
from sdm.models.tabiclv2.model import _TabICLv2
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention, TransformerBlock
from sdm.testing import withCUDA


def _tiny_model(
    device: torch.device,
    estimator_execution: Literal[
        "sequential",
        "batched",
        "batched_memory_efficient",
    ],
) -> TabICLv2:
    model = TabICLv2(
        pretrained=False,
        device=device,
        estimator_execution=estimator_execution,
    )
    model.cls_model = _TabICLv2(
        num_classes=10,
        num_quantiles=0,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=2,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        norm_bias=True,
        device=device,
    )
    model.reg_model = _TabICLv2(
        num_classes=0,
        num_quantiles=999,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=2,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        norm_bias=False,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(1)
    for module in model.modules():
        if isinstance(module, Attention):
            torch.nn.init.normal_(
                module.out_lin.weight,
                std=0.02,
                generator=generator,
            )
            torch.nn.init.normal_(
                module.out_lin.bias,
                std=0.02,
                generator=generator,
            )
        elif isinstance(module, TransformerBlock):
            assert isinstance(module.mlp, torch.nn.Sequential)
            linear = module.mlp[-1]
            assert isinstance(linear, torch.nn.Linear)
            torch.nn.init.normal_(
                linear.weight,
                std=0.02,
                generator=generator,
            )
            torch.nn.init.normal_(
                linear.bias,
                std=0.02,
                generator=generator,
            )
    model.eval()
    return model


def test_estimator_execution_defaults_to_sequential() -> None:
    model = TabICLv2(pretrained=False)

    assert model.estimator_execution == "sequential"


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_batched_estimator_execution_matches_sequential(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    sequential = _tiny_model(device, "sequential")
    batched = _tiny_model(device, "batched")
    batched.load_state_dict(sequential.state_dict())

    x_context = torch.randn(7, 5, device=device)
    x_query = torch.randn(3, 5, device=device)
    if dtype.is_floating_point:
        y_context = torch.randn(7, 1, device=device)
        core = batched.reg_model
    else:
        y_context = torch.tensor(
            [[0], [1], [2], [0], [1], [2], [0]],
            device=device,
        )
        core = batched.cls_model

    estimator_batch_sizes: list[int] = []

    def record_estimator_batch(
        _module: torch.nn.Module,
        args: tuple[object, ...],
    ) -> None:
        x = args[0]
        assert isinstance(x, torch.Tensor)
        estimator_batch_sizes.append(x.size(0))

    core.register_forward_pre_hook(record_estimator_batch)
    target = (
        sp.Choice(
            sp.Identity(),
            sp.Standardize(),
            method="round_robin",
        )
        if dtype.is_floating_point
        else sp.ShuffleCategories(method="shift")
    )
    recipe = sp.Recipe(
        features=sp.ShuffleColumns(method="shift"),
        target=target,
    )

    def generator() -> torch.Generator:
        return torch.Generator(device=device).manual_seed(7)

    expected = sequential(
        x_context,
        y_context,
        x_query,
        recipe=recipe,
        num_estimators=3,
        generator=generator(),
    )
    actual = batched(
        x_context,
        y_context,
        x_query,
        recipe=recipe,
        num_estimators=3,
        generator=generator(),
    )
    assert estimator_batch_sizes == [3]
    assert actual.columns == expected.columns
    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        atol=5e-4,
        rtol=5e-3,
    )

    sequential.fit(
        x_context,
        y_context,
        recipe=recipe,
        num_estimators=3,
        generator=generator(),
    )
    batched.fit(
        x_context,
        y_context,
        recipe=recipe,
        num_estimators=3,
        generator=generator(),
    )
    assert estimator_batch_sizes == [3, 3]

    sequential.estimator_execution = "batched"
    batched.estimator_execution = "sequential"
    expected = sequential.predict(x_query)
    actual = batched.predict(x_query)
    assert estimator_batch_sizes == [3, 3, 3]
    assert actual.columns == expected.columns
    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        atol=5e-4,
        rtol=5e-3,
    )


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@withCUDA
def test_memory_efficient_execution_matches_batched(
    device: torch.device,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        row_embedding_module,
        "_MEMORY_EFFICIENT_ROW_CHUNK_SIZE",
        4,
    )
    monkeypatch.setattr(
        row_embedding_module,
        "_MEMORY_EFFICIENT_COLUMN_CHUNK_SIZE",
        3,
    )
    batched = _tiny_model(device, "batched")
    memory_efficient = _tiny_model(device, "batched_memory_efficient")
    memory_efficient.load_state_dict(batched.state_dict())
    x_context = torch.randn(7, 7, device=device)
    x_query = torch.randn(5, 7, device=device)
    if dtype.is_floating_point:
        y_context = torch.randn(7, 1, device=device)
        target = sp.Identity()
    else:
        y_context = torch.tensor(
            [[0], [1], [2], [0], [1], [2], [0]],
            device=device,
        )
        target = sp.ShuffleCategories(method="shift")
    recipe = sp.Recipe(
        features=sp.ShuffleColumns(method="shift"),
        target=target,
    )

    def generator() -> torch.Generator:
        return torch.Generator(device=device).manual_seed(7)

    expected = batched(
        x_context,
        y_context,
        x_query,
        recipe=recipe,
        num_estimators=3,
        generator=generator(),
    )
    actual = memory_efficient(
        x_context,
        y_context,
        x_query,
        recipe=recipe,
        num_estimators=3,
        generator=generator(),
    )
    assert actual.columns == expected.columns
    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        atol=5e-4,
        rtol=5e-3,
    )

    batched.fit(
        x_context,
        y_context,
        recipe=recipe,
        num_estimators=3,
        generator=generator(),
    )
    memory_efficient.fit(
        x_context,
        y_context,
        recipe=recipe,
        num_estimators=3,
        generator=generator(),
    )
    expected = batched.predict(x_query)
    actual = memory_efficient.predict(x_query)
    assert actual.columns == expected.columns
    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        atol=5e-4,
        rtol=5e-3,
    )


@withCUDA
def test_cached_prediction_reuses_fit_execution_policy(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        row_embedding_module,
        "_MEMORY_EFFICIENT_ROW_CHUNK_SIZE",
        4,
    )
    original = RowEmbedding._memory_efficient_forward
    calls: list[RowEmbedding] = []

    def record_memory_efficient(
        self: RowEmbedding,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        RowEmbedding,
        "_memory_efficient_forward",
        record_memory_efficient,
    )
    standard = _tiny_model(device, "batched")
    memory_efficient = _tiny_model(device, "batched_memory_efficient")
    memory_efficient.load_state_dict(standard.state_dict())
    x_context = torch.randn(7, 5, device=device)
    y_context = torch.randn(7, 1, device=device)
    x_query = torch.randn(5, 5, device=device)

    standard.fit(x_context, y_context, recipe=sp.Recipe())
    assert calls == []
    memory_efficient.fit(x_context, y_context, recipe=sp.Recipe())
    assert len(calls) == 1

    standard.estimator_execution = "batched_memory_efficient"
    memory_efficient.estimator_execution = "batched"
    standard.predict(x_query)
    assert len(calls) == 1
    memory_efficient.predict(x_query)
    assert len(calls) == 2


@pytest.mark.parametrize("cached", [False, True])
@withCUDA
def test_memory_efficient_execution_autocast(
    device: torch.device,
    cached: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        row_embedding_module,
        "_MEMORY_EFFICIENT_ROW_CHUNK_SIZE",
        4,
    )
    model = _tiny_model(device, "batched_memory_efficient")
    x_context = torch.randn(7, 5, device=device)
    y_context = torch.randn(7, 1, device=device)
    x_query = torch.randn(5, 5, device=device)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with torch.autocast(device_type=device.type, dtype=dtype):
        if cached:
            model.fit(x_context, y_context, recipe=sp.Recipe())
            out = model.predict(x_query)
        else:
            out = model(
                x_context,
                y_context,
                x_query,
                recipe=sp.Recipe(),
            )

    assert out.numerical.dtype == torch.float32


@pytest.mark.parametrize(
    "estimator_execution",
    ["batched", "batched_memory_efficient"],
)
@withCUDA
def test_batched_estimator_execution_groups_compatible_members(
    device: torch.device,
    estimator_execution: Literal["batched", "batched_memory_efficient"],
) -> None:
    sequential = _tiny_model(device, "sequential")
    model = _tiny_model(device, estimator_execution)
    model.load_state_dict(sequential.state_dict())
    recipe = sp.Recipe(
        features=sp.Choice(
            sp.Identity(),
            sp.SelectColumns(3),
            method="round_robin",
        ),
    )
    x_context = torch.randn(7, 5, device=device)
    y_context = torch.randn(7, 1, device=device)
    x_query = torch.randn(3, 5, device=device)
    estimator_batch_sizes: list[int] = []

    def record_estimator_batch(
        _module: torch.nn.Module,
        args: tuple[object, ...],
    ) -> None:
        x = args[0]
        assert isinstance(x, torch.Tensor)
        estimator_batch_sizes.append(x.size(0))

    model.reg_model.register_forward_pre_hook(record_estimator_batch)

    expected = sequential(
        x_context,
        y_context,
        x_query,
        recipe=recipe,
        num_estimators=4,
    )
    actual = model(
        x_context,
        y_context,
        x_query,
        recipe=recipe,
        num_estimators=4,
    )
    assert estimator_batch_sizes == [2, 2]
    torch.testing.assert_close(actual.numerical, expected.numerical)

    sequential.fit(
        x_context,
        y_context,
        recipe=recipe,
        num_estimators=4,
    )
    model.fit(
        x_context,
        y_context,
        recipe=recipe,
        num_estimators=4,
    )
    assert estimator_batch_sizes == [2, 2, 2, 2]

    expected = sequential.predict(x_query)
    actual = model.predict(x_query)
    assert estimator_batch_sizes == [2, 2, 2, 2, 2, 2]
    torch.testing.assert_close(actual.numerical, expected.numerical)
