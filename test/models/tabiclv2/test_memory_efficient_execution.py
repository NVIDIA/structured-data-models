from typing import Any

import pytest
import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabiclv2 import row_embedding as row_embedding_module
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.testing import withCUDA


def _row_embedding(
    device: torch.device | None = None,
    *,
    num_classes: int = 10,
    num_layers: int = 2,
) -> RowEmbedding:
    device = torch.device("cpu") if device is None else device
    model = RowEmbedding(
        num_classes=num_classes,
        channels=8,
        num_layers=num_layers,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
        device=device,
    ).eval()
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
    return model


@pytest.mark.parametrize("num_classes", [0, 10, 25])
@pytest.mark.parametrize(("fit_rows", "query_rows"), [(3, 5), (7, 3)])
@withCUDA
def test_memory_efficient_row_embedding_matches_standard(
    device: torch.device,
    num_classes: int,
    fit_rows: int,
    query_rows: int,
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
    model = _row_embedding(
        device,
        num_classes=min(num_classes, 10),
    )
    x = torch.randn(3, 12, 7, device=device)
    if num_classes == 0:
        y = torch.randn(3, 7, device=device)
        model_num_classes = None
    else:
        y = torch.randint(num_classes, (3, 7), device=device)
        model_num_classes = num_classes

    standard_cache = Cache()
    memory_cache = Cache()
    with torch.inference_mode():
        expected = model(x, y, num_classes=model_num_classes)
        actual = model(
            x,
            y,
            num_classes=model_num_classes,
            memory_efficient=True,
        )
        model(
            x[:, :fit_rows],
            y[:, :fit_rows],
            num_classes=model_num_classes,
            cache=standard_cache,
        )
        model(
            x[:, :fit_rows],
            y[:, :fit_rows],
            num_classes=model_num_classes,
            cache=memory_cache,
            memory_efficient=True,
        )
        expected_replay = model(
            x[:, :query_rows],
            y[:, :0],
            num_classes=model_num_classes,
            cache=standard_cache.freeze(),
        )
        actual_replay = model(
            x[:, :query_rows],
            y[:, :0],
            num_classes=model_num_classes,
            cache=memory_cache.freeze(),
            memory_efficient=True,
        )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        actual_replay,
        expected_replay,
        atol=2e-5,
        rtol=2e-5,
    )
    assert memory_cache.keys() == standard_cache.keys()
    for key in standard_cache:
        expected_entry = standard_cache[key]
        actual_entry = memory_cache[key]
        assert isinstance(expected_entry, KVCacheEntry)
        assert isinstance(actual_entry, KVCacheEntry)
        torch.testing.assert_close(
            actual_entry.key,
            expected_entry.key,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            actual_entry.value,
            expected_entry.value,
            atol=2e-5,
            rtol=2e-5,
        )


def test_memory_efficient_row_embedding_uses_2048_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _row_embedding(num_layers=1)
    original = RowEmbedding._memory_efficient_forward
    scheduled_rows: list[int] = []

    def record_schedule(
        self: RowEmbedding,
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        scheduled_rows.append(x.size(-2))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(
        RowEmbedding,
        "_memory_efficient_forward",
        record_schedule,
    )
    y = torch.randint(10, (2,))
    with torch.inference_mode():
        model(torch.randn(2048, 2), y, memory_efficient=True)
        assert scheduled_rows == []
        model(torch.randn(2049, 2), y, memory_efficient=True)

    assert scheduled_rows == [2049]


def test_memory_efficient_row_embedding_rejects_unsupported_active_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        row_embedding_module,
        "_MEMORY_EFFICIENT_ROW_CHUNK_SIZE",
        4,
    )
    model = _row_embedding(num_layers=1)
    x = torch.randn(5, 4)
    y = torch.randint(10, (2,))

    model.train()
    with (
        torch.inference_mode(),
        pytest.raises(
            RuntimeError,
            match="requires evaluation mode",
        ),
    ):
        model(x, y, memory_efficient=True)

    model.eval()
    with pytest.raises(RuntimeError, match="gradients to be disabled"):
        model(x, y, memory_efficient=True)

    with monkeypatch.context() as compiling:
        compiling.setattr(torch.compiler, "is_compiling", lambda: True)
        with (
            torch.inference_mode(),
            pytest.raises(
                RuntimeError,
                match=r"does not support torch\.compile",
            ),
        ):
            model(x, y, memory_efficient=True)

    with (
        torch.inference_mode(),
        pytest.raises(
            ValueError,
            match="contiguous prefix",
        ),
    ):
        model(
            x,
            y,
            train_mask=torch.tensor([True, False, True, False, False]),
            memory_efficient=True,
        )
    with (
        torch.inference_mode(),
        pytest.raises(
            ValueError,
            match="does not support 'max_keys'",
        ),
    ):
        model(x, y, max_keys=1, memory_efficient=True)


def test_memory_efficient_row_embedding_bounds_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordContiguous(TorchDispatchMode):
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def __torch_dispatch__(
            self,
            func: OpOverload,
            types: tuple[type, ...],
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
        ) -> object:
            del types
            kwargs = kwargs or {}
            is_contiguous_clone = (
                func is torch.ops.aten.clone.default
                and kwargs.get("memory_format") == torch.contiguous_format
            )
            if (
                func is torch.ops.aten.contiguous.default
                or is_contiguous_clone
            ):
                tensor = args[0]
                assert isinstance(tensor, torch.Tensor)
                self.shapes.append(tuple(tensor.size()))
            return func(*args, **kwargs)

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
    model = _row_embedding()
    clone = _row_embedding()
    clone.load_state_dict(model.state_dict(), strict=True)
    assert clone.state_dict().keys() == model.state_dict().keys()
    assert not any("memory_efficient" in key for key in model.state_dict())
    x = torch.randn(3, 12, 7)
    y = torch.randint(10, (3, 7))
    projected_shapes: list[tuple[int, ...]] = []

    def record_projection(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        out: torch.Tensor,
    ) -> None:
        projected_shapes.append(tuple(out.size()))

    handle = model.lin.register_forward_hook(record_projection)
    standard_mode = RecordContiguous()
    memory_mode = RecordContiguous()
    with torch.inference_mode(), standard_mode:
        model(x, y)
    projected_shapes.clear()
    with torch.inference_mode(), memory_mode:
        model(x, y, memory_efficient=True)
    handle.remove()

    full_projection = (3, 7, 12, 8)
    assert full_projection in standard_mode.shapes
    assert full_projection not in memory_mode.shapes
    assert projected_shapes
    assert all(shape[-3] <= 4 or shape[-2] <= 3 for shape in projected_shapes)
