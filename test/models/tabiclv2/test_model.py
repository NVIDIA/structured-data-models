import pytest
import torch

from sdm import Recipe
from sdm.cache import Cache, KVCacheEntry
from sdm.models import TabICLv2
from sdm.models.tabiclv2.model import _TabICLv2
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention, InducedTransformerBlock
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("shape", [(), (2,)])
def test_forward(
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    model = TabICLv2(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "TabICLv2()"
    else:
        assert repr(model) == "TabICLv2(device=cuda:0)"

    R_context, R_query, C = 5, 3, 6

    x_context = torch.randn(*shape, R_context, C, device=device)
    x_query = torch.randn(*shape, R_query, C, device=device)
    if dtype.is_floating_point:
        y_context = torch.randn((*shape, R_context, 1), device=device)
        torch.manual_seed(1)
        out = model(x_context, y_context, x_query)
        assert out.size() == (R_query, 999)
    else:
        y_context = torch.randint(
            low=0,
            high=10,
            size=(R_context, 1),
            device=device,
        ).expand(*shape, -1, -1)
        num_classes = len(y_context.unique())
        torch.manual_seed(1)
        out = model(x_context, y_context, x_query)
        assert out.size() == (R_query, num_classes)

    assert out.dtype == x_query.dtype
    assert out.device == x_query.device
    assert torch.is_inference(out)

    torch.manual_seed(1)
    model.fit(x_context, y_context)
    assert model._cache is not None
    assert model._cache.size() > 0
    assert model.predict(x_query).allclose(out)
    model.clear()


@withCUDA
@pytest.mark.parametrize("cached", [False, True])
def test_autocast_output_is_float32(
    device: torch.device,
    cached: bool,
) -> None:
    model = TabICLv2(pretrained=False, device=device)
    x_context = torch.randn(5, 6, device=device)
    y_context = torch.randn(5, 1, device=device)
    x_query = torch.randn(3, 6, device=device)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with torch.autocast(device_type=device.type, dtype=dtype):
        if cached:
            model.fit(x_context, y_context, recipe=Recipe())
            out = model.predict(x_query)
        else:
            out = model(x_context, y_context, x_query, recipe=Recipe())

    assert out.numerical.dtype == torch.float32


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_num_estimators(batch_shape: tuple[int, ...]) -> None:
    model = TabICLv2(pretrained=False)

    R_context, R_query, C = 5, 3, 6

    x_context = torch.randn(*batch_shape, R_context, C)
    x_query = torch.randn(*batch_shape, R_query, C)
    y_context = torch.randn(*batch_shape, R_context, 1)

    out = model(x_context, y_context, x_query, num_estimators=2)
    assert out.size() == (*batch_shape, R_query, 999)

    model.fit(x_context, y_context, num_estimators=3)
    assert model._cache is not None
    assert 0 in model._cache
    assert 1 in model._cache
    assert model._cache.size() > 0
    assert model._cache.is_cpu

    out = model.predict(x_query)
    assert out.size() == (*batch_shape, R_query, 999)
    model.clear()


@withCUDA
def test_row_embedding_mixed_radix_digit(device: torch.device) -> None:
    row_embedding = RowEmbedding(
        num_classes=10,
        channels=8,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
        device=device,
    )
    for module in row_embedding.modules():
        # Randomly initialize to return non-zero output
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.02)

    x = torch.randn(8, 6, device=device)

    # The labels 5 * a + b and 5 * b + a decompose into the digits (a, b) and
    # (b, a) under bases [5, 5], so averaging over digits must be invariant
    # to swapping them:
    a = torch.tensor([4, 0, 1, 2, 3], device=device)
    b = torch.tensor([4, 1, 2, 3, 0], device=device)
    y = 5 * a + b
    y_swapped = 5 * b + a
    out = row_embedding(x, y, num_classes=25)
    torch.testing.assert_close(
        out,
        row_embedding(x, y_swapped, num_classes=25),
    )


@onlyCUDA
@pytest.mark.parametrize(
    ("dtype", "stabilize", "expected_query_dtype", "expected_autocast"),
    [
        (torch.float16, True, torch.float32, False),
        (torch.float16, False, torch.float16, True),
        (torch.bfloat16, True, torch.bfloat16, True),
    ],
)
def test_row_embedding_float16_recording(
    dtype: torch.dtype,
    stabilize: bool,
    expected_query_dtype: torch.dtype,
    expected_autocast: bool,
) -> None:
    device = torch.device("cuda")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported")
    row_embedding = RowEmbedding(
        num_classes=3,
        channels=8,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
        device=device,
        stabilize_float16_context=stabilize,
    ).eval()
    col_layers: list[InducedTransformerBlock] = []
    for col_layer in row_embedding.col_layers:
        assert isinstance(col_layer, InducedTransformerBlock)
        col_layers.append(col_layer)
    x_context = torch.randn(5, 6, device=device)
    y_context = torch.randint(3, size=(5,), device=device)
    cache = Cache()
    recording_dtypes: list[tuple[bool, torch.dtype]] = []

    def capture_recording_dtype(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        query = kwargs["query"]
        assert isinstance(query, torch.Tensor)
        recording_dtypes.append(
            (
                torch.is_autocast_enabled(query.device.type),
                query.dtype,
            )
        )

    handles = [
        col_layer.output_block.register_forward_pre_hook(
            capture_recording_dtype,
            with_kwargs=True,
        )
        for col_layer in col_layers
    ]
    with torch.amp.autocast(device.type, dtype=dtype):
        context_out = row_embedding(
            x=x_context,
            y=y_context,
            cache=cache,
        )
    for handle in handles:
        handle.remove()

    entries = [
        value for value in cache.values() if isinstance(value, KVCacheEntry)
    ]
    assert recording_dtypes == [
        (expected_autocast, expected_query_dtype)
    ] * len(col_layers)
    assert context_out.dtype == torch.float32
    assert entries
    assert all(entry.key.dtype == dtype for entry in entries)
    assert all(entry.value.dtype == dtype for entry in entries)

    if dtype != torch.float16 or not stabilize:
        return

    replay_dtypes: list[tuple[bool, torch.dtype, torch.dtype]] = []

    def capture_replay_dtype(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        query = kwargs["query"]
        key_value = kwargs["key_value"]
        assert isinstance(query, torch.Tensor)
        assert isinstance(key_value, KVCacheEntry)
        replay_dtypes.append(
            (
                torch.is_autocast_enabled(query.device.type),
                query.dtype,
                key_value.key.dtype,
            )
        )

    cache.freeze()
    handles = [
        col_layer.output_block.register_forward_pre_hook(
            capture_replay_dtype,
            with_kwargs=True,
        )
        for col_layer in col_layers
    ]
    with torch.amp.autocast(device.type, dtype=dtype):
        query_out = row_embedding(
            x=torch.randn(3, 6, device=device),
            y=y_context.new_empty(0),
            cache=cache,
        )
    for handle in handles:
        handle.remove()

    assert replay_dtypes == [(True, torch.float16, torch.float16)] * len(
        col_layers
    )
    assert query_out.isfinite().all()


@onlyCUDA
@pytest.mark.parametrize("recording", [False, True])
def test_tabiclv2_float16_context(recording: bool) -> None:
    device = torch.device("cuda")
    model = _TabICLv2(
        num_classes=3,
        num_quantiles=0,
        channels=8,
        num_embedding_layers=2,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=3,
        num_readout_tokens=2,
        num_icl_layers=2,
        num_icl_heads=2,
        norm_bias=True,
        device=device,
    ).eval()
    col_layers: list[InducedTransformerBlock] = []
    for col_layer in model.row_embedding.col_layers:
        assert isinstance(col_layer, InducedTransformerBlock)
        col_layers.append(col_layer)
    context_dtypes: list[tuple[bool, torch.dtype]] = []

    def capture_context_dtype(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        query = kwargs["query"]
        assert isinstance(query, torch.Tensor)
        context_dtypes.append(
            (
                torch.is_autocast_enabled(query.device.type),
                query.dtype,
            )
        )

    cache = Cache() if recording else None
    handles = [
        col_layer.output_block.register_forward_pre_hook(
            capture_context_dtype,
            with_kwargs=True,
        )
        for col_layer in col_layers
    ]
    with torch.amp.autocast(device.type, dtype=torch.float16):
        model(
            x=torch.randn(5 if recording else 8, 6, device=device),
            y=torch.randint(3, size=(5,), device=device),
            num_classes=3,
            cache=cache,
        )
    for handle in handles:
        handle.remove()

    assert context_dtypes == [(False, torch.float32)] * len(col_layers)
    if cache is None:
        return

    entries = [
        value for value in cache.values() if isinstance(value, KVCacheEntry)
    ]
    assert entries
    assert all(entry.key.dtype == torch.float16 for entry in entries)
    assert all(entry.value.dtype == torch.float16 for entry in entries)


def test_tabiclv2_hierarchical_log_probs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IdentityRowEmbedding(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            *,
            num_classes: int | None = None,
            cache: object | None = None,
        ) -> torch.Tensor:
            return x

    class StubICLBlock(torch.nn.Module):
        temperature = 0.9

        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            *,
            num_classes: int | None = None,
            cache: object | None = None,
        ) -> torch.Tensor:
            assert num_classes == 3
            test_size = x.size(-2) - y.size(-1)
            log_probs = x.new_tensor([0.15, 0.45, 0.4]).log()
            return log_probs.expand(test_size, -1).mul(self.temperature)

    model = _TabICLv2(
        num_classes=2,
        num_quantiles=0,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=3,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        norm_bias=True,
    )
    monkeypatch.setattr(model, "row_embedding", IdentityRowEmbedding())
    monkeypatch.setattr(model, "icl_block", StubICLBlock())

    y = torch.tensor([0, 1, 2])
    out = model(torch.randn(5, 4), y, num_classes=3)

    probabilities = torch.tensor([0.15, 0.45, 0.4]).expand(2, -1)
    expected = probabilities.log().mul(0.9)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_tabiclv2_many_classes_forward_and_cache(
    device: torch.device,
) -> None:
    torch.manual_seed(1)
    model = TabICLv2(pretrained=False, device=device)
    num_classes, test_size = 11, 2
    x_context = torch.randn(num_classes, 6, device=device)
    x_query = torch.randn(test_size, 6, device=device)
    y_context = torch.arange(
        num_classes,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(-1)

    torch.manual_seed(1)
    out = model(x_context, y_context, x_query)

    assert out.size() == (test_size, num_classes)
    probabilities = out.numerical
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(test_size, device=device),
    )

    torch.manual_seed(1)
    model.fit(x_context, y_context)
    assert model.predict(x_query).allclose(out)


@onlyCUDA
@onlyFullTest
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_compile(dtype: torch.dtype) -> None:
    torch._dynamo.reset()
    model = TabICLv2(pretrained=False, device="cuda")

    R_context, R_query, C = 5, 3, 6
    x_context = torch.randn(R_context, C, device="cuda")
    x_query = torch.randn(R_query, C, device="cuda")

    if dtype.is_floating_point:
        y_context = torch.randn(R_context, 1, device="cuda")
    else:
        y_context = torch.randint(0, 10, size=(R_context, 1), device="cuda")

    torch.manual_seed(1)
    expected = model(x_context, y_context, x_query)
    submodel = model.models[
        "regression" if dtype.is_floating_point else "classification"
    ]
    submodel.compile(fullgraph=True)

    torch.manual_seed(1)
    predicted = model(x_context, y_context, x_query)
    assert predicted.allclose(expected, atol=5e-4, rtol=5e-3)
    assert torch.is_inference(predicted)

    torch.manual_seed(1)
    model.fit(x_context, y_context)
    predicted = model.predict(x_query)
    assert predicted.allclose(expected, atol=5e-4, rtol=5e-3)
    assert torch.is_inference(predicted)
