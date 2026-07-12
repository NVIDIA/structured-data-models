import warnings
from contextlib import ExitStack
from unittest import mock

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.cache import KVCacheEntry
from sdm.models import TabICLv2
from sdm.models.tabiclv2.model import _TabICLv2
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention, InducedTransformerBlock, TransformerBlock
from sdm.processing import Recipe, Sequential, SoftmaxTemperature
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


def _small_tabiclv2(device: torch.device) -> TabICLv2:
    model = TabICLv2(pretrained=False, device=device)
    model.cls_model = _TabICLv2(
        num_classes=10,
        num_quantiles=0,
        channels=8,
        num_embedding_layers=2,
        num_embedding_heads=2,
        num_inducing_points=3,
        group_size=3,
        num_readout_tokens=2,
        num_icl_layers=2,
        num_icl_heads=2,
        norm_bias=True,
        device=device,
        dtype=torch.float64,
    )
    return model.eval()


def _randomize_residual_exits(model: torch.nn.Module) -> None:
    # Transformer residual exits are zero-initialized. Randomizing them makes
    # cache equivalence sensitive to the projected keys and values.
    with torch.no_grad():
        for block in model.modules():
            if not isinstance(block, TransformerBlock):
                continue
            block.attn.out_lin.weight.normal_(std=0.05)
            block.attn.out_lin.bias.normal_(std=0.05)
            mlp_out = block.mlp[-1]
            assert isinstance(mlp_out, torch.nn.Linear)
            mlp_out.weight.normal_(std=0.05)
            mlp_out.bias.normal_(std=0.05)


def _assert_chunked(spies: list[mock.Mock], limit: int) -> None:
    for spy in spies:
        assert spy.call_count > 1
        for call in spy.call_args_list:
            query = call.kwargs.get("query")
            if query is None:
                query = call.args[0]
            assert query.shape[:-2].numel() <= limit


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_tabiclv2(
    device: torch.device,
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
) -> None:
    model = TabICLv2(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "TabICLv2()"
    else:
        assert repr(model) == "TabICLv2(device=cuda:0)"

    R, C, R_train = 8, 6, 5

    x = torch.randn(*batch_shape, R, C, device=device)
    if dtype.is_floating_point:
        y = torch.randn(*batch_shape, R_train, device=device)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, 999)
    else:
        # TODO Increase max value once TabICLv2 supports 10+ classes:
        y = torch.randint(0, 10, (*batch_shape, R_train), device=device)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, 10)

    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.is_inference(out)

    if len(batch_shape) > 0:
        looped = torch.stack(
            [model(x[i], y[i]) for i in range(batch_shape[0])]
        )
        torch.testing.assert_close(out, looped)

    model.fit(x[..., :R_train, :], y)
    torch.testing.assert_close(model.predict(x[..., R_train:, :]), out)
    model.clear()


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_tabiclv2_num_estimators(batch_shape: tuple[int, ...]) -> None:
    model = TabICLv2(pretrained=False)

    R, C, R_train = 8, 6, 5
    x = torch.randn(*batch_shape, R, C)
    y = torch.randint(0, 10, (*batch_shape, R_train))

    out = model(x, y)

    # Members are identical for now, so their average matches a single member:
    ensembled = model(x, y, num_estimators=3)
    assert ensembled.size() == out.size()
    torch.testing.assert_close(ensembled, out)

    model.fit(x[..., :R_train, :], y, num_estimators=3)
    torch.testing.assert_close(model.predict(x[..., R_train:, :]), out)
    model.clear()


def test_tabiclv2_recipe() -> None:
    model = TabICLv2(pretrained=False)

    R, C, R_train = 8, 6, 5
    x = TableTensor.from_tensor(torch.randn(R, C))
    y = TableTensor.from_tensor(torch.randn(R_train, 1))

    # An empty recipe matches the recipe-less forward pass:
    torch.testing.assert_close(
        model(x, y, recipe=Recipe()),
        model(x, y),
    )

    # Output steps run after the model and ensembling:
    raw = model(x, y)
    output_recipe = Recipe(output=[SoftmaxTemperature()])
    out = model(x, y, recipe=output_recipe)
    torch.testing.assert_close(out, raw.softmax(dim=-1))
    model.fit(x[:R_train], y, recipe=output_recipe)
    torch.testing.assert_close(model.predict(x[R_train:]), out)

    # The recipe matches its manual driver-side application:
    out = model(x, y, recipe=model.default_recipe())
    assert out.size() == (R - R_train, 999)
    assert torch.is_inference(out)
    recipe = model.default_recipe()
    recipe.features.fit(x[:R_train])
    raw = model(
        x=recipe.features.transform(x),
        y=recipe.target.fit_transform(y),
    )
    assert isinstance(recipe.target, Sequential)
    expected = recipe.target.inverse_transform(
        TableTensor.from_tensor(raw.clone())
    ).numerical
    torch.testing.assert_close(out, expected)

    # The fitted recipe state is reused across predict calls:
    model.fit(x[:R_train], y, recipe=model.default_recipe())
    torch.testing.assert_close(model.predict(x[R_train:]), out)
    model.clear()
    assert model._recipe is None


def test_default_recipe_regression_roundtrip() -> None:
    recipe = TabICLv2.default_recipe()

    features = TableTensor(
        columns={
            "numerical": ("a", "b", "c", "d"),
            "categorical": ("kind",),
        },
        numerical=torch.randn(16, 4),
        categorical=CategoricalTensor(
            data=(torch.arange(16, dtype=torch.int32) % 2).unsqueeze(-1),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )
    target = TableTensor.from_tensor(torch.randn(16, 1), columns=["y"])

    model_features = recipe.features.fit_transform(features)
    model_target = recipe.target.fit_transform(target)

    assert model_features.size() == features.size()
    assert model_target.size() == target.size()
    assert model_features.categorical.size(-1) == 0
    assert set(model_features.columns[Stype.numerical]) == {
        "a",
        "b",
        "c",
        "d",
        "kind",
    }

    assert isinstance(recipe.target, Sequential)
    restored = recipe.target.inverse_transform(model_target)
    torch.testing.assert_close(
        restored.numerical, target.numerical, atol=1e-4, rtol=1e-4
    )


@withCUDA
def test_tabiclv2_batch_size_limit(device: torch.device) -> None:
    model = TabICLv2(pretrained=False, device=device).eval()

    R, C, R_train = 40, 6, 30
    x = torch.randn(R, C, device=device)
    y = torch.randint(0, 10, (R_train,), device=device)

    # A call-time batch_size_limit reaches and chunks the attention blocks.
    blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
    assert blocks

    with ExitStack() as stack:
        spies = [
            stack.enter_context(
                mock.patch.object(block, "_block", wraps=block._block)
            )
            for block in blocks
        ]
        chunked = model(x, y, batch_size_limit=4)
        # Chunking engaged in at least one block.
        assert any(spy.call_count > 0 for spy in spies)

    unchunked = model(x, y)
    torch.testing.assert_close(chunked, unchunked, atol=1e-4, rtol=1e-3)


@withCUDA
def test_tabiclv2_fit_predict_batch_size_limit(
    device: torch.device,
) -> None:
    model = _small_tabiclv2(device)
    _randomize_residual_exits(model.cls_model)

    batch_shape = (2, 3)
    batch_size_limit = 4
    num_train = 5
    num_test = 3
    num_columns = 4
    x_train = torch.randn(
        *batch_shape,
        num_train,
        num_columns,
        device=device,
        dtype=torch.float64,
    )
    x_test = torch.randn(
        *batch_shape,
        num_test,
        num_columns,
        device=device,
        dtype=torch.float64,
    )
    y = torch.randint(0, 10, (*batch_shape, num_train), device=device)

    # Cache-producing sites are transformer_2 in every induced column block
    # and every ICL layer. Their flattened batches are B*C and B,
    # respectively, and both exceed the limit chosen above.
    col_blocks: list[TransformerBlock] = []
    for layer in model.cls_model.row_embedding.col_layers:
        assert isinstance(layer, InducedTransformerBlock)
        col_blocks.append(layer.transformer_2)
    icl_blocks: list[TransformerBlock] = []
    for layer in model.cls_model.icl_block.layers:
        assert isinstance(layer, TransformerBlock)
        icl_blocks.append(layer)
    cache_blocks = [*col_blocks, *icl_blocks]

    expected = model(torch.cat([x_train, x_test], dim=-2), y)

    with ExitStack() as stack:
        fit_spies = [
            stack.enter_context(
                mock.patch.object(block, "_block", wraps=block._block)
            )
            for block in [*col_blocks, *icl_blocks[:-1]]
        ]
        final_cache_spy = stack.enter_context(
            mock.patch.object(
                icl_blocks[-1].attn,
                "_project_key_value",
                wraps=icl_blocks[-1].attn._project_key_value,
            )
        )
        model.fit(x_train, y, batch_size_limit=batch_size_limit)

    _assert_chunked(fit_spies, batch_size_limit)
    # The final fit-time ICL query is empty, but its complete train cache is
    # still projected in bounded native-batch tiles.
    assert final_cache_spy.call_count > 1
    for call in final_cache_spy.call_args_list:
        key_value = call.args[0]
        assert key_value.shape[:-2].numel() <= batch_size_limit
    assert model._caches is not None
    assert len(model._caches) == 1
    cache = model._caches[0]
    assert cache.is_replaying

    cache_snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for i in range(len(col_blocks)):
        key = f"row_embedding.col_layer{i}"
        entry = cache[key]
        assert isinstance(entry, KVCacheEntry)
        expected_shape = (
            *batch_shape,
            num_columns,
            3,  # inducing points
            2,  # heads
            4,  # head channels
        )
        assert entry.key.shape == expected_shape
        assert entry.value.shape == expected_shape
        cache_snapshot[key] = (entry.key.clone(), entry.value.clone())

    for i in range(len(icl_blocks)):
        key = f"icl_block.layer{i}"
        entry = cache[key]
        assert isinstance(entry, KVCacheEntry)
        expected_shape = (
            *batch_shape,
            num_train,
            2,  # heads
            8,  # head channels: 2 readout tokens * 8 channels / 2 heads
        )
        assert entry.key.shape == expected_shape
        assert entry.value.shape == expected_shape
        cache_snapshot[key] = (entry.key.clone(), entry.value.clone())

    with ExitStack() as stack:
        predict_spies = [
            stack.enter_context(
                mock.patch.object(block, "_block", wraps=block._block)
            )
            for block in cache_blocks
        ]
        actual = model.predict(x_test, batch_size_limit=batch_size_limit)

    _assert_chunked(predict_spies, batch_size_limit)
    torch.testing.assert_close(actual, expected)

    # Replay only reads the cache: it retains the complete training context
    # and does not append one copy per prediction chunk.
    assert model._caches is not None
    assert model._caches[0] is cache
    for key, (expected_key, expected_value) in cache_snapshot.items():
        entry = cache[key]
        assert isinstance(entry, KVCacheEntry)
        torch.testing.assert_close(entry.key, expected_key)
        torch.testing.assert_close(entry.value, expected_value)


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
    out = row_embedding(x, y)
    torch.testing.assert_close(out, row_embedding(x, y_swapped))


@onlyCUDA
@onlyFullTest
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_tabiclv2_compile(dtype: torch.dtype) -> None:
    torch._dynamo.reset()
    model = TabICLv2(pretrained=False, device="cuda")

    R, C, R_train = 8, 6, 5
    x = torch.randn(R, C, device="cuda")
    if dtype.is_floating_point:
        y = torch.randn(R_train, device="cuda")
    else:
        y = torch.randint(0, 10, (R_train,), device="cuda")

    expected = model(x, y)
    submodel = model.reg_model if dtype.is_floating_point else model.cls_model
    submodel.compile(fullgraph=True)

    actual = model(x, y)
    torch.testing.assert_close(actual, expected)
    assert torch.is_inference(actual)

    # `batch_size_limit` chunking is skipped while compiling, so passing it
    # must not introduce graph breaks under fullgraph=True.
    actual = model(x, y, batch_size_limit=1)
    torch.testing.assert_close(actual, expected)

    model.fit(x[:R_train], y)
    predicted = model.predict(x[R_train:])
    torch.testing.assert_close(predicted, expected)
    assert torch.is_inference(predicted)


def test_row_embedding_max_keys_seqused_conflict() -> None:
    row_embedding = RowEmbedding(
        num_classes=2,
        channels=8,
        num_layers=1,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
    )
    with pytest.raises(ValueError, match="max_keys"):
        row_embedding(
            x=torch.randn(6, 4),
            y=torch.tensor([0, 1]),
            train_mask=torch.tensor([False, True, False, False, True, False]),
            max_keys=1,
            seqused_train=torch.tensor(2, dtype=torch.int32),
        )


def test_row_embedding() -> None:
    row_embedding = RowEmbedding(
        num_classes=2,
        channels=8,
        num_layers=1,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
    )

    out = row_embedding(
        x=torch.randn(6, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([False, True, False, False, True, False]),
        max_keys=1,
    )
    assert out.size() == (6, 16)


@onlyFullTest
@withCUDA
def test_tabiclv2_fit_predict_compile(device: torch.device) -> None:
    torch.manual_seed(0)
    torch._dynamo.reset()
    model = TabICLv2(pretrained=False, device=device)

    R, C, R_train = 8, 6, 5
    x = torch.randn(R, C, device=device)
    y = torch.randint(0, 10, (R_train,), device=device)

    expected = model(x, y)
    model.fit(x[:R_train], y)
    expected_pred = model.predict(x[R_train:])
    model.clear()

    # The cache-record and cache-replay routes must also compile without
    # graph breaks (each takes its own branch and builds a separate graph).
    model.cls_model.compile(fullgraph=True, backend="eager")
    model.reg_model.compile(fullgraph=True, backend="eager")
    torch.testing.assert_close(model(x, y), expected)
    model.fit(x[:R_train], y)
    torch.testing.assert_close(model.predict(x[R_train:]), expected_pred)


@onlyFullTest
@withCUDA
def test_tabiclv2_autocast_compile(device: torch.device) -> None:
    torch.manual_seed(0)
    torch._dynamo.reset()
    model = TabICLv2(pretrained=False, device=device)

    R, C, R_train = 8, 6, 5
    x = torch.randn(R, C, device=device)
    y = torch.randint(0, 10, (R_train,), device=device)

    with torch.amp.autocast(device.type, torch.bfloat16):
        expected = model(x, y)
        model.fit(x[:R_train], y)
        expected_pred = model.predict(x[R_train:])
        model.clear()

    # The recipe shipped in examples/tabiclv2.py: autocast around compiled
    # submodels, including the cache record/replay routes.
    model.cls_model.compile(fullgraph=True, backend="eager")
    with torch.amp.autocast(device.type, torch.bfloat16):
        torch.testing.assert_close(model(x, y), expected)
        model.fit(x[:R_train], y)
        torch.testing.assert_close(model.predict(x[R_train:]), expected_pred)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_tabiclv2_seqused_train_padding(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    # Zero-initialized residual exits would hide masking bugs (junk rows
    # could not influence outputs even without masking).
    _randomize_residual_exits(model)

    R_train, R_test, C = 11, 6, 5
    x = torch.randn(R_train + R_test, C, device=device)
    if dtype.is_floating_point:
        y = torch.randn(R_train, device=device)
    else:
        y = torch.randint(0, 10, (R_train,), device=device)

    expected = model(x, y)

    # Pad train rows with junk features and junk-but-valid targets, and pad
    # test rows with junk features. Neither may influence the true rows.
    x_padded = torch.cat(
        [
            x[:R_train],
            torch.full((5, C), 123.0, device=device),
            x[R_train:],
            torch.full((3, C), -7.0, device=device),
        ]
    )
    y_padded = torch.cat([y, y.new_zeros(5)])
    seqused_train = torch.tensor(R_train, dtype=torch.int32, device=device)

    out = model(x_padded, y_padded, seqused_train=seqused_train)

    assert out.size(-2) == R_test + 3
    # Masked and unmasked attention select different CUDA kernels, so
    # allow kernel-switch-scale noise (observed max ~5e-4); junk leakage
    # through a masking bug would show as O(0.1) or NaN.
    torch.testing.assert_close(out[:R_test], expected, atol=1e-3, rtol=1e-3)


@withCUDA
def test_tabiclv2_seqused_train_batched(device: torch.device) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_train, R_test, C = 11, 4, 5
    lengths = [7, 9, 11]
    x = torch.randn(len(lengths), R_train + R_test, C, device=device)
    y = torch.randint(0, 10, (len(lengths), R_train), device=device)
    seqused_train = torch.tensor(lengths, dtype=torch.int32, device=device)

    out = model(x, y, seqused_train=seqused_train)

    # Each batch element must match its individually unpadded forward.
    for i, length in enumerate(lengths):
        x_i = torch.cat([x[i, :length], x[i, R_train:]])
        expected = model(x_i, y[i, :length])
        torch.testing.assert_close(out[i], expected, atol=1e-3, rtol=1e-3)


@withCUDA
def test_tabiclv2_seqused_train_fit_predict(device: torch.device) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_train, R_test, C = 11, 6, 5
    x = torch.randn(R_train + R_test, C, device=device)
    y = torch.randint(0, 10, (R_train,), device=device)

    model.fit(x[:R_train], y)
    expected = model.predict(x[R_train:])
    model.clear()

    # Padded fit must cache masked key/value projections, and predict must
    # reuse the stored counts automatically.
    x_padded = torch.cat(
        [x[:R_train], torch.full((5, C), 123.0, device=device)]
    )
    y_padded = torch.cat([y, y.new_zeros(5)])
    model.fit(
        x_padded,
        y_padded,
        seqused_train=torch.tensor(R_train, dtype=torch.int32, device=device),
    )
    out = model.predict(x[R_train:])
    model.clear()

    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


def test_tabiclv2_seqused_train_validates_dtype() -> None:
    model = TabICLv2(pretrained=False)
    x = torch.randn(8, 5)
    y = torch.randint(0, 10, (6,))
    with pytest.raises(ValueError, match=r"torch\.int32"):
        model(x, y, seqused_train=torch.tensor(6))
    with pytest.raises(ValueError, match=r"torch\.int32"):
        model.fit(x[:6], y, seqused_train=torch.tensor(6))


@withCUDA
def test_tabiclv2_seqused_train_compile(device: torch.device) -> None:
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_train, R_test, C = 11, 6, 5
    x = torch.randn(R_train + 5 + R_test, C + 3, device=device)
    y = torch.randint(0, 10, (R_train + 5,), device=device)
    seqused_train = torch.tensor(R_train, dtype=torch.int32, device=device)
    seqused_cols = torch.tensor(C, dtype=torch.int32, device=device)

    expected = model(
        x, y, seqused_train=seqused_train, seqused_cols=seqused_cols
    )

    # The padded route must compile without graph breaks.
    model.cls_model.compile(fullgraph=True, backend="eager")
    out = model(x, y, seqused_train=seqused_train, seqused_cols=seqused_cols)

    torch.testing.assert_close(out, expected)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_tabiclv2_seqused_cols_padding(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_train, R_test, C = 11, 6, 5
    x = torch.randn(R_train + R_test, C, device=device)
    if dtype.is_floating_point:
        y = torch.randn(R_train, device=device)
    else:
        y = torch.randint(0, 10, (R_train,), device=device)

    expected = model(x, y)

    # Junk-valued padded columns must not influence any prediction: they
    # are excluded from feature grouping and masked from row attention.
    x_padded = torch.cat(
        [x, torch.full((R_train + R_test, 3), 55.0, device=device)], dim=-1
    )
    out = model(
        x_padded,
        y,
        seqused_cols=torch.tensor(C, dtype=torch.int32, device=device),
    )

    torch.testing.assert_close(out, expected)


@withCUDA
def test_tabiclv2_seqused_rows_and_cols_padding(
    device: torch.device,
) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_train, R_test, C = 11, 6, 5
    x = torch.randn(R_train + R_test, C, device=device)
    y = torch.randint(0, 10, (R_train,), device=device)

    expected = model(x, y)

    # Combined bucketing: pad columns, train rows, and test rows at once.
    x_padded = torch.cat(
        [x, torch.full((R_train + R_test, 3), 55.0, device=device)], dim=-1
    )
    x_padded = torch.cat(
        [
            x_padded[:R_train],
            torch.full((5, C + 3), 123.0, device=device),
            x_padded[R_train:],
            torch.full((3, C + 3), -7.0, device=device),
        ]
    )
    y_padded = torch.cat([y, y.new_zeros(5)])
    out = model(
        x_padded,
        y_padded,
        seqused_train=torch.tensor(R_train, dtype=torch.int32, device=device),
        seqused_cols=torch.tensor(C, dtype=torch.int32, device=device),
    )

    torch.testing.assert_close(out[:R_test], expected, atol=1e-3, rtol=1e-3)


@withCUDA
def test_tabiclv2_seqused_cols_fit_predict(device: torch.device) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_train, R_test, C = 11, 6, 5
    x = torch.randn(R_train + R_test, C, device=device)
    y = torch.randint(0, 10, (R_train,), device=device)

    model.fit(x[:R_train], y)
    expected = model.predict(x[R_train:])
    model.clear()

    # Predict must reuse the stored column count; test rows are padded to
    # the same column width as the fitted rows.
    x_padded = torch.cat(
        [x, torch.full((R_train + R_test, 3), 55.0, device=device)], dim=-1
    )
    model.fit(
        x_padded[:R_train],
        y,
        seqused_cols=torch.tensor(C, dtype=torch.int32, device=device),
    )
    out = model.predict(x_padded[R_train:])
    model.clear()

    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


def test_tabiclv2_seqused_cols_validates() -> None:
    model = TabICLv2(pretrained=False)
    x = torch.randn(8, 5)
    y = torch.randint(0, 10, (6,))
    with pytest.raises(ValueError, match=r"torch\.int32"):
        model(x, y, seqused_cols=torch.tensor(5))
    with pytest.raises(ValueError, match="scalar"):
        model(
            x,
            y,
            seqused_cols=torch.tensor([5], dtype=torch.int32),
        )


def test_tabiclv2_seqused_cols_out_of_range_clamped() -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False)
    _randomize_residual_exits(model)

    x = torch.randn(12, 5)
    y = torch.randint(0, 10, (9,))

    # Over-counts clamp to the true width (degenerate but memory-safe).
    full = model(x, y, seqused_cols=torch.tensor(5, dtype=torch.int32))
    over = model(x, y, seqused_cols=torch.tensor(7, dtype=torch.int32))
    torch.testing.assert_close(over, full)


def test_tabiclv2_seqused_cols_table_tensor_warns() -> None:
    model = TabICLv2(pretrained=False)
    table = TableTensor(
        columns={"numerical": ["a", "b", "c", "d"], "datetime": ["t"]},
        numerical=torch.randn(12, 4),
        datetime=torch.arange(12, dtype=torch.int64).view(12, 1),
    )
    y = torch.randint(0, 10, (9,))

    # `seqused_cols` counts the numerical block; table columns beyond it
    # are dropped before masking, which deserves a warning so callers do
    # not compute the count from the table width. The call also emits the
    # generic ignored-columns warning, so record all and match ours.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model(
            table,
            y,
            seqused_cols=torch.tensor(3, dtype=torch.int32),
        )
    assert any("numerical block" in str(entry.message) for entry in record)


def test_tabiclv2_cudnn_varlen_toggle_and_degrade() -> None:
    from sdm.nn import _cudnn_varlen, enable_cudnn_varlen

    torch.manual_seed(0)
    model = TabICLv2(pretrained=False)
    _randomize_residual_exits(model)

    R_train, R_test, C = 11, 6, 5
    x = torch.randn(R_train + 5 + R_test, C)
    y = torch.randint(0, 10, (R_train + 5,))
    seqused_train = torch.tensor(R_train, dtype=torch.int32)

    expected = model(x, y, seqused_train=seqused_train)

    # Without the optional dependency (or on CPU) the boolean-mask path
    # keeps serving exactly, and disabling always reports inactive.
    active = enable_cudnn_varlen(True)
    if not _cudnn_varlen.is_available():
        assert not active
    try:
        out = model(x, y, seqused_train=seqused_train)
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
    finally:
        assert enable_cudnn_varlen(False) is False


def test_cudnn_varlen_eligibility_gates() -> None:
    from sdm.nn import _cudnn_varlen

    query = torch.randn(4, 32, 8, 64)
    key = torch.randn(4, 48, 8, 64)

    # A disabled flag short-circuits everything.
    _cudnn_varlen.enable_cudnn_varlen(False)
    assert not _cudnn_varlen.eligible(
        query, key, num_query_heads=8, num_key_value_heads=8
    )

    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        # CPU tensors are never eligible (also covers environments
        # without the optional dependency, where enabling is inert).
        assert not _cudnn_varlen.eligible(
            query, key, num_query_heads=8, num_key_value_heads=8
        )
        if torch.cuda.is_available() and _cudnn_varlen.is_available():
            base = query.cuda().bfloat16()
            base_k = key.cuda().bfloat16()
            # Eligibility mirrors serving: inference (no-grad) context.
            with torch.no_grad():
                assert _cudnn_varlen.eligible(
                    base, base_k, num_query_heads=8, num_key_value_heads=8
                )
                # Grouped-query attention stays on the boolean-mask path.
                assert not _cudnn_varlen.eligible(
                    base, base_k, num_query_heads=8, num_key_value_heads=2
                )
                # Head dims must be multiples of eight (and at most 128).
                assert not _cudnn_varlen.eligible(
                    base[..., :36],
                    base_k[..., :36],
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
            # Gradient-enabled calls stay on the boolean-mask path.
            with torch.enable_grad():
                assert not _cudnn_varlen.eligible(
                    base.clone().requires_grad_(True),
                    base_k,
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
    finally:
        _cudnn_varlen.enable_cudnn_varlen(False)


def test_cudnn_varlen_build_failure_degrades() -> None:
    from sdm.nn import _cudnn_varlen

    # A graph-build failure must degrade to the masked fallback inside
    # the op (probed once, negatively cached), never raise mid-serving.
    class _FailingFrontend:
        class data_type:
            BFLOAT16 = HALF = FLOAT = INT32 = object()

        @staticmethod
        def create_handle() -> object:
            return object()

        @staticmethod
        def pygraph(**kwargs: object) -> object:
            raise RuntimeError("No execution plans support the graph.")

    original = _cudnn_varlen._cudnn_fe
    _cudnn_varlen._cudnn_fe = _FailingFrontend()
    _cudnn_varlen._enabled = True
    try:
        query = torch.randn(2, 16, 8, 64)
        key = torch.randn(2, 32, 8, 64)
        value = torch.randn(2, 32, 8, 64)
        seqused = torch.tensor([20, 32], dtype=torch.int32)
        with pytest.warns(UserWarning, match="masked fallback"):
            out = _cudnn_varlen.cudnn_varlen_sdpa(query, key, value, seqused)
        expected = _cudnn_varlen._masked_fallback(query, key, value, seqused)
        torch.testing.assert_close(out, expected)
        # Negatively cached: the second call neither warns nor rebuilds.
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _cudnn_varlen.cudnn_varlen_sdpa(query, key, value, seqused)
        assert not record
    finally:
        _cudnn_varlen._cudnn_fe = original
        _cudnn_varlen.enable_cudnn_varlen(False)


@withCUDA
def test_tabiclv2_cudnn_varlen_equivalence(device: torch.device) -> None:
    from sdm.nn import _cudnn_varlen, enable_cudnn_varlen

    if device.type != "cuda" or not _cudnn_varlen.is_available():
        pytest.skip("requires CUDA and nvidia-cudnn-frontend")

    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device).to(torch.bfloat16)
    _randomize_residual_exits(model)

    R_train, R_test, C = 96, 32, 8
    x = torch.randn(
        R_train + 32 + R_test, C, device=device, dtype=torch.bfloat16
    )
    y = torch.randint(0, 10, (R_train + 32,), device=device)
    seqused_train = torch.tensor(R_train, dtype=torch.int32, device=device)

    expected = model(x, y, seqused_train=seqused_train)
    enable_cudnn_varlen(True)
    try:
        out = model(x, y, seqused_train=seqused_train)
    finally:
        enable_cudnn_varlen(False)

    # The variable-length kernels differ from the masked kernels, so
    # allow kernel-switch-scale noise (same class as the padding tests).
    torch.testing.assert_close(
        out[:R_test], expected[:R_test], atol=1e-2, rtol=1e-2
    )
