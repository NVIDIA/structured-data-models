import warnings

import pytest
import torch

from sdm import Recipe, TableTensor
from sdm.cache import Cache
from sdm.models import TabICLv2
from sdm.models.tabiclv2 import row_embedding as row_embedding_module
from sdm.models.tabiclv2.model import _TabICLv2
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention, TransformerBlock
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


def _randomize_residual_exits(model: torch.nn.Module) -> None:
    # Transformer residual exits are zero-initialized, which would hide
    # masking bugs since padded rows and columns could not influence the
    # output even without masking. Randomizing them makes the equivalence
    # assertions below sensitive to leaking padding.
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


@withCUDA
def test_row_embedding_automatic_batch_size_limit(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RowEmbedding(
        num_classes=2,
        channels=8,
        num_layers=2,
        num_heads=2,
        group_size=2,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
        device=device,
    ).eval()
    context = torch.randn(2, 4, 4, device=device)
    query = torch.randn(2, 2, 4, device=device)
    y = torch.randint(2, (2, 4), device=device)
    chunk_kwargs = {} if device.type == "cuda" else {"batch_size_limit": 1}

    with torch.inference_mode():
        monkeypatch.setattr(
            row_embedding_module,
            "cuda_attention_memory_limit",
            lambda _device: 1 << 60,
        )
        expected = model(torch.cat([context, query], dim=-2), y)[:, 4:]

        monkeypatch.setattr(
            row_embedding_module,
            "cuda_attention_memory_limit",
            lambda _device: 12 * 6 * 8 * 4,
        )
        actual = model(torch.cat([context, query], dim=-2), y, **chunk_kwargs)[
            :, 4:
        ]
        cache = Cache()
        model(context, y, cache=cache, **chunk_kwargs)
        replayed = model(
            query,
            y[:, :0],
            cache=cache.freeze(),
            **chunk_kwargs,
        )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(replayed, expected)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
# @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])  # TODO Reenable
@pytest.mark.parametrize("batch_shape", [()])
def test_forward(
    device: torch.device,
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
) -> None:
    model = TabICLv2(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "TabICLv2()"
    else:
        assert repr(model) == "TabICLv2(device=cuda:0)"

    R_context, R_query, C = 5, 3, 6

    x_context = torch.randn(*batch_shape, R_context, C, device=device)
    x_query = torch.randn(*batch_shape, R_query, C, device=device)
    if dtype.is_floating_point:
        y_context = torch.randn((*batch_shape, R_context, 1), device=device)
        torch.manual_seed(1)
        out = model(x_context, y_context, x_query)
        assert out.size() == (*batch_shape, R_query, 999)
    else:
        y_context = torch.randint(
            low=0,
            high=10,
            size=(*batch_shape, R_context, 1),
            device=device,
        )
        num_classes = len(y_context.unique())
        torch.manual_seed(1)
        out = model(x_context, y_context, x_query)
        assert out.size() == (*batch_shape, R_query, num_classes)

    assert out.dtype == x_query.dtype
    assert out.device == x_query.device
    assert torch.is_inference(out)

    if len(batch_shape) > 0:
        looped = torch.stack(
            [
                model(x_context[i], y_context[i], x_query[i])
                for i in range(batch_shape[0])
            ],
            dim=0,
        )
        assert out.assert_close(looped)

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


# @pytest.mark.parametrize("batch_shape", [(), (2,)])  # TODO Reenable
@pytest.mark.parametrize("batch_shape", [()])
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
    submodel = model.reg_model if dtype.is_floating_point else model.cls_model
    submodel.compile(fullgraph=True)

    torch.manual_seed(1)
    predicted = model(x_context, y_context, x_query)
    assert predicted.allclose(expected, atol=5e-4, rtol=5e-3)
    assert torch.is_inference(predicted)

    # Chunking is skipped while compiling, so passing `batch_size_limit`
    # must not introduce graph breaks under `fullgraph=True`.
    torch.manual_seed(1)
    predicted = model(x_context, y_context, x_query, batch_size_limit=1)
    assert predicted.allclose(expected, atol=5e-4, rtol=5e-3)

    torch.manual_seed(1)
    model.fit(x_context, y_context)
    predicted = model.predict(x_query)
    assert predicted.allclose(expected, atol=5e-4, rtol=5e-3)
    assert torch.is_inference(predicted)


@withCUDA
def test_batch_size_limit_matches_unchunked(device: torch.device) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_context, R_query, C = 12, 5, 6
    x_context = torch.randn(R_context, C, device=device)
    x_query = torch.randn(R_query, C, device=device)
    y_context = torch.randint(0, 10, (R_context, 1), device=device)

    # A pass-through recipe keeps the comparison deterministic: the default
    # recipe draws from the global generator on every call.
    expected = model(x_context, y_context, x_query, recipe=Recipe())
    out = model(
        x_context,
        y_context,
        x_query,
        recipe=Recipe(),
        batch_size_limit=1,
    )
    torch.testing.assert_close(
        out.numerical,
        expected.numerical,
        atol=1e-3,
        rtol=1e-3,
    )

    # `predict` replays the keyword arguments captured by `fit`, so the
    # memory cap set at fit time also bounds the cached replay.
    model.fit(x_context, y_context, recipe=Recipe(), batch_size_limit=1)
    predicted = model.predict(x_query)
    model.clear()
    torch.testing.assert_close(
        predicted.numerical,
        expected.numerical,
        atol=1e-3,
        rtol=1e-3,
    )


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


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_seqused_train_padding(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_context, R_query, C = 11, 6, 5
    x_context = torch.randn(R_context, C, device=device)
    x_query = torch.randn(R_query, C, device=device)
    if dtype.is_floating_point:
        y_context = torch.randn(R_context, 1, device=device)
    else:
        y_context = torch.randint(0, 10, (R_context, 1), device=device)

    expected = model(x_context, y_context, x_query, recipe=Recipe())

    # Pad in-context rows with junk features and junk-but-valid targets, and
    # pad query rows with junk features. Neither may influence the true rows.
    # Padded targets repeat real ones so that padding cannot widen the class
    # set (which would change the number of output columns).
    x_context_padded = torch.cat(
        [x_context, torch.full((5, C), 123.0, device=device)]
    )
    y_context_padded = torch.cat([y_context, y_context[:5]])
    x_query_padded = torch.cat(
        [x_query, torch.full((3, C), -7.0, device=device)]
    )

    out = model(
        x_context_padded,
        y_context_padded,
        x_query_padded,
        recipe=Recipe(),
        seqused_train=torch.tensor(
            R_context,
            dtype=torch.int32,
            device=device,
        ),
    )

    assert out.size(-2) == R_query + 3
    # Masked and unmasked attention select different kernels, so allow
    # kernel-switch-scale noise; junk leaking through a masking bug would
    # show up as O(0.1) differences or NaN.
    torch.testing.assert_close(
        out.numerical[..., :R_query, :],
        expected.numerical,
        atol=1e-3,
        rtol=1e-3,
    )


@withCUDA
def test_seqused_train_batched(device: torch.device) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    lengths = [4, 6, 8]
    R_context, R_query, C = 8, 4, 5
    x_context = torch.randn(len(lengths), R_context, C, device=device)
    x_query = torch.randn(len(lengths), R_query, C, device=device)
    # Every valid prefix holds the same classes, so the batched run and the
    # per-element runs produce matching output columns.
    y_context = (
        torch.arange(2, device=device)
        .repeat(len(lengths), R_context // 2)
        .view(len(lengths), R_context, 1)
    )
    seqused_train = torch.tensor(lengths, dtype=torch.int32, device=device)

    out = model(
        x_context,
        y_context,
        x_query,
        recipe=Recipe(),
        seqused_train=seqused_train,
    )

    # Each batch element must match its individually unpadded forward pass.
    for i, length in enumerate(lengths):
        expected = model(
            x_context[i, :length],
            y_context[i, :length],
            x_query[i],
            recipe=Recipe(),
        )
        torch.testing.assert_close(
            out.numerical[:, i],
            expected.numerical,
            atol=1e-3,
            rtol=1e-3,
        )


@withCUDA
def test_seqused_train_fit_predict(device: torch.device) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_context, R_query, C = 11, 6, 5
    x_context = torch.randn(R_context, C, device=device)
    x_query = torch.randn(R_query, C, device=device)
    y_context = torch.randint(0, 10, (R_context, 1), device=device)

    model.fit(x_context, y_context, recipe=Recipe())
    expected = model.predict(x_query)
    model.clear()

    # A padded fit must cache masked key/value projections, and predict must
    # reuse the stored count automatically.
    x_context_padded = torch.cat(
        [x_context, torch.full((5, C), 123.0, device=device)]
    )
    y_context_padded = torch.cat([y_context, y_context[:5]])
    model.fit(
        x_context_padded,
        y_context_padded,
        recipe=Recipe(),
        seqused_train=torch.tensor(
            R_context,
            dtype=torch.int32,
            device=device,
        ),
    )
    out = model.predict(x_query)
    model.clear()

    torch.testing.assert_close(
        out.numerical,
        expected.numerical,
        atol=1e-3,
        rtol=1e-3,
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_seqused_cols_padding(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_context, R_query, C = 11, 6, 5
    x_context = torch.randn(R_context, C, device=device)
    x_query = torch.randn(R_query, C, device=device)
    if dtype.is_floating_point:
        y_context = torch.randn(R_context, 1, device=device)
    else:
        y_context = torch.randint(0, 10, (R_context, 1), device=device)

    expected = model(x_context, y_context, x_query, recipe=Recipe())

    # Junk-valued padded columns must not influence any prediction: they are
    # excluded from feature grouping and masked from row-wise attention.
    padding = torch.full((R_context, 3), 55.0, device=device)
    out = model(
        torch.cat([x_context, padding], dim=-1),
        y_context,
        torch.cat([x_query, padding[:R_query]], dim=-1),
        recipe=Recipe(),
        seqused_cols=torch.tensor(C, dtype=torch.int32, device=device),
    )

    torch.testing.assert_close(
        out.numerical,
        expected.numerical,
        atol=1e-3,
        rtol=1e-3,
    )


@withCUDA
def test_seqused_rows_and_cols_padding(device: torch.device) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_context, R_query, C = 11, 6, 5
    x_context = torch.randn(R_context, C, device=device)
    x_query = torch.randn(R_query, C, device=device)
    y_context = torch.randint(0, 10, (R_context, 1), device=device)

    expected = model(x_context, y_context, x_query, recipe=Recipe())

    # Combined bucketing: pad columns, in-context rows, and query rows at
    # once, exactly as a bucketed serving recipe would.
    x_context_padded = torch.cat(
        [
            torch.cat(
                [x_context, torch.full((R_context, 3), 55.0, device=device)],
                dim=-1,
            ),
            torch.full((5, C + 3), 123.0, device=device),
        ]
    )
    x_query_padded = torch.cat(
        [
            torch.cat(
                [x_query, torch.full((R_query, 3), 55.0, device=device)],
                dim=-1,
            ),
            torch.full((3, C + 3), -7.0, device=device),
        ]
    )
    y_context_padded = torch.cat([y_context, y_context[:5]])

    out = model(
        x_context_padded,
        y_context_padded,
        x_query_padded,
        recipe=Recipe(),
        seqused_train=torch.tensor(
            R_context,
            dtype=torch.int32,
            device=device,
        ),
        seqused_cols=torch.tensor(C, dtype=torch.int32, device=device),
    )

    torch.testing.assert_close(
        out.numerical[..., :R_query, :],
        expected.numerical,
        atol=1e-3,
        rtol=1e-3,
    )


@withCUDA
def test_seqused_cols_fit_predict(device: torch.device) -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_context, R_query, C = 11, 6, 5
    x_context = torch.randn(R_context, C, device=device)
    x_query = torch.randn(R_query, C, device=device)
    y_context = torch.randint(0, 10, (R_context, 1), device=device)

    model.fit(x_context, y_context, recipe=Recipe())
    expected = model.predict(x_query)
    model.clear()

    # Predict must reuse the stored column count; query rows are padded to
    # the same column width as the fitted rows.
    padding = torch.full((R_context, 3), 55.0, device=device)
    model.fit(
        torch.cat([x_context, padding], dim=-1),
        y_context,
        recipe=Recipe(),
        seqused_cols=torch.tensor(C, dtype=torch.int32, device=device),
    )
    out = model.predict(torch.cat([x_query, padding[:R_query]], dim=-1))
    model.clear()

    torch.testing.assert_close(
        out.numerical,
        expected.numerical,
        atol=1e-3,
        rtol=1e-3,
    )


@withCUDA
def test_seqused_train_compile(device: torch.device) -> None:
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)
    _randomize_residual_exits(model)

    R_context, R_query, C = 11, 6, 5
    x_context = torch.randn(R_context + 5, C + 3, device=device)
    x_query = torch.randn(R_query, C + 3, device=device)
    y_context = torch.randint(0, 10, (R_context + 5, 1), device=device)
    seqused_train = torch.tensor(
        R_context,
        dtype=torch.int32,
        device=device,
    )
    seqused_cols = torch.tensor(C, dtype=torch.int32, device=device)

    expected = model(
        x_context,
        y_context,
        x_query,
        recipe=Recipe(),
        seqused_train=seqused_train,
        seqused_cols=seqused_cols,
    )

    # The padded route must compile without graph breaks. Counts are passed
    # as tensors, so the graph treats them as data rather than specializing.
    model.cls_model.compile(fullgraph=True, backend="eager")
    out = model(
        x_context,
        y_context,
        x_query,
        recipe=Recipe(),
        seqused_train=seqused_train,
        seqused_cols=seqused_cols,
    )

    torch.testing.assert_close(out.numerical, expected.numerical)


def test_seqused_train_validates() -> None:
    model = TabICLv2(pretrained=False)
    x_context = torch.randn(6, 5)
    x_query = torch.randn(2, 5)
    y_context = torch.randint(0, 10, (6, 1))

    with pytest.raises(ValueError, match=r"torch\.int32"):
        model(
            x_context,
            y_context,
            x_query,
            recipe=Recipe(),
            seqused_train=torch.tensor(6),
        )
    with pytest.raises(ValueError, match=r"torch\.int32"):
        model.fit(
            x_context,
            y_context,
            recipe=Recipe(),
            seqused_train=torch.tensor(6),
        )


def test_seqused_cols_validates() -> None:
    model = TabICLv2(pretrained=False)
    x_context = torch.randn(6, 5)
    x_query = torch.randn(2, 5)
    y_context = torch.randint(0, 10, (6, 1))

    with pytest.raises(ValueError, match=r"torch\.int32"):
        model(
            x_context,
            y_context,
            x_query,
            recipe=Recipe(),
            seqused_cols=torch.tensor(5),
        )
    with pytest.raises(ValueError, match="scalar"):
        model(
            x_context,
            y_context,
            x_query,
            recipe=Recipe(),
            seqused_cols=torch.tensor([5], dtype=torch.int32),
        )


def test_seqused_requires_pass_through_recipe() -> None:
    model = TabICLv2(pretrained=False)
    x_context = torch.randn(6, 5)
    x_query = torch.randn(2, 5)
    y_context = torch.randint(0, 10, (6, 1))

    # The default recipe fits its state on the in-context rows, including
    # the padded ones, which would let padding influence predictions.
    with pytest.raises(ValueError, match="pass-through"):
        model(
            x_context,
            y_context,
            x_query,
            seqused_train=torch.tensor(6, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="pass-through"):
        model.fit(
            x_context,
            y_context,
            seqused_cols=torch.tensor(5, dtype=torch.int32),
        )


def test_seqused_train_hierarchical_unsupported() -> None:
    model = TabICLv2(pretrained=False)
    num_classes = 11
    x_context = torch.randn(num_classes, 5)
    x_query = torch.randn(2, 5)
    y_context = torch.arange(num_classes).view(num_classes, 1)

    # Hierarchical nodes re-group the in-context rows by class, so the
    # "only the first `seqused_train` rows are valid" contract no longer
    # describes any node.
    with pytest.raises(ValueError, match="hierarchical"):
        model(
            x_context,
            y_context,
            x_query,
            recipe=Recipe(),
            seqused_train=torch.tensor(
                num_classes,
                dtype=torch.int32,
            ),
        )


def test_seqused_cols_out_of_range_clamped() -> None:
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False)
    _randomize_residual_exits(model)

    x_context = torch.randn(9, 5)
    x_query = torch.randn(3, 5)
    y_context = torch.randint(0, 10, (9, 1))

    # Over-counts clamp to the true width (degenerate but memory-safe).
    full = model(
        x_context,
        y_context,
        x_query,
        recipe=Recipe(),
        seqused_cols=torch.tensor(5, dtype=torch.int32),
    )
    over = model(
        x_context,
        y_context,
        x_query,
        recipe=Recipe(),
        seqused_cols=torch.tensor(7, dtype=torch.int32),
    )
    torch.testing.assert_close(over.numerical, full.numerical)


def test_seqused_cols_table_tensor_warns() -> None:
    model = TabICLv2(pretrained=False)
    table = TableTensor(
        columns={"numerical": ["a", "b", "c", "d"], "datetime": ["t"]},
        numerical=torch.randn(12, 4),
        datetime=torch.arange(12, dtype=torch.int64).view(12, 1),
    )
    y_context = torch.randint(0, 10, (9, 1))

    # `seqused_cols` counts the numerical block; table columns beyond it are
    # dropped before masking, which deserves a warning so callers do not
    # compute the count from the table width. The call also emits the
    # generic ignored-columns warning, so record all and match ours.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model(
            table[:9],
            y_context,
            table[9:],
            recipe=Recipe(),
            seqused_cols=torch.tensor(3, dtype=torch.int32),
        )

    assert any("numerical block" in str(entry.message) for entry in record)


@onlyFullTest
@withCUDA
def test_autocast_compile(device: torch.device) -> None:
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device=device)

    R_context, R_query, C = 5, 3, 6
    x_context = torch.randn(R_context, C, device=device)
    x_query = torch.randn(R_query, C, device=device)
    y_context = torch.randint(0, 10, (R_context, 1), device=device)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with torch.autocast(device_type=device.type, dtype=dtype):
        expected = model(x_context, y_context, x_query, recipe=Recipe())
        model.fit(x_context, y_context, recipe=Recipe())
        expected_predicted = model.predict(x_query)
        model.clear()

    # The serving recipe wraps autocast around the compiled sub-models,
    # including the cache-record and cache-replay routes.
    model.cls_model.compile(fullgraph=True, backend="eager")
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model(x_context, y_context, x_query, recipe=Recipe())
        model.fit(x_context, y_context, recipe=Recipe())
        predicted = model.predict(x_query)
        model.clear()

    torch.testing.assert_close(out.numerical, expected.numerical)
    torch.testing.assert_close(
        predicted.numerical,
        expected_predicted.numerical,
    )
