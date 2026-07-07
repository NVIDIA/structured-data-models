from typing import cast

import pytest
import torch
from sdm import TableTensor
from sdm.cache import Cache
from sdm.models import TabICLv2
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.processing import Sequential
from sdm.testing import withCUDA


def _make_small_row_embedding(
    *,
    num_classes: int,
    num_layers: int = 1,
) -> RowEmbedding:
    return RowEmbedding(
        num_classes=num_classes,
        channels=4,
        num_layers=num_layers,
        num_heads=1,
        group_size=1,
        num_inducing_points=2,
        num_readout_tokens=1,
        norm_bias=True,
    )


class _RecordingColumnLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[torch.Tensor] = []

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        return_key_value: bool = False,
    ) -> torch.Tensor:
        assert not return_key_value
        self.contexts.append(key_value.detach().clone())
        return query


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
        y = torch.randint(0, 10, (*batch_shape, R_train), device=device)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, 10)

    assert out.dtype == x.dtype
    assert out.device == x.device

    if len(batch_shape) > 0:
        looped = torch.stack(
            [model(x[i], y[i]) for i in range(batch_shape[0])]
        )
        torch.testing.assert_close(out, looped)

    model.fit(x[..., :R_train, :], y)
    torch.testing.assert_close(model.predict(x[..., R_train:, :]), out)
    model.clear()


def test_default_recipe_regression_roundtrip() -> None:
    recipe = TabICLv2(pretrained=False).default_recipe()

    features = TableTensor.from_tensor(
        torch.randn(16, 4), columns=["a", "b", "c", "d"]
    )
    target = TableTensor.from_tensor(torch.randn(16, 1), columns=["y"])

    model_features = recipe.features.fit_transform(features.numerical)
    model_target = recipe.target.fit_transform(target.numerical)

    assert model_features.size() == features.size()
    assert model_target.size() == target.size()

    assert isinstance(recipe.target, Sequential)
    restored = recipe.target.inverse_transform(model_target)
    torch.testing.assert_close(
        restored, target.numerical, atol=1e-4, rtol=1e-4
    )


def test_row_embedding_rejects_empty_context_by_default() -> None:
    row_embedding = _make_small_row_embedding(num_classes=0)
    x = torch.randn(2, 3, 4, 3)

    with pytest.raises(ValueError, match="Column-attention context is empty"):
        row_embedding(
            x,
            torch.empty(2, 3, 0),
            train_mask=torch.zeros(4, dtype=torch.bool),
        )


def test_row_embedding_explicit_fallback_uses_all_local_rows() -> None:
    row_embedding = _make_small_row_embedding(num_classes=0)
    recorder = _RecordingColumnLayer()
    row_embedding.col_layers = torch.nn.ModuleList([recorder])
    x = torch.randn(2, 3, 4, 3)

    out = row_embedding(
        x,
        torch.empty(2, 3, 0),
        train_mask=torch.zeros(4, dtype=torch.bool),
        fallback_to_all=True,
    )

    assert recorder.contexts[0].size() == (2, 3, 3, 4, 4)
    assert out.size() == (2, 3, 4, 4)


def test_row_embedding_rejects_fallback_cache_recording() -> None:
    row_embedding = _make_small_row_embedding(num_classes=0)
    cache = Cache()

    with pytest.raises(ValueError, match=r"cache.*fallback"):
        row_embedding(
            torch.randn(4, 3),
            torch.empty(0),
            train_mask=torch.zeros(4, dtype=torch.bool),
            cache=cache,
            fallback_to_all=True,
        )

    assert len(cache) == 0


def test_row_embedding_caps_context_deterministically_per_layer() -> None:
    row_embedding = _make_small_row_embedding(
        num_classes=0,
        num_layers=2,
    )
    with torch.no_grad():
        row_embedding.lin.weight.fill_(1)
        row_embedding.lin.bias.zero_()
        assert isinstance(row_embedding.y_lin, torch.nn.Linear)
        row_embedding.y_lin.weight.zero_()
        row_embedding.y_lin.bias.zero_()

    recorders = [_RecordingColumnLayer(), _RecordingColumnLayer()]
    row_embedding.col_layers = torch.nn.ModuleList(recorders)
    x = torch.arange(8, dtype=torch.float).unsqueeze(-1)
    generator = torch.Generator().manual_seed(123)

    row_embedding(
        x,
        torch.arange(8, dtype=torch.float),
        train_mask=torch.ones(8, dtype=torch.bool),
        max_train=3,
        generator=generator,
    )

    expected_generator = torch.Generator().manual_seed(123)
    expected_indices = [
        torch.randperm(8, generator=expected_generator)[:3] for _ in range(2)
    ]
    for recorder, index in zip(recorders, expected_indices):
        assert recorder.contexts[0].size() == (1, 3, 4)
        torch.testing.assert_close(
            recorder.contexts[0][0, :, 0],
            x[index, 0],
        )


@pytest.mark.parametrize("max_train", [True, 1.5])
def test_row_embedding_rejects_non_integer_max_train(
    max_train: object,
) -> None:
    row_embedding = _make_small_row_embedding(num_classes=0)

    with pytest.raises(TypeError, match="integer or None"):
        row_embedding(
            torch.randn(3, 2),
            torch.randn(2),
            max_train=cast(int, max_train),
        )


@pytest.mark.parametrize("max_train", [0, -1])
def test_row_embedding_rejects_non_positive_max_train(
    max_train: int,
) -> None:
    row_embedding = _make_small_row_embedding(num_classes=0)

    with pytest.raises(ValueError, match="positive or None"):
        row_embedding(torch.randn(3, 2), torch.randn(2), max_train=max_train)


@pytest.mark.parametrize(
    ("train_mask", "error", "message"),
    [
        (torch.ones(3), TypeError, "boolean dtype"),
        (torch.ones(3, 1, dtype=torch.bool), ValueError, "1D tensor"),
        (torch.ones(2, dtype=torch.bool), ValueError, "length 3"),
        (
            torch.tensor([True, False, False]),
            ValueError,
            "select exactly 2 rows",
        ),
    ],
)
def test_row_embedding_validates_train_mask(
    train_mask: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    row_embedding = _make_small_row_embedding(num_classes=0)

    with pytest.raises(error, match=message):
        row_embedding(
            torch.randn(3, 2),
            torch.randn(2),
            train_mask=train_mask,
        )


def test_row_embedding_requires_train_mask_on_x_device() -> None:
    row_embedding = _make_small_row_embedding(num_classes=0)

    with pytest.raises(ValueError, match="same device"):
        row_embedding(
            torch.empty(3, 2, device="meta"),
            torch.randn(2),
            train_mask=torch.tensor([True, True, False]),
        )


def test_row_embedding_rejects_generator_device_mismatch_before_rng() -> None:
    row_embedding = _make_small_row_embedding(num_classes=0).to("meta")
    generator = torch.Generator().manual_seed(123)
    state = generator.get_state().clone()

    with pytest.raises(ValueError, match="same device"):
        row_embedding(
            torch.empty(4, 2, device="meta"),
            torch.empty(4, device="meta"),
            max_train=2,
            generator=generator,
        )

    torch.testing.assert_close(generator.get_state(), state)


def test_row_embedding_target_dtypes() -> None:
    x = torch.randn(2, 5, 3)
    classification = _make_small_row_embedding(num_classes=2)
    y_bool = torch.tensor([[False, True, False], [True, False, True]])
    torch.testing.assert_close(
        classification(x, y_bool),
        classification(x, y_bool.long()),
    )

    with pytest.raises(TypeError, match="Classification targets"):
        classification(x, y_bool.float())
    with pytest.raises(TypeError, match="Classification targets"):
        classification(x, y_bool.to(torch.complex64))
    with pytest.raises(IndexError):
        classification(x, torch.tensor([[0, 1, 2], [1, 0, 1]]))

    regression = _make_small_row_embedding(num_classes=0)
    y_double = torch.randn(2, 3, dtype=torch.float64)
    torch.testing.assert_close(
        regression(x, y_double),
        regression(x, y_double.float()),
    )
    with pytest.raises(TypeError, match="Regression targets"):
        regression(x, y_bool.long())


def test_row_embedding_cache_parity_with_capped_context() -> None:
    row_embedding = _make_small_row_embedding(
        num_classes=0,
        num_layers=2,
    )
    for module in row_embedding.modules():
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.02)

    x_train = torch.randn(2, 5, 3)
    x_test = torch.randn(2, 2, 3)
    y = torch.randn(2, 5)
    expected = row_embedding(
        torch.cat((x_train, x_test), dim=-2),
        y,
        max_train=3,
        generator=torch.Generator().manual_seed(123),
    )[..., -x_test.size(-2) :, :]

    cache = Cache()
    row_embedding(
        x_train,
        y,
        cache=cache,
        max_train=3,
        generator=torch.Generator().manual_seed(123),
    )
    cache.freeze()
    replay_generator = torch.Generator().manual_seed(456)
    replay_state = replay_generator.get_state().clone()
    actual = row_embedding(
        x_test,
        y[..., :0],
        cache=cache,
        max_train=1,
        generator=replay_generator,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(replay_generator.get_state(), replay_state)


@pytest.mark.parametrize(
    ("num_classes", "y", "equivalent_y", "wrong_y", "message"),
    [
        (
            2,
            torch.tensor([[False, True], [True, False]]),
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "Classification targets",
        ),
        (
            0,
            torch.tensor([[0.25, -0.5], [1.5, 2.0]], dtype=torch.float64),
            torch.tensor([[0.25, -0.5], [1.5, 2.0]]),
            torch.tensor([[0, 1], [1, 0]]),
            "Regression targets",
        ),
    ],
)
def test_icl_block_target_dtypes_and_input_immutability(
    num_classes: int,
    y: torch.Tensor,
    equivalent_y: torch.Tensor,
    wrong_y: torch.Tensor,
    message: str,
) -> None:
    block = ICLBlock(
        num_classes=num_classes,
        channels=4,
        num_layers=1,
        num_heads=1,
        norm_bias=True,
    )
    x = torch.randn(2, 4, 4)
    before = x.clone()

    out = block(x, y)

    assert out.size() == (2, 2, 4)
    torch.testing.assert_close(out, block(x, equivalent_y))
    torch.testing.assert_close(x, before)
    with pytest.raises(TypeError, match=message):
        block(x, wrong_y)


def test_icl_block_preserves_gradients_without_mutating_input() -> None:
    block = ICLBlock(
        num_classes=0,
        channels=4,
        num_layers=1,
        num_heads=1,
        norm_bias=True,
    )
    for module in block.modules():
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.02)

    x = torch.randn(2, 4, 4, requires_grad=True)
    y = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    before = x.detach().clone()

    block(x, y).square().sum().backward()

    torch.testing.assert_close(x, before)
    assert x.grad is not None
    assert x.grad.isfinite().all()
    assert y.grad is not None
    assert y.grad.isfinite().all()
