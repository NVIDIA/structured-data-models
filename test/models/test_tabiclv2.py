import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import TabICLv2
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.processing import Sequential
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


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


def _row_embedding(num_classes: int = 3) -> RowEmbedding:
    row_embedding = RowEmbedding(
        num_classes=num_classes,
        channels=8,
        num_layers=1,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
    )
    for module in row_embedding.modules():
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.02)
    return row_embedding


def test_row_embedding_train_mask() -> None:
    row_embedding = _row_embedding()
    x = torch.randn(6, 4)
    y = torch.tensor([1, 2])
    train_mask = torch.tensor([False, True, False, False, True, False])
    permutation = torch.cat(
        (train_mask.nonzero().flatten(), (~train_mask).nonzero().flatten())
    )

    out = row_embedding(x, y, train_mask=train_mask)
    permuted_out = row_embedding(x[permutation], y)

    torch.testing.assert_close(out[permutation], permuted_out)


def test_row_embedding_context_fallback_and_limit() -> None:
    row_embedding = _row_embedding()
    x = torch.randn(6, 4)
    context_sizes: list[int] = []

    def capture_context(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        key_value = kwargs["key_value"]
        assert isinstance(key_value, torch.Tensor)
        context_sizes.append(key_value.size(-2))

    handle = row_embedding.col_layers[0].register_forward_pre_hook(
        capture_context,
        with_kwargs=True,
    )
    with pytest.raises(ValueError, match="context is empty"):
        row_embedding(
            x,
            torch.empty(0, dtype=torch.long),
            train_mask=torch.zeros(6, dtype=torch.bool),
        )
    row_embedding(
        x,
        torch.empty(0, dtype=torch.long),
        train_mask=torch.zeros(6, dtype=torch.bool),
        fallback_to_all=True,
    )
    row_embedding(
        x,
        torch.arange(6) % 3,
        train_mask=torch.ones(6, dtype=torch.bool),
        max_train=2,
        generator=torch.Generator().manual_seed(0),
    )
    handle.remove()

    assert context_sizes == [6, 2]


def test_row_embedding_rejects_fallback_cache_recording() -> None:
    with pytest.raises(ValueError, match="fallback context"):
        _row_embedding()(
            torch.randn(3, 2),
            torch.empty(0, dtype=torch.long),
            train_mask=torch.zeros(3, dtype=torch.bool),
            cache=Cache(),
            fallback_to_all=True,
        )


@pytest.mark.parametrize("max_train", [0, True, 1.5])
def test_row_embedding_validates_max_train(max_train: object) -> None:
    with pytest.raises((TypeError, ValueError), match="max_train"):
        _row_embedding()(
            torch.randn(3, 2),
            torch.tensor([0, 1]),
            max_train=max_train,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("num_classes", "y", "wrong_y", "message"),
    [
        (
            3,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0.0, 1.0]),
            "Classification targets",
        ),
        (
            0,
            torch.tensor([0.5, -1.0], dtype=torch.float64),
            torch.tensor([0, 1]),
            "Regression targets",
        ),
    ],
)
def test_row_embedding_target_dtypes(
    num_classes: int,
    y: torch.Tensor,
    wrong_y: torch.Tensor,
    message: str,
) -> None:
    row_embedding = _row_embedding(num_classes)
    x = torch.randn(3, 2)

    assert row_embedding(x, y).size() == (3, 16)
    with pytest.raises(TypeError, match=message):
        row_embedding(x, wrong_y)


def test_row_embedding_limited_context_cache() -> None:
    row_embedding = _row_embedding()
    x_train = torch.randn(5, 4)
    x_test = torch.randn(3, 4)
    y = torch.arange(5) % 3

    joint = row_embedding(
        torch.cat((x_train, x_test)),
        y,
        max_train=3,
        generator=torch.Generator().manual_seed(0),
    )[x_train.size(0) :]

    cache = Cache()
    row_embedding(
        x_train,
        y,
        cache=cache,
        max_train=3,
        generator=torch.Generator().manual_seed(0),
    )
    cache.freeze()
    replay_generator = torch.Generator().manual_seed(1)
    replay_state = replay_generator.get_state().clone()
    cached = row_embedding(
        x_test,
        y[:0],
        cache=cache,
        max_train=1,
        generator=replay_generator,
    )

    torch.testing.assert_close(cached, joint)
    torch.testing.assert_close(replay_generator.get_state(), replay_state)


@pytest.mark.parametrize(
    ("num_classes", "y"),
    [
        (3, torch.tensor([0, 1, 2], dtype=torch.int32)),
        (0, torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)),
    ],
)
def test_icl_block_target_dtype_does_not_mutate_input(
    num_classes: int,
    y: torch.Tensor,
) -> None:
    icl_block = ICLBlock(
        num_classes=num_classes,
        channels=8,
        num_layers=1,
        num_heads=2,
        norm_bias=True,
    )
    x = torch.randn(5, 8, requires_grad=True)
    original = x.detach().clone()

    out = icl_block(x, y)

    assert out.size() == (2, 8)
    assert out.dtype == x.dtype
    torch.testing.assert_close(x.detach(), original)

    if y.is_floating_point():
        y.requires_grad_()
        icl_block(x, y).square().sum().backward()
        assert x.grad is not None
        assert x.grad.isfinite().all()
        assert y.grad is not None
        assert y.grad.isfinite().all()


@pytest.mark.parametrize(
    ("num_classes", "y", "message"),
    [
        (3, torch.tensor([0.0, 1.0]), "Classification targets"),
        (0, torch.tensor([0, 1]), "Regression targets"),
    ],
)
def test_icl_block_rejects_wrong_target_dtype(
    num_classes: int,
    y: torch.Tensor,
    message: str,
) -> None:
    block = ICLBlock(
        num_classes=num_classes,
        channels=8,
        num_layers=1,
        num_heads=2,
        norm_bias=True,
    )
    with pytest.raises(TypeError, match=message):
        block(torch.randn(3, 8), y)
