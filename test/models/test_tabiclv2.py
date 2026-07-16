import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.models import TabICLv2
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.processing import Recipe
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


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
        # TODO Increase max value once TabICLv2 supports 10+ classes:
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
    assert model.predict(x_query).allclose(out)
    model.clear()


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
    out = model.predict(x_query)
    assert out.size() == (*batch_shape, R_query, 999)
    model.clear()


def _categorical_target(codes: list[int], vocab: list[str]) -> TableTensor:
    return TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            data=torch.tensor(codes, dtype=torch.int64).view(-1, 1),
            categories=(StringTensor.from_list(vocab),),
        ),
    )


def test_forward_unobserved_categories() -> None:
    model = TabICLv2(pretrained=False)

    R_context, R_query, C = 5, 3, 6
    x_context = torch.randn(R_context, C)
    x_query = torch.randn(R_query, C)
    # A pre-encoded target whose vocabulary holds unobserved categories:
    y_context = _categorical_target(
        codes=[0, 1, 2, 0, 1],
        vocab=["a", "b", "c", "d", "e"],
    )

    out = model(x_context, y_context, x_query, recipe=Recipe())
    assert out.size() == (1, R_query, 5)
    assert out.columns[Stype.numerical] == ("a", "b", "c", "d", "e")

    model.fit(x_context, y_context, recipe=Recipe())
    out = model.predict(x_query)
    assert out.size() == (1, R_query, 5)
    assert out.columns[Stype.numerical] == ("a", "b", "c", "d", "e")
    model.clear()


def test_forward_too_many_classes() -> None:
    model = TabICLv2(pretrained=False)

    R_context, R_query, C = 5, 3, 6
    x_context = torch.randn(R_context, C)
    x_query = torch.randn(R_query, C)
    y_context = _categorical_target(
        codes=[0, 1, 2, 0, 1],
        vocab=[f"c{i}" for i in range(12)],
    )

    with pytest.raises(NotImplementedError, match="at most 10 classes"):
        model(x_context, y_context, x_query, recipe=Recipe())
    with pytest.raises(NotImplementedError, match="at most 10 classes"):
        model.fit(x_context, y_context, recipe=Recipe())


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


def test_row_embedding_num_classes_fallback() -> None:
    row_embedding = RowEmbedding(
        num_classes=10,
        channels=8,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
    )
    for module in row_embedding.modules():
        # Randomly initialize to return non-zero output
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.02)

    x = torch.randn(8, 6)

    # A vocabulary-derived count matches the data-derived fallback:
    y = torch.tensor([0, 1, 2, 0, 1])
    out = row_embedding(x, y)
    torch.testing.assert_close(out, row_embedding(x, y, num_classes=5))

    # Mixed-radix labels above the class capacity:
    y = torch.tensor([14, 0, 21, 7, 3])
    out = row_embedding(x, y)
    torch.testing.assert_close(out, row_embedding(x, y, num_classes=22))


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

    torch.manual_seed(1)
    model.fit(x_context, y_context)
    predicted = model.predict(x_query)
    assert predicted.allclose(expected, atol=5e-4, rtol=5e-3)
    assert torch.is_inference(predicted)


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
