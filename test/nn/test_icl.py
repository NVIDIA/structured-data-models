import pytest
import torch
from sdm.nn import ICLBlock
from torch import Tensor


class RecordingLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Size, torch.Size | None]] = []

    def forward(
        self,
        query: Tensor,
        key_value: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        self.calls.append(
            (
                query.shape,
                None if key_value is None else key_value.shape,
            )
        )
        return query


class KeySummaryLayer(torch.nn.Module):
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        assert key_value is not None
        return query + key_value.sum(dim=1, keepdim=True)


def test_icl_classification_shape() -> None:
    block = ICLBlock(
        channels=8,
        num_layers=2,
        num_heads=2,
        max_classes=7,
    )
    x = torch.randn(6, 8)
    y_train = torch.tensor([0, 1, 6, 2])

    out = block(x=x, y_train=y_train)

    assert out.shape == (2, 7)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_icl_regression_shape() -> None:
    block = ICLBlock(
        channels=8,
        num_layers=2,
        num_heads=2,
        feedforward_channels=12,
        qassmax=False,
        norm_bias=False,
        max_quantiles=11,
    )
    x = torch.randn(7, 8)
    y_train = torch.randn(3)

    out = block(x=x, y_train=y_train)

    assert out.shape == (4, 11)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_icl_single_test_row_shape() -> None:
    block = ICLBlock(
        channels=8,
        num_layers=2,
        num_heads=2,
        max_classes=7,
    )
    x = torch.randn(5, 8)
    y_train = torch.tensor([0, 1, 6, 2])

    out = block(x=x, y_train=y_train)

    assert out.shape == (1, 7)


def test_icl_zero_test_rows() -> None:
    block = ICLBlock(
        channels=8,
        num_layers=2,
        num_heads=2,
        max_classes=5,
    )
    x = torch.randn(4, 8)
    y_train = torch.tensor([0, 1, 2, 3])

    out = block(x=x, y_train=y_train)

    assert out.shape == (0, 5)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_icl_test_rows_do_not_enter_attention_context() -> None:
    block = ICLBlock(
        channels=8,
        num_layers=2,
        num_heads=2,
        max_classes=5,
    )
    block.layers = torch.nn.ModuleList([KeySummaryLayer(), KeySummaryLayer()])
    x = torch.randn(6, 8)
    y_train = torch.tensor([0, 1, 2])

    baseline = block(x=x, y_train=y_train)
    changed = x.clone()
    changed[4:] = torch.randn_like(changed[4:]) * 1000
    out = block(x=changed, y_train=y_train)

    torch.testing.assert_close(out[:1], baseline[:1])


def test_icl_final_layer_queries_only_test_rows() -> None:
    first_layer = RecordingLayer()
    final_layer = RecordingLayer()
    block = ICLBlock(
        channels=4,
        num_layers=2,
        num_heads=2,
        max_classes=3,
    )
    block.layers = torch.nn.ModuleList([first_layer, final_layer])
    x = torch.randn(5, 4)
    y_train = torch.tensor([0, 1])

    out = block(x=x, y_train=y_train)

    assert out.shape == (3, 3)
    assert first_layer.calls == [
        (torch.Size([1, 5, 4]), torch.Size([1, 2, 4]))
    ]
    assert final_layer.calls == [
        (torch.Size([1, 3, 4]), torch.Size([1, 2, 4]))
    ]


def test_icl_gradients_flow() -> None:
    block = ICLBlock(
        channels=6,
        num_layers=2,
        num_heads=3,
        max_classes=4,
    )
    x = torch.randn(7, 6, requires_grad=True)
    y_train = torch.tensor([0, 1, 2])

    out = block(x=x, y_train=y_train)
    out.square().mean().backward()

    assert x.grad is not None
    assert x.grad[3:].abs().sum().item() > 0
    assert block.cls_head.weight.grad is not None
    assert block.cls_head.weight.grad.abs().sum().item() > 0


def test_icl_error_cases() -> None:
    block = ICLBlock(
        channels=4,
        num_layers=1,
        num_heads=2,
        max_classes=3,
    )

    with pytest.raises(ValueError, match=r"`x` must have rank 2"):
        block(x=torch.randn(1, 2, 4), y_train=torch.tensor([0]))

    with pytest.raises(ValueError, match=r"`x` must be a floating-point"):
        block(x=torch.ones(3, 4, dtype=torch.long), y_train=torch.tensor([0]))

    with pytest.raises(ValueError, match=r"`x` must have 4 channels"):
        block(x=torch.randn(3, 5), y_train=torch.tensor([0]))

    with pytest.raises(ValueError, match=r"`y_train` must have rank 1"):
        block(x=torch.randn(3, 4), y_train=torch.tensor([[0]]))

    with pytest.raises(ValueError, match=r"at least one train row"):
        block(x=torch.randn(2, 4), y_train=torch.empty(0, dtype=torch.long))

    with pytest.raises(ValueError, match=r"`y_train` length"):
        block(x=torch.randn(2, 4), y_train=torch.tensor([0, 1, 2]))

    with pytest.raises(ValueError, match=r"classification labels"):
        block(x=torch.randn(2, 4), y_train=torch.tensor([-1]))

    with pytest.raises(ValueError, match=r"classification labels"):
        block(x=torch.randn(2, 4), y_train=torch.tensor([3]))


def test_icl_requires_at_least_one_layer() -> None:
    with pytest.raises(ValueError, match=r"`num_layers` must be at least 1"):
        ICLBlock(channels=4, num_layers=0, num_heads=2)

    with pytest.raises(
        ValueError, match=r"`max_quantiles` must be at least 1"
    ):
        ICLBlock(channels=4, max_quantiles=0, num_heads=2)

    with pytest.raises(ValueError, match=r"`max_classes` must be at least 1"):
        ICLBlock(channels=4, max_classes=0, num_heads=2)

    with pytest.raises(
        ValueError, match=r"`feedforward_channels` must be at least 1"
    ):
        ICLBlock(channels=4, feedforward_channels=0, num_heads=2)
