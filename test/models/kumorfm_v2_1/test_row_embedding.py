import pytest
import torch
from sdm.models.kumorfm_v2_1 import RowEmbedding
from sdm.nn import TransformerBlock
from sdm.testing import withCUDA
from torch import Tensor


def _make_row_embedding(
    *,
    num_layers: int = 2,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> RowEmbedding:
    return RowEmbedding(
        max_classes=4,
        channels=8,
        num_layers=num_layers,
        num_heads=2,
        group_size=3,
        num_inducing_points=3,
        num_readout_tokens=2,
        device=device,
        dtype=dtype,
    )


def _enable_row_context_path(row_embedding: RowEmbedding) -> None:
    final_row_layer = row_embedding.row_layers[-1]
    assert isinstance(final_row_layer, TransformerBlock)
    with torch.no_grad():
        torch.nn.init.eye_(final_row_layer.attn.out_lin.weight)


@pytest.mark.parametrize("regression", [False, True])
def test_row_embedding_label_paths(regression: bool) -> None:
    torch.manual_seed(0)
    row_embedding = _make_row_embedding()
    x = torch.randn(6, 5)
    y = torch.randn(3) if regression else torch.tensor([0, 2, 1])

    out = row_embedding(x, y)

    assert out.size() == (6, 16)
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()


def test_row_embedding_boolean_classification_labels() -> None:
    row_embedding = _make_row_embedding()
    x = torch.randn(4, 3)

    out = row_embedding(x, torch.tensor([False, True]))

    assert out.size() == (4, 16)
    assert torch.isfinite(out).all()


def test_prefix_matches_equivalent_train_mask() -> None:
    torch.manual_seed(1)
    row_embedding = _make_row_embedding()
    x = torch.randn(6, 5)
    y = torch.tensor([0, 2, 1])
    train_mask = torch.tensor([True, True, True, False, False, False])

    expected = row_embedding(x, y)
    actual = row_embedding(x, y, train_mask=train_mask)

    torch.testing.assert_close(actual, expected)


def test_non_prefix_train_mask_matches_row_permutation() -> None:
    torch.manual_seed(2)
    row_embedding = _make_row_embedding(dtype=torch.float64)
    x = torch.randn(6, 5, dtype=torch.float64)
    y = torch.randn(3, dtype=torch.float64)
    train_mask = torch.tensor([False, True, False, True, True, False])
    permutation = torch.cat(
        [
            train_mask.nonzero(as_tuple=True)[0],
            (~train_mask).nonzero(as_tuple=True)[0],
        ]
    )

    actual = row_embedding(x, y, train_mask=train_mask)
    prefix = row_embedding(x.index_select(0, permutation), y)
    expected = prefix.index_select(0, permutation.argsort())

    torch.testing.assert_close(actual, expected)


class _RecordingColumnLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[Tensor] = []

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        self.contexts.append(key_value.detach().clone())
        return query + key_value.mean(dim=-2, keepdim=True)


def test_empty_context_falls_back_to_all_local_rows() -> None:
    row_embedding = _make_row_embedding(num_layers=1)
    recorder = _RecordingColumnLayer()
    row_embedding.col_layers = torch.nn.ModuleList([recorder])
    x = torch.randn(5, 4)
    y = torch.empty(0)
    train_mask = torch.zeros(5, dtype=torch.bool)

    out = row_embedding(x, y, train_mask=train_mask)

    assert recorder.contexts[0].size(-2) == x.size(0)
    assert torch.isfinite(out).all()


def test_max_train_subsampling_is_deterministic_per_generator() -> None:
    torch.manual_seed(3)
    row_embedding = _make_row_embedding()
    _enable_row_context_path(row_embedding)
    recorders = [_RecordingColumnLayer(), _RecordingColumnLayer()]
    row_embedding.col_layers = torch.nn.ModuleList(recorders)
    x = torch.randn(7, 5)
    y = torch.tensor([0, 1, 2, 3, 0])

    def run(seed: int) -> tuple[Tensor, list[Tensor]]:
        for recorder in recorders:
            recorder.contexts.clear()
        generator = torch.Generator().manual_seed(seed)
        out = row_embedding(x, y, max_train=2, generator=generator)
        contexts = [recorder.contexts[0].clone() for recorder in recorders]
        return out, contexts

    out_a, contexts_a = run(11)
    out_b, contexts_b = run(11)
    out_c, contexts_c = run(12)

    torch.testing.assert_close(out_a, out_b)
    for context_a, context_b in zip(contexts_a, contexts_b):
        assert context_a.size(-2) == 2
        torch.testing.assert_close(context_a, context_b)
    assert any(
        not torch.equal(context_a, context_c)
        for context_a, context_c in zip(contexts_a, contexts_c)
    )
    assert not torch.allclose(out_a, out_c)


def test_row_embedding_rejects_malformed_inputs() -> None:
    row_embedding = _make_row_embedding()
    x = torch.randn(5, 4)
    y = torch.tensor([0, 1])

    with pytest.raises(ValueError, match=r"`x` must have shape"):
        row_embedding(x.unsqueeze(0), y)
    with pytest.raises(TypeError, match=r"`x` must be a floating-point"):
        row_embedding(torch.ones(5, 4, dtype=torch.long), y)
    with pytest.raises(ValueError, match="at least one row"):
        row_embedding(torch.empty(0, 4), torch.empty(0))
    with pytest.raises(ValueError, match="at least one column"):
        row_embedding(torch.empty(5, 0), y)
    with pytest.raises(ValueError, match=r"`y_train` must have shape"):
        row_embedding(x, y.unsqueeze(-1))
    with pytest.raises(TypeError, match=r"`y_train` must contain"):
        row_embedding(x, torch.tensor([1 + 2j, 2 + 3j]))
    with pytest.raises(ValueError, match="more labels than rows"):
        row_embedding(x, torch.zeros(6))
    with pytest.raises(ValueError, match="Classification labels"):
        row_embedding(x, torch.tensor([0, 4]))
    with pytest.raises(ValueError, match=r"`train_mask` must have shape"):
        row_embedding(x, y, train_mask=torch.ones(1, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"`train_mask` length"):
        row_embedding(x, y, train_mask=torch.ones(4, dtype=torch.bool))
    with pytest.raises(TypeError, match=r"`train_mask` must have boolean"):
        row_embedding(x, y, train_mask=torch.ones(5, dtype=torch.long))
    with pytest.raises(ValueError, match="label count"):
        row_embedding(x, y, train_mask=torch.ones(5, dtype=torch.bool))

    with pytest.raises(ValueError, match=r"`max_train` must be at least 1"):
        row_embedding(x, y, max_train=0)
    with pytest.raises(TypeError, match=r"`max_train` must be an integer"):
        row_embedding(x, y, max_train=True)


@withCUDA
@pytest.mark.parametrize("regression", [False, True])
def test_row_embedding_dtype_device_gradients_and_input_immutability(
    device: torch.device,
    regression: bool,
) -> None:
    torch.manual_seed(4)
    row_embedding = _make_row_embedding(device=device, dtype=torch.float64)
    _enable_row_context_path(row_embedding)
    x = torch.randn(
        6, 5, device=device, dtype=torch.float64, requires_grad=True
    )
    x_before = x.detach().clone()
    if regression:
        y = torch.randn(
            3,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
    else:
        y = torch.tensor([0, 2, 1], device=device)

    out = row_embedding(x, y)
    loss = (out * torch.randn_like(out)).sum()
    loss.backward()

    assert out.dtype == x.dtype
    assert out.device == x.device
    torch.testing.assert_close(x.detach(), x_before, rtol=0, atol=0)
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
    if regression:
        assert y.grad is not None
        assert row_embedding.y_reg_lin.weight.grad is not None
    else:
        assert row_embedding.y_cls_lin.weight.grad is not None
