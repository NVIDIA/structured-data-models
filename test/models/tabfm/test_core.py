import pytest
import torch
from test.models.tabfm._testing import frozen_google_state_dict
from torch import Tensor

from sdm.models.tabfm.core import _TabFM
from sdm.testing import fullgraph, onlyFullTest, withCUDA

_GOOGLE_OUTPUTS = {
    (True, torch.float32): (
        "-.0311832726 .0325720832 .0963274390 -.0308884792 .0329499505 "
        ".0967883766 -.0308885779 .0329498686 .0967883170 -.0310927909 "
        ".0327384099 .0965696126 -.0312582962 .0324328281 .0961239487 "
        "-.0308796000 .0329651013 .0968098044"
    ),
    (True, torch.bfloat16): (
        "-.0311279297 .0327148438 .0961914062 -.0307617188 .0329589844 "
        ".0966796875 -.0307617188 .0329589844 .0966796875 -.0311279297 "
        ".0327148438 .0961914062 -.03125 .0324707031 .0961914062 "
        "-.0307617188 .0329589844 .0966796875"
    ),
    (False, torch.float32): (
        ".0448102131 .0450483188 .0450480357 .0449030921 .0447199494 "
        ".0450894758"
    ),
    (False, torch.bfloat16): (
        ".0446777344 .0451660156 .0451660156 .044921875 .0446777344 "
        ".0451660156"
    ),
}


def _core(
    is_classifier: bool,
    *,
    row_chunk_size: int | None = 4096,
    col_chunk_size: int | None = 16,
    ffn_chunk_size: int | None = 8192,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> _TabFM:
    return _TabFM(
        embed_dim=2,
        max_classes=3,
        col_num_blocks=1,
        col_num_heads=1,
        col_num_inducing_points=2,
        row_num_blocks=1,
        row_num_heads=1,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_num_heads=2,
        feedforward_factor=2,
        feature_group_size=2,
        num_frequencies=2,
        decoder_hidden_channels=5,
        is_classifier=is_classifier,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        ffn_chunk_size=ffn_chunk_size,
        device=device,
        dtype=dtype,
    )


def _google_state(module: torch.nn.Module) -> dict[str, Tensor]:
    state = frozen_google_state_dict(module)
    for name, tensor in state.items():
        key = name.replace(".rope.inv_freq", ".rope.freqs")
        size = tensor.numel()
        offset = (sum(map(ord, key)) % 11 - 5) * 0.003
        if key == "cls_tokens":
            start, end = -0.1, 0.1
        elif (
            key.endswith(("_ln.weight", "ln_w.weight", "out_ln.weight"))
            or key == "icl_predictor.ln.weight"
        ):
            start, end = 0.8, 1.2
        elif "fourier_frequencies" in key:
            start, end = -0.4, 0.4
        else:
            continue
        values = torch.linspace(start, end, size, device=tensor.device)
        if key != "cls_tokens":
            values.add_(offset)
        state[name] = values.reshape(tensor.shape)
    return state


def _inputs(
    is_classifier: bool,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    features = torch.tensor(
        [
            [
                [0.0, float("nan"), 0.5],
                [-0.75, 1.0, -0.25],
                [0.25, -1.0, 0.75],
            ],
            [
                [1.0, -0.5, 99.0],
                [float("nan"), 0.75, -88.0],
                [-1.25, 0.5, 77.0],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    targets = torch.tensor(
        [[1, -100, 99], [2, 0, -100]]
        if is_classifier
        else [[0.5, -100, 99], [1.25, -0.75, -100]],
        device=device,
        dtype=torch.float32 if is_classifier else torch.float64,
    )
    context_size = torch.tensor([1, 2], device=device)
    categorical_mask = torch.tensor(
        [[False, True, False], [True, False, True]],
        device=device,
    )
    active_features = torch.tensor([3, 2], device=device)
    return (
        features,
        targets,
        context_size,
        categorical_mask,
        active_features,
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_classifier", [True, False])
def test_core_matches_google_golden(
    device: torch.device,
    dtype: torch.dtype,
    is_classifier: bool,
) -> None:
    module = _core(is_classifier, device=device, dtype=dtype)
    module.load_state_dict(_google_state(module), strict=True)

    output = module(*_inputs(is_classifier, device, dtype))

    # Frozen from google-research/tabfm@b8a8b090's PyTorch TabFM.
    expected = output.new_tensor(
        [
            float(value)
            for value in _GOOGLE_OUTPUTS[is_classifier, dtype].split()
        ]
    ).view_as(output)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 2e-3)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("is_classifier", [True, False])
def test_core_isolates_queries_and_padded_features(
    is_classifier: bool,
) -> None:
    module = _core(is_classifier)
    module.load_state_dict(_google_state(module))
    inputs = _inputs(is_classifier)
    features, targets, context_size, categorical_mask, active_features = inputs
    expected = module(*inputs)

    query = torch.arange(3)[None, :] >= context_size[:, None]
    sentinels = (
        torch.tensor([[-100, 99, -7]], dtype=targets.dtype)
        if is_classifier
        else torch.tensor(
            [[float("nan"), float("inf"), -float("inf")]],
            dtype=targets.dtype,
        )
    ).expand_as(targets)
    changed_targets = torch.where(query, sentinels, targets)
    torch.testing.assert_close(
        module(
            features,
            changed_targets,
            context_size,
            categorical_mask,
            active_features,
        ),
        expected,
        rtol=0,
        atol=0,
    )

    changed_features = features.clone()
    changed_features[0, 1].add_(10)
    changed = module(
        changed_features,
        targets,
        context_size,
        categorical_mask,
        active_features,
    )
    torch.testing.assert_close(changed[0, [0, 2]], expected[0, [0, 2]])
    assert not torch.allclose(changed[0, 1], expected[0, 1])

    changed_features = features.clone()
    changed_features[1, :, 2].add_(1000)
    torch.testing.assert_close(
        module(
            changed_features,
            targets,
            context_size,
            categorical_mask,
            active_features,
        ),
        expected,
    )


def test_core_rejects_complex_features() -> None:
    features, targets, context_size, categorical_mask, active_features = (
        _inputs(False)
    )

    with pytest.raises(ValueError, match="complex"):
        _core(False)(
            features.to(torch.complex64),
            targets,
            context_size,
            categorical_mask,
            active_features,
        )


def test_core_uses_google_state_roots_and_zero_cls_tokens() -> None:
    module = _core(True)
    state_keys = set(module.state_dict())
    assert {key.partition(".")[0] for key in state_keys} == {
        "cls_tokens",
        "cell_embedder",
        "col_embedder",
        "col_embedder_2",
        "row_interactor",
        "row_interactor_2",
        "icl_predictor",
    }
    assert torch.count_nonzero(module.cls_tokens) == 0
    assert module.cls_tokens.requires_grad
    assert {key for key in state_keys if ".rope." in key} == {
        "row_interactor.tf_row.rope.inv_freq",
        "row_interactor_2.tf_row.rope.inv_freq",
    }


@onlyFullTest
@pytest.mark.parametrize("is_classifier", [True, False])
def test_core_chunks_compile_and_backpropagate(
    is_classifier: bool,
) -> None:
    unchunked = _core(
        is_classifier,
        row_chunk_size=None,
        col_chunk_size=None,
        ffn_chunk_size=None,
    )
    chunked = _core(
        is_classifier,
        row_chunk_size=2,
        col_chunk_size=2,
        ffn_chunk_size=3,
    )
    state = _google_state(unchunked)
    unchunked.load_state_dict(state)
    chunked.load_state_dict(state)
    features, targets, context_size, categorical_mask, active_features = (
        _inputs(is_classifier)
    )
    features.requires_grad_()
    if not is_classifier:
        targets = targets.clone()
        query = torch.arange(3)[None, :] >= context_size[:, None]
        targets[query] = float("nan")
        targets.requires_grad_()
    with torch.no_grad():
        expected = unchunked(
            features,
            targets,
            context_size,
            categorical_mask,
            active_features,
        )
    output = fullgraph(chunked)(
        features,
        targets,
        context_size,
        categorical_mask,
        active_features,
    )
    torch.testing.assert_close(output, expected)
    assert output.isfinite().all()
    output.square().sum().backward()

    assert features.grad is not None
    assert features.grad.isfinite().all()
    assert all(
        parameter.grad is not None and parameter.grad.isfinite().all()
        for parameter in chunked.parameters()
        if parameter.requires_grad
    )
    if not is_classifier:
        assert targets.grad is not None
        assert targets.grad.isfinite().all()
        assert torch.count_nonzero(targets.grad[query]) == 0
        assert torch.count_nonzero(targets.grad[~query]) > 0
