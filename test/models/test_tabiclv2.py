import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from sdm.models.tabiclv2 import decode_regression_quantiles
from sdm.models.tabiclv2 import model as tabiclv2_model
from sdm.models.tabiclv2 import output as tabiclv2_output
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.processing import Identity, StandardScale
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
        assert out.size() == (1, *batch_shape, R_query, 999)
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
        assert out.size() == (1, *batch_shape, R_query, num_classes)

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
    assert out.size() == (2, *batch_shape, R_query, 999)

    model.fit(x_context, y_context, num_estimators=3)
    out = model.predict(x_query)
    assert out.size() == (3, *batch_shape, R_query, 999)
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
    out = row_embedding(x, y)
    torch.testing.assert_close(out, row_embedding(x, y_swapped))


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


class _LocalCheckpointProbe(TabICLv2):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.loaded: tuple[Path, bool] | None = None

    def _load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        is_classifier: bool,
    ) -> None:
        self.loaded = (Path(checkpoint_path), is_classifier)


class _StateDictRecorder:
    def __init__(self) -> None:
        self.state_dict_arg: dict[str, torch.Tensor] | None = None
        self.strict: bool | None = None

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        self.state_dict_arg = state_dict
        self.strict = strict


def test_load_regression_checkpoint_uses_only_local_checked_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "regression.ckpt"
    checkpoint.write_bytes(b"local regression checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = _LocalCheckpointProbe()

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "local checkpoint loading must not call Hugging Face"
        )

    monkeypatch.setattr(tabiclv2_model, "hf_hub_download", fail_if_called)

    assert (
        model.load_regression_checkpoint(checkpoint, f"  {digest.upper()}\n")
        is model
    )
    assert model.loaded == (checkpoint, False)


def test_load_regression_checkpoint_rejects_bad_or_missing_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "regression.ckpt"
    checkpoint.write_bytes(b"local regression checkpoint")
    model = _LocalCheckpointProbe()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        model.load_regression_checkpoint(checkpoint, "0" * 64)
    assert model.loaded is None

    with pytest.raises(FileNotFoundError, match="does not exist"):
        model.load_regression_checkpoint(
            tmp_path / "missing.ckpt", "not-a-hash"
        )


@pytest.mark.parametrize(
    "checkpoint_sha256",
    [
        "0" * 63,
        "g" * 64,
        123,
    ],
)
def test_load_regression_checkpoint_rejects_malformed_digest(
    tmp_path: Path,
    checkpoint_sha256: Any,
) -> None:
    checkpoint = tmp_path / "regression.ckpt"
    checkpoint.write_bytes(b"local regression checkpoint")
    model = _LocalCheckpointProbe()

    with pytest.raises(ValueError, match="SHA-256"):
        model.load_regression_checkpoint(checkpoint, checkpoint_sha256)


@pytest.mark.parametrize(
    "payload",
    [{}, {"state_dict": []}, None],
)
def test_checkpoint_loader_rejects_invalid_state_dict_container(
    tmp_path: Path,
    payload: object,
) -> None:
    checkpoint = tmp_path / "regression.ckpt"
    torch.save(payload, checkpoint)

    model = TabICLv2.__new__(TabICLv2)
    torch.nn.Module.__init__(model)
    model.register_parameter("probe", torch.nn.Parameter(torch.ones(())))
    model.reg_model = _StateDictRecorder()

    with pytest.raises(ValueError, match="state_dict"):
        model._load_checkpoint(checkpoint, is_classifier=False)


def test_checkpoint_loader_uses_strict_remapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_state = {"source": torch.ones(1)}
    checkpoint = tmp_path / "regression.ckpt"
    torch.save({"state_dict": raw_state}, checkpoint)

    model = TabICLv2.__new__(TabICLv2)
    torch.nn.Module.__init__(model)
    model.register_parameter("probe", torch.nn.Parameter(torch.ones(())))
    recorder = _StateDictRecorder()
    model.reg_model = recorder
    expected_state = {"target": torch.zeros(1)}
    remap_calls: list[tuple[dict[str, torch.Tensor], bool]] = []

    def remap(
        state_dict: dict[str, torch.Tensor],
        *,
        is_classifier: bool,
    ) -> dict[str, torch.Tensor]:
        remap_calls.append((state_dict, is_classifier))
        return expected_state

    monkeypatch.setattr(tabiclv2_model, "_remap_ckpt", remap)
    model._load_checkpoint(checkpoint, is_classifier=False)

    assert remap_calls[0][0].keys() == raw_state.keys()
    assert remap_calls[0][1] is False
    assert recorder.state_dict_arg is expected_state
    assert recorder.strict is True


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decode_regression_quantiles_repairs_crossing(
    device: torch.device,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantiles = torch.arange(2 * 3 * 999, dtype=dtype, device=device).reshape(
        2, 3, 999
    )
    quantiles[..., 0] = 10_000
    quantiles[..., -1] = -10_000
    expected = quantiles.sort(dim=-1).values.mean(dim=-1)
    original_sort = tabiclv2_output.Tensor.sort
    called = False

    def spy_sort(inp: torch.Tensor, *, dim: int):
        nonlocal called
        called = True
        return original_sort(inp, dim=dim)

    monkeypatch.setattr(tabiclv2_output.Tensor, "sort", spy_sort)
    decoded = decode_regression_quantiles(quantiles, Identity())

    assert called
    assert decoded.shape == (2, 3)
    assert decoded.dtype == dtype
    assert decoded.device == quantiles.device
    torch.testing.assert_close(decoded, expected)


def test_decode_regression_quantiles_applies_target_inverse() -> None:
    target = TableTensor.from_tensor(
        torch.tensor([[1.0], [5.0], [9.0]], dtype=torch.float64),
        columns=["target"],
    )
    target_transform = StandardScale().fit(target)
    quantiles = torch.full((2, 4, 999), 1.5, dtype=torch.float64)

    decoded = decode_regression_quantiles(quantiles, target_transform)
    expected = torch.full((2, 4), 1.5, dtype=torch.float64)
    expected = expected * target_transform.scale + target_transform.mean

    torch.testing.assert_close(decoded, expected)


@pytest.mark.parametrize(
    ("quantiles", "match"),
    [
        pytest.param(torch.ones(2, 3, 998), "shape", id="shape"),
        pytest.param(
            torch.ones(2, 3, 999, dtype=torch.int64),
            "floating point",
            id="dtype",
        ),
    ],
)
def test_decode_regression_quantiles_rejects_invalid_output(
    quantiles: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        decode_regression_quantiles(quantiles, Identity())
