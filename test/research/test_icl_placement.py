"""Numerical tests for research placement adapters."""

import copy

import pytest
import torch
from research.multigpu.placement import (
    ICLPlacement,
    cache_bytes_by_device,
    install_icl_placement,
)

from sdm import TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models.ensemble_parallel import EnsembleParallel
from sdm.models.kumo.tabular.icl import ICLBlock as KumoICL
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.models.tabiclv2.icl import ICLBlock as RelationalICL


def make_block(family: str, classes: int = 3):
    kwargs = {
        "num_classes": classes,
        "out_channels": classes or 5,
        "channels": 16,
        "num_layers": 4,
        "num_heads": 2,
    }
    block = (
        KumoICL(**kwargs)
        if family == "tabular"
        else RelationalICL(**kwargs, norm_bias=True)
    )
    # Kumo residual projections initialize to zero; randomizing makes this
    # sensitive to lost layer outputs and misplaced cache projections.
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-0.2, 0.2)
    return block.eval()


@pytest.mark.parametrize("family", ["tabular", "relational"])
@pytest.mark.parametrize("mode", ["stage", "layers"])
@pytest.mark.parametrize("classes", [0, 3])
@pytest.mark.parametrize("gpu", [False, True])
def test_one_shot_and_cached_parity(family, mode, classes, gpu):
    if gpu and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    devices = ["cuda:0", "cuda:1"] if gpu else ["cpu", "cpu"]
    baseline = make_block(family, classes).to(devices[0])
    placed = ICLPlacement(copy.deepcopy(baseline), devices, mode)
    x = torch.randn(2, 15, 16, device=devices[0])
    y = (
        torch.randint(classes, (2, 9), device=devices[0])
        if classes
        else torch.randn(2, 9, device=devices[0])
    )
    with torch.inference_mode():
        expected = baseline(x.clone(), y)
        actual = placed(x.clone(), y)
        torch.testing.assert_close(actual, expected)
        cache_a, cache_b = Cache(), Cache()
        baseline(x[:, :9].clone(), y, cache=cache_a)
        placed(x[:, :9].clone(), y, cache=cache_b)
        cache_a.freeze()
        cache_b.freeze()
        expected = baseline(x[:, 9:].clone(), y[:, :0], cache=cache_a)
        actual = placed(x[:, 9:].clone(), y[:, :0], cache=cache_b)
        torch.testing.assert_close(actual, expected)
        assert actual.device == x.device
        assert cache_a.size() == cache_b.size()
        if gpu:
            assert str(torch.device(devices[-1])) in cache_bytes_by_device(
                cache_b
            )


@pytest.mark.parametrize("mode", ["stage", "layers"])
@pytest.mark.parametrize("gpu", [False, True])
def test_hierarchical_relational_parity(mode, gpu):
    if gpu and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    devices = ["cuda:0", "cuda:1"] if gpu else ["cpu", "cpu"]
    baseline = make_block("relational", 3).to(devices[0])
    placed = ICLPlacement(copy.deepcopy(baseline), devices, mode)
    x = torch.randn(15, 16, device=devices[0])
    y = torch.arange(9, device=devices[0]) % 7
    with torch.inference_mode():
        a, b = Cache(), Cache()
        expected = baseline(x.clone(), y, num_classes=7, cache=a)
        actual = placed(x.clone(), y, num_classes=7, cache=b)
        torch.testing.assert_close(actual, expected)
        expected = baseline(
            x[9:].clone(), y[:0], num_classes=7, cache=a.freeze()
        )
        actual = placed(x[9:].clone(), y[:0], num_classes=7, cache=b.freeze())
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", ["stage", "layers"])
def test_full_recipe_resident_executor(mode):
    model = KumoTabular(task="classification", pretrained=False, device="meta")
    model.models["classification"] = _KumoTabular(
        num_classes=3,
        num_quantiles=0,
        cell_channels=8,
        num_embedding_layers=2,
        num_embedding_heads=2,
        num_inducing_points=4,
        num_frequencies=4,
        num_readout_tokens=2,
        icl_channels=16,
        num_icl_layers=2,
        num_icl_heads=2,
    )
    model.eval()
    placed = copy.deepcopy(model)
    install_icl_placement(
        placed.models["classification"], ["cpu", "cpu"], mode
    )
    a, b = EnsembleParallel([model]), EnsembleParallel([placed])
    x = TableTensor.from_columns(
        {"a": torch.randn(32), "b": torch.randn(32)},
        stypes={"a": "numerical", "b": "numerical"},
    )
    y = TableTensor.from_columns(
        {"target": torch.arange(32) % 3},
        stypes={"target": "categorical"},
    )
    a.fit(
        x[:24],
        y[:24],
        num_estimators=2,
        generator=torch.Generator().manual_seed(75),
        member_seed=90,
    )
    b.fit(
        x[:24],
        y[:24],
        num_estimators=2,
        generator=torch.Generator().manual_seed(75),
        member_seed=90,
    )
    torch.testing.assert_close(
        a.predict(x[24:]).numerical, b.predict(x[24:]).numerical
    )
    a.close()
    b.close()


@pytest.mark.parametrize("mode", ["stage", "layers"])
@pytest.mark.parametrize("gpu", [False, True])
def test_reduced_query_heads_cache(mode, gpu):
    if gpu and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    devices = ["cuda:0", "cuda:1"] if gpu else ["cpu", "cpu"]
    baseline = make_block("tabular").to(devices[0])
    baseline.kv_heads = 1
    placed = ICLPlacement(copy.deepcopy(baseline), devices, mode)
    x = torch.randn(15, 16, device=devices[0])
    y = torch.arange(9, device=devices[0]) % 3
    with torch.inference_mode():
        a, b = Cache(), Cache()
        baseline(x[:9].clone(), y, cache=a)
        placed(x[:9].clone(), y, cache=b)
        expected = baseline(x[9:].clone(), y[:0], cache=a.freeze())
        actual = placed(x[9:].clone(), y[:0], cache=b.freeze())
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("family", ["tabular", "relational"])
@pytest.mark.parametrize("mode", ["stage", "layers"])
def test_nondefault_stream_empty_fit_and_refit(family, mode):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    devices = ["cuda:0", "cuda:1"]
    baseline = make_block(family).to(devices[0])
    remote_setup = torch.cuda.Stream(device=1)
    with torch.cuda.stream(remote_setup):
        placed = ICLPlacement(copy.deepcopy(baseline), devices, mode)
    caller = torch.cuda.Stream(device=0)
    for _ in range(3):
        with torch.cuda.stream(caller), torch.inference_mode():
            x = torch.randn(15, 16, device=devices[0])
            y = torch.arange(9, device=devices[0]) % 3
            a, b = Cache(), Cache()
            baseline(x[:9].clone(), y, cache=a)
            placed(x[:9].clone(), y, cache=b)
            expected = baseline(x[9:].clone(), y[:0], cache=a.freeze())
            actual = placed(x[9:].clone(), y[:0], cache=b.freeze())
        caller.synchronize()
        torch.testing.assert_close(actual, expected)
