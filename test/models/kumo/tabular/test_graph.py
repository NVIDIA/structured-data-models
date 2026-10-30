# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import io
import subprocess
import sys
from typing import cast

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.cache import Cache, KVCacheEntry
from sdm.models import KumoTabular
from sdm.models.kumo.tabular.graph import _CudaGraphs
from sdm.testing import onlyCUDA


def build(device: str = "cpu", task: str = "regression") -> KumoTabular:
    model = KumoTabular(
        task=task, size="small", pretrained=False, device=device
    )
    # Zero residual branches otherwise make untrained outputs independent of x.
    with torch.no_grad():
        for p in model.parameters():
            if not p.any():
                p.normal_(std=0.02)
    return model


def fit(
    model: KumoTabular, device: str, estimators: int = 1, classes: int = 0
) -> TableTensor:
    values = torch.randn(24, 3, device=device)
    values[0, 0] = torch.nan
    x = TableTensor.from_tensor(values)
    if classes:
        y = TableTensor(
            columns={"categorical": ("target",)},
            categorical=CategoricalTensor.from_tensor(
                torch.arange(24, device=device).remainder(classes)[:, None]
            ),
        )
    else:
        y = TableTensor.from_tensor(torch.randn(24, 1, device=device))
    model.fit(
        x=x,
        y=y,
        num_estimators=estimators,
        recipe=sp.Recipe(
            features=[], target=sp.Identity(), output=[sp.AverageEstimators()]
        ),
    )
    return x[:3]


def test_cpu_and_state_dict() -> None:
    model = build()
    query = fit(model, "cpu")
    expected = model.predict(query)
    keys = tuple(model.state_dict())
    model.enable_cuda_graphs()
    torch.testing.assert_close(
        actual=model.predict(query).numerical, expected=expected.numerical
    )
    assert tuple(model.state_dict()) == keys
    model.disable_cuda_graphs()
    torch.testing.assert_close(
        actual=model.predict(query).numerical, expected=expected.numerical
    )


def test_loading_model_saved_before_cuda_graphs() -> None:
    model = build()
    query = fit(model, "cpu")
    expected = model.predict(query).numerical
    # Whole-model files from before replay support lack this field.
    for network in model.models.values():
        del network._cuda_graphs
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    restored.eval().cpu()
    torch.testing.assert_close(restored.predict(query).numerical, expected)
    restored.clear()
    query = fit(restored, "cpu")
    restored.enable_cuda_graphs()
    restored.predict(query)


@onlyCUDA
@pytest.mark.parametrize("classes", [0, 2, 12])
@pytest.mark.parametrize("estimators", [1, 4])
def test_replay_values_refit_and_disable(
    classes: int, estimators: int
) -> None:
    model = build("cuda", "classification" if classes else "regression")
    query = fit(model, "cuda", estimators, classes)
    changed = TableTensor.from_tensor(query.numerical * 2 + 1)
    expected = [model.predict(q).numerical.clone() for q in [query, changed]]
    model.enable_cuda_graphs()
    first = model.predict(query).numerical
    saved = first.clone()
    actual = model.predict(changed).numerical
    torch.testing.assert_close(first, expected[0])
    torch.testing.assert_close(actual, expected[1])
    torch.testing.assert_close(
        actual=first, expected=saved
    )  # Replay must not overwrite callers.
    assert not torch.allclose(expected[0], expected[1])
    query = fit(model, "cuda", estimators, classes)
    actual = model.predict(query).numerical.clone()
    model.disable_cuda_graphs()
    torch.testing.assert_close(model.predict(query).numerical, actual)


@onlyCUDA
@pytest.mark.parametrize("budget", [0, 2 * 1024**3])
def test_shapes_precision_and_serialization(budget: int) -> None:
    model = build("cuda")
    query = fit(model, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        query = fit(model, "cuda")
    model.enable_cuda_graphs(max_graphs=1, max_memory_bytes=budget)
    for q in [query, query[:1], query]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = model.predict(q).numerical.clone()
        other = copy.deepcopy(model)
        other.disable_cuda_graphs()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.testing.assert_close(other.predict(q).numerical, actual)
    # State loading must drop captured weight pointers, including assign=True.
    replacement = build("cuda")
    model.load_state_dict(replacement.state_dict(), assign=True)
    # Retain the old context to isolate weight-pointer invalidation
    # from fit(), which would independently discard graphs.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model.predict(query).numerical.clone()
        model.disable_cuda_graphs()
        torch.testing.assert_close(model.predict(query).numerical, actual)


@onlyCUDA
def test_graph_launch_and_shape_limit() -> None:
    model = build("cuda")
    query = fit(model, "cuda")
    model.enable_cuda_graphs(max_graphs=1)
    model.predict(query)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        model.predict(query)
        torch.cuda.synchronize()
    assert any("cudaGraphLaunch" in e.key for e in p.key_averages())
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        actual = model.predict(query[:1]).numerical.clone()
        torch.cuda.synchronize()
    assert not any("cudaGraphLaunch" in e.key for e in p.key_averages())
    model.disable_cuda_graphs()
    torch.testing.assert_close(model.predict(query[:1]).numerical, actual)


@onlyCUDA
def test_precision_and_stream_changes() -> None:
    model = build("cuda")
    query = fit(model, "cuda")
    expected = model.predict(query).numerical.clone()
    model.enable_cuda_graphs()
    torch.testing.assert_close(model.predict(query).numerical, expected)
    # Precision changes require a new fitted cache, just as in eager mode.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="cached under dtypes"):
            model.predict(query)
        query = fit(model, "cuda")
        actual = model.predict(query).numerical.clone()
        other = copy.deepcopy(model)
        other.disable_cuda_graphs()
        torch.testing.assert_close(other.predict(query).numerical, actual)
    query = fit(model, "cuda")
    other = copy.deepcopy(model)
    other.disable_cuda_graphs()
    expected = other.predict(query).numerical
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = model.predict(query).numerical
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected)


@onlyCUDA
def test_float32_matmul_precision() -> None:
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        model = build("cuda")
        query = fit(model, "cuda")
        expected = model.predict(query).numerical.clone()
        torch.set_float32_matmul_precision("high")
        model.enable_cuda_graphs()
        model.predict(query)
        torch.set_float32_matmul_precision("highest")
        actual = model.predict(query).numerical
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    finally:
        torch.set_float32_matmul_precision(precision)


@onlyCUDA
def test_capture_exceeding_memory_budget() -> None:
    model = build("cuda")
    query = fit(model, "cuda")
    expected = model.predict(query).numerical.clone()
    model.enable_cuda_graphs()
    model.predict(query)
    # Select a budget below this workload's measured pool + buffer requirement,
    # but above its static inputs, so rejection happens after capture.
    budget = model.models["regression"]._cuda_graphs.memory_bytes - 1
    model.disable_cuda_graphs()
    before = torch.cuda.memory_allocated()
    model.enable_cuda_graphs(max_memory_bytes=budget)
    for attempt in range(2):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            actual = model.predict(query).numerical
            torch.cuda.synchronize()
        events = [e.key for e in p.key_averages()]
        assert any("cudaStreamBeginCapture" in e for e in events) == (
            attempt == 0
        )
        assert not any("cudaGraphLaunch" in e for e in events)
        torch.testing.assert_close(actual, expected)
    # Discarding the rejected graph must also release its captured tensors.
    assert torch.cuda.memory_allocated() - before < 1024**2


@onlyCUDA
@pytest.mark.parametrize("priority_only", [False, True])
def test_attention_backend_change(priority_only: bool) -> None:
    model = build("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        query = fit(model, "cuda")
        model.enable_cuda_graphs(max_graphs=1)
        initial = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        changed = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]
        if not priority_only:
            initial = [SDPBackend.FLASH_ATTENTION]
            changed = [SDPBackend.MATH]
        with sdpa_kernel(initial, set_priority=True):
            model.predict(query)
        # With the graph limit reached, a different backend must run normally.
        with sdpa_kernel(changed, set_priority=True):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as p:
                actual = model.predict(query).numerical
                torch.cuda.synchronize()
            assert any(
                "_scaled_dot_product_attention_math" in e.key
                for e in p.key_averages()
            )
            model.disable_cuda_graphs()
            torch.testing.assert_close(actual, model.predict(query).numerical)


def test_new_precision_api() -> None:
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        pytest.skip("Requires the PyTorch 2.9 precision API")
    # A subprocess isolates the mutually exclusive old/new global APIs.
    subprocess.run(
        args=[
            sys.executable,
            "-c",
            """
import torch
from sdm.models.kumo.tabular.graph import _matmul_precision
for precision in ["tf32", "ieee"]:
    torch.backends.fp32_precision = precision
    assert _matmul_precision() == precision
""",
        ],
        check=True,
    )


@onlyCUDA
@pytest.mark.parametrize("classes", [2, 12])
@pytest.mark.parametrize("estimators", [1, 4])
def test_string_class_labels(classes: int, estimators: int) -> None:
    model = build(device="cuda", task="classification")
    x = TableTensor.from_tensor(torch.randn(24, 3, device="cuda"))
    y = TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            code=torch.arange(24, device="cuda").remainder(classes)[:, None],
            categories=(
                StringTensor.from_list(
                    values=[f"class{i}" for i in range(classes)], device="cuda"
                ),
            ),
        ),
    )
    model.fit(x=x, y=y, num_estimators=estimators)
    queries = [x[:3], TableTensor.from_tensor(x[:3].numerical * 2 + 1)]
    expected = [model.predict(query) for query in queries]
    model.enable_cuda_graphs()
    for query, reference in zip(queries, expected):
        actual = model.predict(query)
        assert actual.columns == reference.columns
        torch.testing.assert_close(actual.numerical, reference.numerical)


@onlyCUDA
@pytest.mark.parametrize("change", ["eval", "train", "device"])
def test_graph_invalidation(change: str) -> None:
    model = build("cuda")
    query = fit(model, "cuda")
    model.enable_cuda_graphs()
    expected = model.predict(query).numerical.clone()
    if change == "eval":
        model.eval()
    elif change == "train":
        model.train().eval()
    else:
        # Do not refit: fit() would independently clear the captured graph.
        model.cpu().cuda()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        actual = model.predict(query).numerical
        torch.cuda.synchronize()
    events = [event.key for event in profile.key_averages()]
    assert any("cudaGraphLaunch" in event for event in events)
    assert any("cudaStreamBeginCapture" in event for event in events) == (
        change != "eval"
    )
    torch.testing.assert_close(actual, expected)


@onlyCUDA
@pytest.mark.parametrize("warm", [False, True])
def test_outer_capture(warm: bool) -> None:
    x = torch.ones(4, device="cuda")
    y = torch.ones_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    cache = Cache({"kv": KVCacheEntry(x.clone(), x.clone())}).freeze()

    def forward(
        x: torch.Tensor,
        y: torch.Tensor,
        categorical_mask: torch.Tensor,
        *,
        cache: Cache,
    ) -> torch.Tensor:
        return x + y + cast(KVCacheEntry, cache["kv"]).value

    helper = _CudaGraphs(max_graphs=4, max_memory_bytes=2 * 1024**3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        if warm:
            helper(
                forward=forward, x=x, y=y, categorical_mask=mask, cache=cache
            )
        else:
            forward(x=x, y=y, categorical_mask=mask, cache=cache)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
        actual = helper(
            forward=forward, x=x, y=y, categorical_mask=mask, cache=cache
        )
    x.fill_(2)
    graph.replay()
    torch.testing.assert_close(actual, torch.full_like(x, 4))
