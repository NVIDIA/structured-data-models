import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.nn.attention import SDPBackend

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def benchmark() -> ModuleType:
    # The benchmark is a script, not a package module; import it the way
    # the GPU workflow's import smoke does (``PYTHONPATH=examples``).
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module("benchmark_tabiclv2")
    finally:
        sys.path.remove(str(EXAMPLES))


def test_sdpa_backends(benchmark: ModuleType) -> None:
    # ``False`` keeps torch's default order, ``True`` pins cuDNN first, and
    # a tuple of names pins exactly that order.
    assert benchmark.sdpa_backends(False) is None
    assert benchmark.sdpa_backends(True) == [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    assert benchmark.sdpa_backends(("MATH", "FLASH_ATTENTION")) == [
        SDPBackend.MATH,
        SDPBackend.FLASH_ATTENTION,
    ]
    # Every configuration's priority field resolves, and the flash cell
    # leaves cuDNN attention out.
    for spec in benchmark.CONFIGS.values():
        benchmark.sdpa_backends(spec[1])
    flash = benchmark.sdpa_backends(
        benchmark.CONFIGS["c21-compile-dynamic-flash"][1]
    )
    assert flash is not None
    assert flash[0] is SDPBackend.FLASH_ATTENTION
    assert SDPBackend.CUDNN_ATTENTION not in flash


def test_cudnn_attention_cell(benchmark: ModuleType) -> None:
    # The cuDNN SDPA backend takes fp16/bf16 only, so the fp32/tf32 cells
    # never reach it; torch's default order and the cuDNN-first pin select
    # it, a pinned order without it does not.
    assert not benchmark.cudnn_attention_cell("c0-fp32")
    assert not benchmark.cudnn_attention_cell("c1-tf32")
    assert benchmark.cudnn_attention_cell("c3-bf16-full")
    assert benchmark.cudnn_attention_cell("c6-compile-dynamic")
    assert benchmark.cudnn_attention_cell("c13-fp16-full")
    assert not benchmark.cudnn_attention_cell("c21-compile-dynamic-flash")
    for config in benchmark.CONFIGS:
        assert isinstance(benchmark.cudnn_attention_cell(config), bool)


def test_bucketed_cell(benchmark: ModuleType) -> None:
    # Only the configurations with a truthy fifth tuple element pad to the
    # bucket grid; the four-element ones serve fresh shapes as they come.
    assert not benchmark.bucketed_cell("c0-fp32")
    assert not benchmark.bucketed_cell("c6-compile-dynamic")
    assert not benchmark.bucketed_cell("c17-regional-compile")
    assert benchmark.bucketed_cell("c15-bucket-compile-dynamic")
    assert benchmark.bucketed_cell("c16-bucket-compile-ro")
    assert benchmark.bucketed_cell("c18-bucket-regional")
    for config, spec in benchmark.CONFIGS.items():
        assert benchmark.bucketed_cell(config) is bool(
            len(spec) > 4 and spec[4]
        )


def test_bucket_grids(benchmark: ModuleType) -> None:
    # Counts round up to the grid, grid sizes map to themselves, zero rows
    # stay zero, and counts beyond the grid round up to the next multiple of
    # the grid's largest step.
    assert benchmark.bucket_rows(0) == 0
    assert benchmark.bucket_rows(1) == 16
    assert benchmark.bucket_rows(17) == 24
    assert benchmark.bucket_rows(16385) == 20480
    assert all(benchmark.bucket_rows(n) == n for n in benchmark.ROW_BUCKETS)
    assert benchmark.bucket_cols(1) == 4
    assert benchmark.bucket_cols(5) == 8
    assert benchmark.bucket_cols(129) == 160
    assert all(benchmark.bucket_cols(n) == n for n in benchmark.COL_BUCKETS)


def test_pad_to_buckets_exact_fit(benchmark: ModuleType) -> None:
    # On-grid tables serve through the unmasked graph family: nothing is
    # padded and no ``seqused_*`` argument is passed.
    x = torch.randn(256, 8)
    y = torch.randint(0, 4, (192,))
    x_padded, y_padded, seqused, num_test = benchmark.pad_to_buckets(x, y)
    assert x_padded is x
    assert y_padded is y
    assert seqused == {}
    assert num_test == 64


@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_pad_to_buckets_off_grid(
    benchmark: ModuleType,
    batch_shape: tuple[int, ...],
) -> None:
    num_train, num_test, num_cols = 9, 4, 5
    x = torch.randn(*batch_shape, num_train + num_test, num_cols)
    y = torch.randint(1, 4, (*batch_shape, num_train))
    x_padded, y_padded, seqused, out_test = benchmark.pad_to_buckets(x, y)

    # 9 train and 4 test rows land in the 16-row bucket each, 5 columns in
    # the 8-column bucket; the true test-row count is reported unchanged.
    assert out_test == num_test
    assert x_padded.size() == (*batch_shape, 32, 8)
    assert y_padded.size() == (*batch_shape, 16)

    # Real rows and columns stay in place, padding is zero.
    torch.testing.assert_close(
        x_padded[..., :num_train, :num_cols], x[..., :num_train, :]
    )
    torch.testing.assert_close(
        x_padded[..., 16 : 16 + num_test, :num_cols], x[..., num_train:, :]
    )
    assert not x_padded[..., num_train:16, :].any()
    assert not x_padded[..., 16 + num_test :, :].any()
    assert not x_padded[..., :, num_cols:].any()

    # Padded labels repeat a real one so the class set (and with it the
    # number of output columns) is unchanged; a zero pad would add a class.
    assert torch.equal(y_padded[..., :num_train], y)
    assert torch.equal(
        y_padded[..., num_train:],
        y[..., :1].expand(*batch_shape, 16 - num_train),
    )

    # The counts carry the batch shape and the model's expected dtype.
    assert set(seqused) == {"seqused_train", "seqused_cols"}
    seqused_train, seqused_cols = (
        seqused["seqused_train"],
        seqused["seqused_cols"],
    )
    assert seqused_train.dtype == seqused_cols.dtype == torch.int32
    assert seqused_train.size() == batch_shape
    assert (seqused_train == num_train).all()
    assert seqused_cols.dim() == 0
    assert seqused_cols.item() == num_cols


def test_pad_to_buckets_without_train_rows(benchmark: ModuleType) -> None:
    # With no real label to repeat, the label pad falls back to zeros of
    # the (zero-size) train bucket instead of failing on an empty expand.
    x = torch.randn(4, 5)
    y = torch.empty(0, dtype=torch.long)
    x_padded, y_padded, seqused, num_test = benchmark.pad_to_buckets(x, y)
    assert num_test == 4
    assert x_padded.size() == (16, 8)
    assert y_padded.numel() == 0
    assert seqused["seqused_train"].item() == 0


def test_reachable_buckets_cover_jittered_stream(
    benchmark: ModuleType,
) -> None:
    # The bucket warmup visits ``reachable_buckets``; every table the
    # jittered stream produces must pad onto one of them, or the timed
    # passes would absorb fresh-shape costs.
    workload = "small"
    buckets = benchmark.reachable_buckets(workload)
    _, num_rows, num_cols, num_train = benchmark.WORKLOADS[workload]
    base = (
        benchmark.bucket_rows(num_train),
        benchmark.bucket_rows(num_rows - num_train),
        benchmark.bucket_cols(num_cols),
    )
    assert base in buckets
    for train, test, cols in buckets:
        assert benchmark.bucket_rows(train) == train
        assert benchmark.bucket_rows(test) == test
        assert benchmark.bucket_cols(cols) == cols

    device = torch.device("cpu")
    for jitter in range(1, 9):
        x, y = benchmark.make_table(
            workload, "cls", seed=jitter, device=device, jitter=jitter
        )
        x_padded, y_padded, _, num_test = benchmark.pad_to_buckets(x, y)
        assert num_test == x.size(-2) - y.size(-1)
        padded_train = y_padded.size(-1)
        shape = (
            padded_train,
            x_padded.size(-2) - padded_train,
            x_padded.size(-1),
        )
        assert shape in buckets


def test_make_table(benchmark: ModuleType) -> None:
    device = torch.device("cpu")
    _, num_rows, num_cols, num_train = benchmark.WORKLOADS["small"]

    # Seeded: the same seed reproduces the table, the canonical shape is
    # the workload's, and classification labels stay within the class set.
    x, y = benchmark.make_table("small", "cls", seed=3, device=device)
    x_again, y_again = benchmark.make_table(
        "small", "cls", seed=3, device=device
    )
    assert torch.equal(x, x_again)
    assert torch.equal(y, y_again)
    assert x.size() == (num_rows, num_cols)
    assert y.size() == (num_train,)
    assert y.dtype == torch.long
    assert y.min() >= 0
    assert y.max() < 4

    # Jitter grows rows by less than a quarter and adds up to three
    # columns, keeping the 3:1 train/test split.
    x, y = benchmark.make_table(
        "small", "reg", seed=3, device=device, jitter=5
    )
    assert num_rows <= x.size(-2) < num_rows + num_rows // 4
    assert num_cols <= x.size(-1) < num_cols + 4
    assert y.size(-1) == (x.size(-2) * 3) // 4
    assert y.is_floating_point()


def test_accuracy_block_classification(benchmark: ModuleType) -> None:
    # One-hot logits with a unit decision margin.
    ref = torch.eye(4).repeat(50, 1)

    block = benchmark.accuracy_block("cls", ref, ref)
    assert block["pass"] is True
    assert block["top1_agreement"] == 1.0
    assert block["margin_ratio"] == 0.0
    assert block["max_abs_diff"] == 0.0

    # A uniform shift well inside the margin passes...
    block = benchmark.accuracy_block("cls", ref + 0.01, ref)
    assert block["pass"] is True
    assert block["margin_ratio"] == pytest.approx(0.01)

    # ...while flipping the decision of most rows fails top-1 agreement.
    flipped = ref.clone()
    flipped[:, 0] += 2.0
    block = benchmark.accuracy_block("cls", flipped, ref)
    assert block["pass"] is False
    assert block["top1_agreement"] == pytest.approx(0.25)

    # Each gate term must fail on its own. A shift that eats a quarter of
    # every row's margin keeps top-1 intact but trips the 10% margin gate
    # (the nvfp4 cell fails exactly this way)...
    shifted = ref.clone()
    shifted[:, 1] += 0.25
    block = benchmark.accuracy_block("cls", shifted, ref)
    assert block["top1_agreement"] == 1.0
    assert block["margin_ratio"] == pytest.approx(0.25)
    assert block["pass"] is False

    # ...and flipping one row in a hundred keeps the median margin ratio at
    # zero but trips the 99.5% top-1 gate.
    one_flip = ref.clone()
    one_flip[::100, 1] += 2.0
    block = benchmark.accuracy_block("cls", one_flip, ref)
    assert block["margin_ratio"] == 0.0
    assert block["top1_agreement"] == pytest.approx(0.99)
    assert block["pass"] is False


def test_accuracy_block_regression(benchmark: ModuleType) -> None:
    # Quantile rows spanning [0, 1], so relative errors equal absolute ones.
    ref = torch.linspace(0.0, 1.0, 9).repeat(10, 1)

    block = benchmark.accuracy_block("reg", ref, ref)
    assert block["pass"] is True
    assert block["median_rel_err"] == 0.0

    block = benchmark.accuracy_block("reg", ref + 0.005, ref)
    assert block["pass"] is True
    assert block["median_rel_err"] == pytest.approx(0.005)
    assert block["p99_rel_err"] == pytest.approx(0.005)

    # A 2% shift trips the 1% median floor.
    block = benchmark.accuracy_block("reg", ref + 0.02, ref)
    assert block["pass"] is False

    # One row shifted by 10% (a tenth of the elements) leaves the median at
    # zero and trips only the 5% p99 floor.
    outlier = ref.clone()
    outlier[0] += 0.1
    block = benchmark.accuracy_block("reg", outlier, ref)
    assert block["median_rel_err"] == 0.0
    assert block["p99_rel_err"] == pytest.approx(0.1)
    assert block["pass"] is False


def test_cuda_cache_file_count(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Counts recursively under CUDA_CACHE_PATH (the driver nests files in
    # sub-directories) so a cell's before/after delta sees every miss; a
    # missing directory counts as empty rather than raising.
    monkeypatch.setenv("CUDA_CACHE_PATH", str(tmp_path / "missing"))
    assert benchmark.cuda_cache_file_count() == 0
    cache = tmp_path / "cache"
    (cache / "0" / "1").mkdir(parents=True)
    (cache / "0" / "1" / "a").write_bytes(b"x")
    (cache / "0" / "b").write_bytes(b"x")
    monkeypatch.setenv("CUDA_CACHE_PATH", str(cache))
    assert benchmark.cuda_cache_file_count() == 2
    # Unset, the driver's default location is used.
    monkeypatch.delenv("CUDA_CACHE_PATH")
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".nv" / "ComputeCache").mkdir(parents=True)
    (tmp_path / ".nv" / "ComputeCache" / "c").write_bytes(b"x")
    assert benchmark.cuda_cache_file_count() == 1


def test_format_record_rows(benchmark: ModuleType) -> None:
    base = {
        "config": "c18-bucket-regional",
        "workload": "small",
        "task": "cls",
        "route": "oneshot",
        "p50_s": 0.0123,
        "iqr_s": 0.0001,
        "table_stream_per_table_s": 0.0137,
        "table_stream_fresh2_per_table_s": 0.0133,
        "cold_first_call_s": 15.4,
        "peak_mem_mb": 275.0,
        "accuracy": {"pass": True},
        "stream_drift": 1.01,
    }
    row = benchmark.format_record(base)
    assert "p50=   12.30ms" in row
    assert "cold=  15.4s" in row
    assert row.endswith("pass=True sdrift=1.010x")

    # A bucketed cell prints its second one-time cost next to the cold
    # call, and a stream-drift ratio off by more than 5% is flagged.
    bucketed = dict(base, bucket_warmup_s=15.2, stream_drift=1.104)
    row = benchmark.format_record(bucketed)
    assert " warm= 15.2s" in row
    assert row.endswith("sdrift=1.104x DRIFT")

    # A failed padding gate overrides the headline pass.
    gated = dict(base, padding_gate={"pass": False})
    assert "pass=False" in benchmark.format_record(gated)

    # A pass-1 first table more than 3x its pass median carried a one-time
    # JIT cost that inflates stream=; a first table within 3x does not.
    steady = [0.0137] * 7
    assert "JIT" not in benchmark.format_record(
        dict(base, table_stream_tables_s=[2.5 * 0.0137, *steady])
    )
    jit = benchmark.format_record(
        dict(base, table_stream_tables_s=[3.5 * 0.0137, *steady])
    )
    assert jit.endswith("pass=True JIT sdrift=1.010x")
    # An unbucketed cuDNN-attention cell that grew the CUDA JIT cache (or
    # ran with it disabled) hit the compiler inside its timed stream even
    # when every table paid alike, so the first-table ratio is silent; the
    # environment fields flag it instead.
    unbucketed = dict(base, config="c6-compile-dynamic")
    warm = dict(unbucketed, environment={"cuda_cache_files_added": 0})
    assert "JIT" not in benchmark.format_record(warm)
    grown = dict(unbucketed, environment={"cuda_cache_files_added": 14})
    assert benchmark.format_record(grown).endswith("JIT sdrift=1.010x")
    disabled = dict(unbucketed, environment={"cuda_cache_disabled": True})
    assert benchmark.format_record(disabled).endswith("JIT sdrift=1.010x")
    # The file count is raw (torch's own JIT'd kernels land in the same
    # directory), so the cache state marks a plan-build miss only for cells
    # whose attention can reach the cuDNN backend: the fp32 reference and
    # the flash-priority cell stay unflagged with a flat stream. A bucketed
    # cell builds its plans in bucket warmup, outside the timed stream, so
    # its growth does not inflate stream= and is not flagged either; the
    # first-table rule still applies to it.
    for config in ("c0-fp32", "c21-compile-dynamic-flash", base["config"]):
        for environment in (
            {"cuda_cache_files_added": 12},
            {"cuda_cache_disabled": True},
        ):
            other = dict(base, config=config, environment=environment)
            assert "JIT" not in benchmark.format_record(other)
            assert "JIT" in benchmark.format_record(
                dict(other, table_stream_tables_s=[3.5 * 0.0137, *steady])
            )

    # Fit/predict rows print the cached-predict p50, the fit and the
    # route's own cold cost instead of the one-shot stream columns.
    fitpredict = {
        "config": "c16-bucket-compile-ro",
        "workload": "large",
        "task": "cls",
        "route": "fitpredict",
        "p50_s": 0.00531,
        "fit_s": 0.01584,
        "fitpredict_cold_s": 12.3,
        "peak_mem_mb": 900.0,
        "accuracy": {"pass": True},
    }
    row = benchmark.format_record(fitpredict)
    assert "p50=    5.31ms fit=   15.84ms cold=  12.3s" in row
    assert "stream=" not in row
