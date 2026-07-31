r"""Benchmark ``ModelTextEmbed.transform`` on STRABLE text columns.

This times the processor path directly: text extraction from
:class:`~sdm.tensor.TableTensor`, the user-provided embedding model call,
shape validation, dtype/device normalization, and output table construction.
It does not run a downstream model such as TabICLv2.

Workflow:

    python benchmarks/bench_llm_text_embed.py \
        --dataset financial-product-complaint \
        --models sentence-transformers/all-MiniLM-L6-v2 intfloat/e5-small-v2 \
        --batch-size 128 --max-rows 4096

Use ``--separate-columns`` to embed each source text column independently
instead of joining all text columns into one synthetic ``__text__`` column.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import ModelTextEmbed


class SentenceTransformerEmbeddingModel(torch.nn.Module):
    """Wrap ``SentenceTransformer.encode`` as a tensor-returning module."""

    def __init__(
        self,
        model_name: str,
        *,
        device: torch.device | str,
        batch_size: int,
        dtype: torch.dtype,
        prompt_name: str | None,
        trust_remote_code: bool,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model = SentenceTransformer(
            model_name,
            device=str(device),
            trust_remote_code=trust_remote_code,
            model_kwargs={"torch_dtype": dtype},
        )
        self.batch_size = batch_size
        self.prompt_name = prompt_name

    def forward(self, strings: Any) -> Tensor:
        """Some docs.

        strings: Any
        """
        if isinstance(strings, pa.Array):
            values = strings.to_pylist()
        else:
            values = strings.to_arrow().to_pylist()
        kwargs: dict[str, Any] = {}
        if self.prompt_name is not None:
            kwargs["prompt_name"] = self.prompt_name

        with torch.inference_mode():
            return self.model.encode(
                values,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                **kwargs,
            )


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"`--dtype` must name a floating-point dtype: {name}")
    return dtype


def _download_strable(dataset: str) -> tuple[str, dict[str, Any]]:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    repo = "inria-soda/STRABLE-benchmark"
    config_path = hf_hub_download(
        repo,
        f"{dataset}/config.json",
        repo_type="dataset",
    )
    data_path = hf_hub_download(
        repo,
        f"{dataset}/data.parquet",
        repo_type="dataset",
    )
    with open(config_path) as f:
        config = json.load(f)
    return data_path, config


def _load_arrow_dataset(
    args: argparse.Namespace,
) -> tuple[pa.Table, dict[str, Any]]:
    data_path, config = _download_strable(args.dataset)
    table = pq.read_table(data_path)
    if args.max_rows is not None:
        table = table.slice(0, args.max_rows)
    return table, config


def _arrow_text_columns(
    table: pa.Table,
    target_name: str | None,
) -> list[str]:
    columns: list[str] = []
    for field in table.schema:
        if field.name == target_name:
            continue
        if pa.types.is_string(field.type) or pa.types.is_large_string(
            field.type
        ):
            columns.append(field.name)
    return columns


def _arrow_numeric_columns(
    table: pa.Table,
    target_name: str | None,
    text_columns: Sequence[str],
) -> list[str]:
    text_column_set = set(text_columns)
    columns: list[str] = []
    for field in table.schema:
        if field.name == target_name or field.name in text_column_set:
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            columns.append(field.name)
    return columns


def _cudf_text_columns(df: Any, target_name: str | None) -> list[str]:
    from cudf.api.types import is_string_dtype

    return [
        column
        for column in df.columns
        if column != target_name and is_string_dtype(df[column].dtype)
    ]


def _joined_text_array(table: pa.Table, columns: Sequence[str]) -> pa.Array:
    rows = zip(*(table[column].to_pylist() for column in columns))
    values = [
        " | ".join(
            f"{column}: {value}"
            for column, value in zip(columns, row)
            if value is not None
        )
        for row in rows
    ]
    return pa.array(values, type=pa.large_string())


def _joined_text_series(df: Any, columns: Sequence[str]) -> Any:
    text = None
    for column in columns:
        values = column + ": " + df[column].fillna("")
        text = values if text is None else text + " | " + values
    return text


def _fill_text_nulls(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    return pa.table(
        {
            column: pc.fill_null(
                table[column],
                pa.scalar("", type=table.schema.field(column).type),
            )
            for column in columns
        }
    )


def _load_table(args: argparse.Namespace) -> tuple[TableTensor, list[str]]:
    if torch.device(args.table_device).type == "cuda":
        import cudf

        data_path, config = _download_strable(args.dataset)
        df = cudf.read_parquet(data_path)
        if args.max_rows is not None:
            df = df.head(args.max_rows)

        target_name = config.get("target_name")
        text_columns = _cudf_text_columns(df, target_name)
        if not text_columns:
            raise SystemExit(
                f"STRABLE table {args.dataset!r} has no text columns"
            )

        if args.separate_columns:
            df = df[text_columns].fillna("")
            stypes = dict.fromkeys(text_columns, Stype.text)
            return (
                TableTensor.from_cudf(
                    df,
                    stypes=stypes,
                    device=args.table_device,
                ),
                text_columns,
            )

        df = cudf.DataFrame(
            {"__text__": _joined_text_series(df, text_columns)}
        )
        return (
            TableTensor.from_cudf(
                df,
                stypes={"__text__": Stype.text},
                device=args.table_device,
            ),
            text_columns,
        )

    data_path, config = _download_strable(args.dataset)
    table = pq.read_table(data_path)
    if args.max_rows is not None:
        table = table.slice(0, args.max_rows)

    target_name = config.get("target_name")
    text_columns = _arrow_text_columns(table, target_name)
    if not text_columns:
        raise SystemExit(f"STRABLE table {args.dataset!r} has no text columns")

    if args.separate_columns:
        arrow_table = _fill_text_nulls(table, text_columns)
        stypes = dict.fromkeys(text_columns, Stype.text)
        return (
            TableTensor.from_arrow(
                arrow_table,
                stypes=stypes,
                device=args.table_device,
            ),
            text_columns,
        )

    arrow_table = pa.table(
        {"__text__": _joined_text_array(table, text_columns)}
    )
    return (
        TableTensor.from_arrow(
            arrow_table,
            stypes={"__text__": Stype.text},
            device=args.table_device,
        ),
        text_columns,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_transform(
    processor: ModelTextEmbed,
    table: TableTensor,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        processor.transform(table)
    _sync(device)

    latencies: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        processor.transform(table)
        _sync(device)
        latencies.append(time.perf_counter() - start)
    return latencies


def _embedding_dim(model: torch.nn.Module, table: TableTensor) -> int:
    column_text = table.text[..., 0].reshape(-1)
    strings = (
        column_text.to_cudf()
        if column_text.is_cuda
        else column_text.to_arrow()
    )
    with torch.inference_mode():
        strings = strings[: min(2, len(strings))]
        embeddings = model(strings)
    if not isinstance(embeddings, Tensor) or embeddings.dim() != 2:
        raise TypeError(
            "Expected embedding model probe to return a 2D Tensor "
            f"(got {type(embeddings).__name__})"
        )
    return embeddings.size(-1)


def _report(
    *,
    model_name: str,
    setup_s: float,
    latencies: list[float],
    output_width: int,
    table: TableTensor,
    text_columns: Sequence[str],
    args: argparse.Namespace,
) -> None:
    lat_ms = [latency * 1000 for latency in latencies]
    result = {
        "dataset": args.dataset,
        "source_text_columns": list(text_columns),
        "joined_text_columns": not args.separate_columns,
        "rows": table.text.size(0),
        "processor_text_columns": table.text.size(-1),
        "model": model_name,
        "device": args.device,
        "table_device": args.table_device,
        "dtype": args.dtype,
        "embedding_batch_size": args.batch_size,
        "output_width": output_width,
        "setup_s": round(setup_s, 3),
        "latency_ms": {
            "mean": round(statistics.mean(lat_ms), 2),
            "p50": round(statistics.median(lat_ms), 2),
            "min": round(min(lat_ms), 2),
        },
        "rows_per_s": round(table.text.size(0) / statistics.mean(latencies)),
    }
    if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
        result["peak_gpu_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024**2
        )

    print(json.dumps(result, indent=2))  # noqa: T201
    if args.out is not None:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


def _emit_result(result: dict[str, Any], args: argparse.Namespace) -> None:
    print(json.dumps(result, indent=2))  # noqa: T201
    if args.out is not None:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


def _make_embedding_processor(
    model_name: str,
    table: TableTensor,
    args: argparse.Namespace,
) -> tuple[ModelTextEmbed, float, int]:
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    start = time.perf_counter()
    model = SentenceTransformerEmbeddingModel(
        model_name,
        device=device,
        batch_size=args.batch_size,
        dtype=dtype,
        prompt_name=args.prompt_name,
        trust_remote_code=args.trust_remote_code,
    )
    embedding_dim = _embedding_dim(model, table)
    processor = ModelTextEmbed(
        embedding_model=model,
        embedding_dim=embedding_dim,
        dtype=dtype,
    )
    setup_s = time.perf_counter() - start
    return processor, setup_s, embedding_dim


def _run_model(
    model_name: str,
    table: TableTensor,
    text_columns: Sequence[str],
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    processor, setup_s, _ = _make_embedding_processor(
        model_name,
        table,
        args,
    )

    latencies = _time_transform(
        processor,
        table,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    output = processor.transform(table)
    _sync(device)

    _report(
        model_name=model_name,
        setup_s=setup_s,
        latencies=latencies,
        output_width=output.numerical.size(-1),
        table=table,
        text_columns=text_columns,
        args=args,
    )
    del output, processor


def _load_tabicl_inputs(
    args: argparse.Namespace,
) -> tuple[TableTensor, Tensor, Tensor | None, list[str], list[str]]:
    text, text_columns = _load_table(args)
    table, config = _load_arrow_dataset(args)
    target_name = config.get("target_name")
    if not isinstance(target_name, str):
        raise SystemExit(
            f"STRABLE table {args.dataset!r} config has no target_name"
        )

    target = table[target_name].to_pandas().reset_index(drop=True)
    mask = target.notna().to_numpy()
    labels, uniques = target[mask].factorize(sort=True)
    if len(uniques) > 10:
        raise SystemExit(
            f"{len(uniques)} classes found; TabICLv2 supports at most 10."
        )
    y = torch.tensor(labels, dtype=torch.int64, device=args.device)
    text = text[mask]

    numeric = None
    numeric_columns: list[str] = []
    if args.include_numeric:
        numeric_columns = _arrow_numeric_columns(
            table,
            target_name,
            text_columns,
        )
        if numeric_columns:
            numeric_table = table.select(numeric_columns).to_pandas()
            numeric_values = numeric_table[mask].to_numpy(dtype="float32")
            numeric = torch.tensor(
                numeric_values,
                dtype=torch.float32,
                device=args.device,
            )

    return text, y, numeric, text_columns, numeric_columns


def _cleanup_model_run(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _macro_f1(pred: Tensor, target: Tensor) -> float:
    classes = target.unique().detach().cpu()
    pred_cpu = pred.detach().cpu()
    target_cpu = target.detach().cpu()
    scores: list[float] = []
    for cls in classes:
        pred_pos = pred_cpu == cls
        target_pos = target_cpu == cls
        tp = (pred_pos & target_pos).sum().item()
        fp = (pred_pos & ~target_pos).sum().item()
        fn = (~pred_pos & target_pos).sum().item()
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(statistics.mean(scores))


def _split_indices(
    y: Tensor, args: argparse.Namespace
) -> tuple[Tensor, Tensor]:
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    indices = torch.arange(y.numel()).cpu().numpy()
    labels = y.detach().cpu().numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    return (
        torch.tensor(train_idx, dtype=torch.long, device=y.device),
        torch.tensor(test_idx, dtype=torch.long, device=y.device),
    )


def _run_tabicl_model(
    model_name: str,
    table: TableTensor,
    y: Tensor,
    numeric: Tensor | None,
    text_columns: Sequence[str],
    numeric_columns: Sequence[str],
    args: argparse.Namespace,
) -> None:
    from sdm.models import TabICLv2  # noqa: PLC0415

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    processor, setup_s, _ = _make_embedding_processor(model_name, table, args)

    _sync(device)
    start = time.perf_counter()
    embedded = processor.transform(table).numerical
    _sync(device)
    embedding_s = time.perf_counter() - start

    features = (
        embedded if numeric is None else torch.cat([numeric, embedded], dim=-1)
    )
    train_idx, test_idx = _split_indices(y, args)
    x_train = TableTensor.from_tensor(features[train_idx])
    x_test = TableTensor.from_tensor(features[test_idx])
    y_train = TableTensor.from_tensor(y[train_idx].unsqueeze(-1))
    y_test = y[test_idx]

    model = TabICLv2(device=device)

    _sync(device)
    start = time.perf_counter()
    model.fit(x=x_train, y=y_train)
    _sync(device)
    tabicl_fit_s = time.perf_counter() - start

    start = time.perf_counter()
    output = model.predict(x=x_test)
    _sync(device)
    tabicl_predict_s = time.perf_counter() - start
    model.clear()

    pred = output.numerical.argmax(dim=-1)
    accuracy = (pred == y_test).float().mean().item()
    result = {
        "mode": "tabicl",
        "dataset": args.dataset,
        "model": model_name,
        "rows": y.numel(),
        "train_rows": train_idx.numel(),
        "test_rows": test_idx.numel(),
        "source_text_columns": list(text_columns),
        "joined_text_columns": not args.separate_columns,
        "numeric_columns": list(numeric_columns),
        "include_numeric": args.include_numeric,
        "device": args.device,
        "table_device": args.table_device,
        "dtype": args.dtype,
        "embedding_batch_size": args.batch_size,
        "feature_width": features.size(-1),
        "embedding_width": embedded.size(-1),
        "timing_s": {
            "embedding_model_setup": round(setup_s, 3),
            "embedding_transform": round(embedding_s, 3),
            "tabicl_fit": round(tabicl_fit_s, 3),
            "tabicl_predict": round(tabicl_predict_s, 3),
        },
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(_macro_f1(pred, y_test), 4),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        result["peak_gpu_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024**2
        )
    _emit_result(result, args)
    del model, output, processor, embedded, features


def main() -> None:
    """Run ``ModelTextEmbed.transform`` for the requested STRABLE table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--mode", choices=["transform", "tabicl"], default="transform"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sentence-transformers/all-MiniLM-L6-v2"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--table-device",
        default=None,
        help=(
            "Device for the input TableTensor text block. Defaults to "
            "--device so CUDA runs keep processor output on GPU."
        ),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt-name", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--separate-columns", action="store_true")
    parser.add_argument("--include-numeric", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.table_device is None:
        args.table_device = args.device

    uses_cuda = (
        torch.device(args.device).type == "cuda"
        or torch.device(args.table_device).type == "cuda"
    )
    if uses_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu.")

    if args.mode == "transform":
        table, text_columns = _load_table(args)
        for model_name in args.models:
            _run_model(model_name, table, text_columns, args)
            _cleanup_model_run(torch.device(args.device))
        return

    table, y, numeric, text_columns, numeric_columns = _load_tabicl_inputs(
        args
    )
    for model_name in args.models:
        _run_tabicl_model(
            model_name,
            table,
            y,
            numeric,
            text_columns,
            numeric_columns,
            args,
        )
        _cleanup_model_run(torch.device(args.device))


if __name__ == "__main__":
    main()
