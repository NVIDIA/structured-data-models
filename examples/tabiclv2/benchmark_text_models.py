"""Compare text models as drop-in TabICLv2 processors on STRABLE.

The benchmark runs the standard ``sdm.processing.SentenceTransformer`` path
with raw table strings, matching the public STRABLE example. Every processor
uses the same sampled rows, split, TabICLv2 recipe, and estimator generator.
Sentence Transformer outputs are reduced to 64 features with PCA fitted only
on the context rows.

Run with:
    PYTHONPATH=. python examples/tabiclv2/benchmark_text_models.py
"""

from __future__ import annotations

import csv
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import sdm
import sdm.processing as sp
from sdm import Stype, TableTensor
from sdm.evaluation import to_class_indices

DATASET_REPO = "inria-soda/STRABLE-benchmark"
DATASET_NAMES = (
    "clear-corpus",
    "mercari",
    "financial-product-complaint",
    "kickstarter-projects",
    "covid-clinical-trials",
)
MAX_ROWS = 4096
BATCH_SIZE = 32
NUM_ESTIMATORS = 8
SEED = 0
OUTPUT = Path("text_model_performance.csv")

Task = Literal["regression", "m-classification", "b-classification"]
ProcessorKind = Literal["none", "tfidf", "embedding"]


@dataclass(frozen=True)
class ProcessorSpec:
    name: str
    kind: ProcessorKind
    tokenizer: str
    model_name: str = ""


PROCESSORS = (
    ProcessorSpec(name="none", kind="none", tokenizer="n/a"),
    ProcessorSpec(name="tfidf", kind="tfidf", tokenizer="character n-gram"),
    ProcessorSpec(
        name="all-MiniLM-L6-v2",
        kind="embedding",
        tokenizer="WordPiece",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ),
    ProcessorSpec(
        name="bge-base-en-v1.5",
        kind="embedding",
        tokenizer="WordPiece",
        model_name="BAAI/bge-base-en-v1.5",
    ),
    ProcessorSpec(
        name="all-distilroberta-v1",
        kind="embedding",
        tokenizer="BPE",
        model_name="sentence-transformers/all-distilroberta-v1",
    ),
    ProcessorSpec(
        name="modernbert-embed-base",
        kind="embedding",
        tokenizer="BPE",
        model_name="nomic-ai/modernbert-embed-base",
    ),
    ProcessorSpec(
        name="Qwen3-Embedding-0.6B",
        kind="embedding",
        tokenizer="BPE",
        model_name="Qwen/Qwen3-Embedding-0.6B",
    ),
)


@dataclass(frozen=True)
class Dataset:
    name: str
    context: pa.Table
    query: pa.Table
    task: Task
    target_name: str
    text_stypes: dict[str, Stype]
    plain_stypes: dict[str, Stype]
    num_text_columns: int


@dataclass(frozen=True)
class Result:
    dataset: str
    task: Task
    processor: str
    tokenizer: str
    model: str
    num_rows: int
    num_context_rows: int
    num_query_rows: int
    num_text_columns: int
    primary_metric: str
    primary_value: float
    secondary_metric: str
    secondary_value: float
    primary_error: float
    error_reduction_vs_none_pct: float | None
    runtime_s: float


def _load_dataset(name: str, rng: np.random.Generator) -> Dataset:
    config_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{name}/config.json",
        repo_type="dataset",
    )
    with Path(config_path).open() as file:
        config = json.load(file)
    task = cast(Task, config["task"])
    target_name = cast(str, config["target_name"])

    data_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{name}/data.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(data_path)
    target_stype = (
        Stype.numerical if task == "regression" else Stype.categorical
    )
    overrides = {target_name: target_stype}
    text_stypes = {
        column: Stype(stype)
        for column, stype in sdm.infer_stypes(
            table,
            overrides=overrides,
            text="infer",
        ).items()
    }
    plain_stypes = {
        column: Stype(stype)
        for column, stype in sdm.infer_stypes(
            table,
            overrides=overrides,
            text="off",
        ).items()
    }
    if len(table) > MAX_ROWS:
        indices = np.sort(rng.choice(len(table), MAX_ROWS, replace=False))
        table = table.take(pa.array(indices))
    order = rng.permutation(len(table))
    split_index = int(0.8 * len(table))
    context = table.take(pa.array(order[:split_index]))
    query = table.take(pa.array(order[split_index:]))

    num_text_columns = sum(
        stype == Stype.text for stype in text_stypes.values()
    )
    return Dataset(
        name=name,
        context=context,
        query=query,
        task=task,
        target_name=target_name,
        text_stypes=text_stypes,
        plain_stypes=plain_stypes,
        num_text_columns=num_text_columns,
    )


def _make_recipe(
    model: sdm.models.TabICLv2,
    spec: ProcessorSpec,
    embedding: sp.SentenceTransformer | None,
) -> sp.Recipe:
    recipe = model.default_recipe()
    if spec.kind == "none":
        return recipe
    if spec.kind == "tfidf":
        text_processor: object = sp.TFIDF(
            ngram_range=(4, 6),
            max_features=256,
        )
    else:
        assert embedding is not None
        text_processor = [embedding, sp.PCA(64)]
    recipe.prepend_features(sp.StypeDispatch(text=text_processor))
    return recipe


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate(
    prediction: TableTensor,
    target: TableTensor,
    task: Task,
) -> tuple[str, float, str, float, float]:
    if task == "regression":
        point_prediction = prediction.numerical.float().mean(dim=-1)
        expected = target.numerical.float().squeeze(-1)
        error = point_prediction - expected
        rmse = error.pow(2).mean().sqrt().item()
        mae = error.abs().mean().item()
        return "rmse", rmse, "mae", mae, rmse

    probabilities, expected = to_class_indices(prediction, target)
    probabilities = probabilities.float()
    accuracy = (probabilities.argmax(dim=-1) == expected).float().mean().item()
    log_loss = (
        -probabilities.clamp_min(1e-12)
        .log()
        .gather(
            dim=-1,
            index=expected.unsqueeze(-1),
        )
        .mean()
        .item()
    )
    return "accuracy", accuracy, "log_loss", log_loss, 1.0 - accuracy


def _run(
    model: sdm.models.TabICLv2,
    dataset: Dataset,
    spec: ProcessorSpec,
    embedding: sp.SentenceTransformer | None,
    baseline_error: float | None,
    *,
    device: torch.device,
) -> Result:
    stypes = (
        dataset.plain_stypes if spec.kind == "none" else dataset.text_stypes
    )
    context = sdm.TableTensor.from_arrow(
        table=dataset.context,
        stypes=stypes,
        device=device,
    )
    query = sdm.TableTensor.from_arrow(
        table=dataset.query,
        stypes=stypes,
        device=device,
    )
    recipe = _make_recipe(model, spec, embedding)
    model_generator = torch.Generator(device=device).manual_seed(SEED)

    _sync(device)
    start = time.perf_counter()
    with torch.amp.autocast(
        device.type,
        torch.float16,
        enabled=context.is_cuda,
    ):
        prediction = model(
            x_context=context.drop_columns(dataset.target_name),
            y_context=context[:, dataset.target_name],
            x_query=query.drop_columns(dataset.target_name),
            recipe=recipe,
            num_estimators=NUM_ESTIMATORS,
            generator=model_generator,
        )
    _sync(device)
    runtime = time.perf_counter() - start

    target = query[:, dataset.target_name]
    primary_metric, primary, secondary_metric, secondary, error = _evaluate(
        prediction,
        target,
        dataset.task,
    )
    reduction = None
    if baseline_error is not None:
        reduction = 100.0 * (baseline_error - error) / baseline_error
    return Result(
        dataset=dataset.name,
        task=dataset.task,
        processor=spec.name,
        tokenizer=spec.tokenizer,
        model=spec.model_name,
        num_rows=context.size(0) + query.size(0),
        num_context_rows=context.size(0),
        num_query_rows=query.size(0),
        num_text_columns=dataset.num_text_columns,
        primary_metric=primary_metric,
        primary_value=primary,
        secondary_metric=secondary_metric,
        secondary_value=secondary,
        primary_error=error,
        error_reduction_vs_none_pct=reduction,
        runtime_s=runtime,
    )


def _write_results(results: list[Result]) -> None:
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=tuple(Result.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def _ranks(results: list[Result]) -> dict[tuple[str, str], int]:
    ranks: dict[tuple[str, str], int] = {}
    for dataset_name in DATASET_NAMES:
        embedding_results = sorted(
            (
                result
                for result in results
                if result.dataset == dataset_name and result.model
            ),
            key=lambda result: result.primary_error,
        )
        for rank, result in enumerate(embedding_results, start=1):
            ranks[(dataset_name, result.processor)] = rank
    return ranks


def _print_results(results: list[Result]) -> None:
    for dataset_name in DATASET_NAMES:
        dataset_results = [
            result for result in results if result.dataset == dataset_name
        ]
        metric = dataset_results[0].primary_metric
        print(f"\n{dataset_name} ({metric})")
        print(
            f"  {'processor':28} {'tokenizer':16} "
            f"{metric:>10} {'error reduction':>16} {'seconds':>9}"
        )
        for result in dataset_results:
            reduction = result.error_reduction_vs_none_pct
            reduction_text = (
                "baseline" if reduction is None else f"{reduction:+.1f}%"
            )
            print(
                f"  {result.processor:28} {result.tokenizer:16} "
                f"{result.primary_value:10.4f} "
                f"{reduction_text:>16} {result.runtime_s:9.2f}"
            )

    ranks = _ranks(results)
    print("\nEmbedding model summary")
    print(
        f"  {'processor':28} {'tokenizer':12} "
        f"{'mean rank':>10} {'mean error reduction':>22}"
    )
    for spec in PROCESSORS:
        if spec.kind != "embedding":
            continue
        model_results = [
            result for result in results if result.processor == spec.name
        ]
        mean_rank = statistics.mean(
            ranks[(result.dataset, result.processor)]
            for result in model_results
        )
        mean_reduction = statistics.mean(
            cast(float, result.error_reduction_vs_none_pct)
            for result in model_results
        )
        print(
            f"  {spec.name:28} {spec.tokenizer:12} "
            f"{mean_rank:10.2f} {mean_reduction:+21.1f}%"
        )

    print("\nTokenizer-family summary (descriptive, not tokenizer-isolated)")
    print(f"  {'tokenizer':12} {'mean rank':>10} {'mean error reduction':>22}")
    for tokenizer in ("WordPiece", "BPE"):
        family_results = [
            result for result in results if result.tokenizer == tokenizer
        ]
        mean_rank = statistics.mean(
            ranks[(result.dataset, result.processor)]
            for result in family_results
        )
        mean_reduction = statistics.mean(
            cast(float, result.error_reduction_vs_none_pct)
            for result in family_results
        )
        print(f"  {tokenizer:12} {mean_rank:10.2f} {mean_reduction:+21.1f}%")
    print("\nLower mean rank and higher error reduction are better.")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda")
    rng = np.random.default_rng(SEED)
    datasets = [_load_dataset(name, rng) for name in DATASET_NAMES]
    model = sdm.models.TabICLv2(device=device)
    baseline_errors: dict[str, float] = {}
    results: list[Result] = []

    for spec in PROCESSORS:
        print(f"\nProcessor: {spec.model_name or spec.name}")
        embedding: sp.SentenceTransformer | None = None
        if spec.kind == "embedding":
            embedding = sp.SentenceTransformer(
                spec.model_name,
                batch_size=BATCH_SIZE,
            ).to(device)
            embedding.eval()

        for dataset in datasets:
            result = _run(
                model,
                dataset,
                spec,
                embedding,
                baseline_errors.get(dataset.name),
                device=device,
            )
            if spec.kind == "none":
                baseline_errors[dataset.name] = result.primary_error
            results.append(result)
            _write_results(results)
            reduction = result.error_reduction_vs_none_pct
            reduction_text = (
                "baseline" if reduction is None else f"{reduction:+.1f}%"
            )
            print(
                f"  {dataset.name}: {result.primary_metric} "
                f"{result.primary_value:.4f}, {reduction_text}, "
                f"{result.runtime_s:.2f}s"
            )

        del embedding
        gc.collect()
        torch.cuda.empty_cache()

    _print_results(results)
    print(f"\nResults written to {OUTPUT}")


if __name__ == "__main__":
    main()
