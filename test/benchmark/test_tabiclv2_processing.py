import json

import torch
from benchmark.tabiclv2_processing import (
    Characteristics,
    benchmark_pipeline,
    build_workload,
    full_characteristics,
    write_results,
)
from sdm.testing import withCUDA


def test_full_characteristic_matrix_has_every_binary_combination() -> None:
    matrix = tuple(full_characteristics())

    assert len(matrix) == 128
    assert len(set(matrix)) == 128
    assert Characteristics() in matrix
    assert (
        Characteristics(
            constant=True,
            hard_outlier=True,
            sigma_outlier=True,
            categorical=True,
            missing=True,
            unknown_category=True,
            high_cardinality=True,
        )
        in matrix
    )


def test_pipeline_benchmark_validates_unknown_categories_and_writes_results(
    tmp_path,
) -> None:
    workload = build_workload(
        size="tiny",
        task="classification",
        characteristics=Characteristics(
            categorical=True,
            missing=True,
            unknown_category=True,
        ),
    )

    results = benchmark_pipeline(workload, repetitions=2, detailed=False)
    output = tmp_path / "results.json"
    write_results(results, output)

    payload = json.loads(output.read_text())
    reference_commit = payload["reference_commit"]
    assert isinstance(reference_commit, str)
    assert len(reference_commit) == 40
    assert set(reference_commit) <= set("0123456789abcdef")
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["correctness_status"] == "pass"
    assert result["operation"] == "total_recipe_overhead"
    assert result["peak_memory_bytes"] is None
    assert output.with_suffix(".csv").is_file()


def test_high_cardinality_workload_contains_unseen_query_values() -> None:
    workload = build_workload(
        size="tiny",
        task="classification",
        characteristics=Characteristics(
            categorical=True,
            unknown_category=True,
            high_cardinality=True,
        ),
    )

    categorical = workload.x.categorical
    assert categorical.categories[0].numel() == 257
    query_codes = categorical.as_tensor()[workload.train_rows :]
    assert torch.equal(query_codes, torch.full_like(query_codes, 256))


@withCUDA
def test_pipeline_benchmark_records_device_and_cuda_memory(
    device: torch.device,
) -> None:
    workload = build_workload(
        size="tiny",
        task="classification",
        characteristics=Characteristics(),
        device=device,
    )

    result = benchmark_pipeline(
        workload,
        repetitions=2,
        detailed=False,
        recipe_variant="power",
    )[0]

    assert result.device == str(device)
    if device.type == "cuda":
        assert result.gpu_model == torch.cuda.get_device_name(device)
        assert result.peak_memory_bytes is not None
    else:
        assert result.gpu_model is None
        assert result.peak_memory_bytes is None


def test_regression_benchmark_includes_all_output_stages() -> None:
    workload = build_workload(
        size="tiny",
        task="regression",
        characteristics=Characteristics(hard_outlier=True),
    )

    results = benchmark_pipeline(workload, repetitions=2, detailed=True)

    assert {result.operation for result in results} == {
        "feature_fit",
        "feature_transform",
        "feature_fit_transform",
        "target_fit",
        "target_transform",
        "target_fit_transform",
        "preprocessing_total",
        "target_inverse_transform",
        "model_output_inverse_mapping",
        "output_transform",
        "total_recipe_overhead",
    }
    assert all(result.correctness_status == "pass" for result in results)
