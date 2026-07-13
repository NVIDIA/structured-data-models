import json

from benchmark.tabiclv2_processing import (
    Characteristics,
    benchmark_pipeline,
    build_workload,
    full_characteristics,
    write_results,
)


def test_full_characteristic_matrix_has_every_binary_combination() -> None:
    matrix = tuple(full_characteristics())

    assert len(matrix) == 64
    assert len(set(matrix)) == 64
    assert Characteristics() in matrix
    assert (
        Characteristics(
            constant=True,
            hard_outlier=True,
            sigma_outlier=True,
            categorical=True,
            missing=True,
            unknown_category=True,
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
    assert payload["reference_commit"] == (
        "f719c886a586ed4a29236345e319ac1ea596c478"
    )
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["correctness_status"] == "pass"
    assert result["operation"] == "total_recipe_overhead"
    assert result["peak_memory_bytes"] is None
    assert output.with_suffix(".csv").is_file()


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
