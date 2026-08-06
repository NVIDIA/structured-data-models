from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
from test.integration.tabiclv2_parity.harness import (
    Task,
    assert_preprocessing_parity,
    candidate_model,
    candidate_tables,
    checkpoint,
    parity_data,
    require_reference,
    strict_parity_enabled,
)

from sdm import Stype
from sdm.models.tabiclv2.recipe import default_recipe


@pytest.mark.parametrize("recipe_execution", ["vectorized", "sequential"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_eight_member_preprocessing_parity(
    task: Task,
    recipe_execution: str,
) -> None:
    assert recipe_execution in {"sequential", "vectorized"}
    assert_preprocessing_parity(
        task,
        recipe_execution=recipe_execution,  # type: ignore[arg-type]
    )


@pytest.mark.skipif(
    not strict_parity_enabled() or not torch.cuda.is_available(),
    reason="set SDM_RUN_TABICLV2_GPU_PARITY=1 on a CUDA host",
)
@pytest.mark.cuda
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_eight_member_public_forward_and_fit_predict_parity(
    task: Task,
) -> None:
    tabicl = require_reference()

    path = checkpoint(task)
    data = parity_data(task)
    common = {
        "n_estimators": 8,
        "random_state": 42,
        "model_path": path,
        "allow_auto_download": False,
        "device": "cuda",
        "use_amp": False,
        "use_fa3": False,
        "offload_mode": "gpu",
        "batch_size": 8,
    }
    if task == "classification":
        reference = tabicl.TabICLClassifier(**common)
        reference.fit(data.x_context, data.target)
        expected = reference.predict_proba(data.x_query)
    else:
        reference = tabicl.TabICLRegressor(**common)
        reference.fit(data.x_context, data.target)
        expected = reference.predict(data.x_query, output_type="mean")

    reference_data = reference.ensemble_generator_.transform(
        data.x_query,
        mode="both",
    )
    reference_raw_parts = []
    class_shuffles = []
    for normalization, (features, targets) in reference_data.items():
        if task == "classification":
            feature_shuffles = reference.ensemble_generator_.feature_shuffles_[
                normalization
            ]
            reference_raw_parts.append(
                reference._batch_forward(
                    features,
                    targets,
                    feature_shuffles,
                )
            )
            class_shuffles.extend(
                reference.ensemble_generator_.class_shuffles_[normalization]
            )
        else:
            reference_raw_parts.append(
                reference._batch_forward(
                    features,
                    targets,
                    output_type="quantiles",
                    alphas=np.linspace(0.001, 0.999, 999).tolist(),
                )
            )
    reference_raw = np.concatenate(reference_raw_parts, axis=0)
    if task == "classification":
        reference_mapped = np.stack(
            [
                output[..., permutation]
                for output, permutation in zip(
                    reference_raw,
                    class_shuffles,
                    strict=True,
                )
            ]
        )
    else:
        reference_mapped = reference.y_scaler_.inverse_transform(
            reference_raw.reshape(-1, 1)
        ).reshape(reference_raw.shape)
    reference_reduced = reference_mapped.mean(axis=0)
    reference_final = (
        reference.softmax(
            reference_reduced,
            axis=-1,
            temperature=reference.softmax_temperature,
        )
        if task == "classification"
        else reference_reduced
    )

    reference_to_candidate = assert_preprocessing_parity(
        task,
        recipe_execution="vectorized",
    )
    model = candidate_model(task, path)
    x_context, target, x_query = candidate_tables(task, device="cuda")
    execution = default_recipe()._bind_members(
        x_context=x_context,
        y_context=target,
        related_context_tables=None,
        member_ids=tuple(range(8)),
        num_members_total=8,
        generator=torch.Generator().manual_seed(42),
    )
    queries = execution.transform(
        x_query=x_query,
        related_query_tables=None,
    )
    raw_tables = []
    with torch.inference_mode():
        for context, query in zip(
            execution.contexts,
            queries,
            strict=True,
        ):
            raw_tables.append(
                model._forward(
                    x_context=context.x,
                    y_context=context.y,
                    x_query=query.x,
                    related_context_tables=None,
                    related_query_tables=None,
                    cache=None,
                    generator=None,
                )
            )
    candidate_raw = np.stack(
        [table.numerical.detach().cpu().numpy() for table in raw_tables]
    )
    member_atol = 3e-4 if task == "classification" else 2e-4
    member_rtol = 4e-4 if task == "classification" else 8e-4
    np.testing.assert_allclose(
        candidate_raw[list(reference_to_candidate)],
        reference_raw,
        atol=member_atol,
        rtol=member_rtol,
        err_msg="first divergence: per-estimator model output",
    )

    if task == "classification":
        canonical_columns = tuple(
            str(value) for value in sorted(np.unique(data.target))
        )
        mapped_tables = raw_tables
        candidate_mapped = np.stack(
            [
                table.numerical[
                    ...,
                    [
                        table.columns[Stype.numerical].index(column)
                        for column in canonical_columns
                    ],
                ]
                .detach()
                .cpu()
                .numpy()
                for table in raw_tables
            ]
        )
    else:
        mapped_tables = list(execution.inverse_transform_target(raw_tables))
        candidate_mapped = np.stack(
            [table.numerical.detach().cpu().numpy() for table in mapped_tables]
        )
    np.testing.assert_allclose(
        candidate_mapped[list(reference_to_candidate)],
        reference_mapped,
        atol=member_atol,
        rtol=member_rtol,
        err_msg="first divergence: canonical output mapping",
    )
    np.testing.assert_allclose(
        candidate_mapped.mean(axis=0),
        reference_reduced,
        atol=member_atol,
        rtol=member_rtol,
        err_msg="first divergence: estimator reduction",
    )
    final_stage = execution.recipe.output.transform(
        torch.stack(mapped_tables, dim=0)
    )
    if task == "classification":
        assert final_stage.columns[Stype.numerical] == canonical_columns
    np.testing.assert_allclose(
        final_stage.numerical.detach().cpu().numpy(),
        reference_final,
        atol=member_atol,
        rtol=member_rtol,
        err_msg="first divergence: final output processing",
    )

    direct: dict[str, np.ndarray] = {}
    cached: dict[str, np.ndarray] = {}
    for recipe_execution in ("vectorized", "sequential"):
        output = model(
            x_context,
            target,
            x_query,
            num_estimators=8,
            recipe_execution=recipe_execution,
            generator=torch.Generator().manual_seed(42),
        ).numerical
        if task == "regression":
            output = output.mean(dim=-1)
        direct[recipe_execution] = output.detach().cpu().numpy()

        model.fit(
            x_context,
            target,
            num_estimators=8,
            recipe_execution=recipe_execution,
            generator=torch.Generator().manual_seed(42),
        )
        output = model.predict(x_query).numerical
        if task == "regression":
            output = output.mean(dim=-1)
        cached[recipe_execution] = output.detach().cpu().numpy()
        model.clear()

    atol = 3e-5 if task == "classification" else 1e-4
    for recipe_execution in ("vectorized", "sequential"):
        np.testing.assert_allclose(
            direct[recipe_execution],
            expected,
            atol=atol,
            rtol=1e-4,
            err_msg=f"first divergence: public forward ({recipe_execution})",
        )
        np.testing.assert_allclose(
            cached[recipe_execution],
            expected,
            atol=atol,
            rtol=1e-4,
            err_msg=f"first divergence: fit/predict ({recipe_execution})",
        )
    np.testing.assert_allclose(
        direct["vectorized"],
        direct["sequential"],
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        cached["vectorized"],
        cached["sequential"],
        atol=1e-5,
        rtol=1e-5,
    )

    del reference, model
    gc.collect()
    torch.cuda.empty_cache()
