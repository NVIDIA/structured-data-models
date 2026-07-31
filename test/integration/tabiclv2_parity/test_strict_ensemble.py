from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from sdm.models.tabiclv2.recipe import default_recipe
from sdm.processing import TargetDecode

from .harness import (
    Task,
    assert_preprocessing_parity,
    candidate_model,
    candidate_tables,
    checkpoint,
    parity_data,
    require_reference,
    strict_parity_enabled,
)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_eight_member_preprocessing_parity(task: Task) -> None:
    assert_preprocessing_parity(task)


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
    reference_member_outputs = []
    class_shuffles = []
    for normalization, (features, targets) in reference_data.items():
        if task == "classification":
            feature_shuffles = reference.ensemble_generator_.feature_shuffles_[
                normalization
            ]
            reference_member_outputs.append(
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
            reference_member_outputs.append(
                reference._batch_forward(
                    features,
                    targets,
                    output_type="quantiles",
                    alphas=np.linspace(0.001, 0.999, 999).tolist(),
                )
            )
    reference_members = np.concatenate(reference_member_outputs, axis=0)
    if task == "classification":
        reference_members = np.stack(
            [
                output[..., shuffle]
                for output, shuffle in zip(reference_members, class_shuffles)
            ]
        )
    else:
        reference_members = reference.y_scaler_.inverse_transform(
            reference_members.reshape(-1, 1)
        ).reshape(reference_members.shape)
    reference_to_candidate = assert_preprocessing_parity(task)

    model = candidate_model(task, path)
    x_context, target, x_query = candidate_tables(task, device="cuda")
    direct: dict[str, np.ndarray] = {}
    cached: dict[str, np.ndarray] = {}
    for mode in ("parallel", "sequential"):
        output = model(
            x_context,
            target,
            x_query,
            num_estimators=8,
            ensemble_mode=mode,
            generator=torch.Generator().manual_seed(42),
        ).numerical
        if task == "regression":
            output = output.mean(dim=-1)
        direct[mode] = output.detach().cpu().numpy()

        model.fit(
            x_context,
            target,
            num_estimators=8,
            ensemble_mode=mode,
            generator=torch.Generator().manual_seed(42),
        )
        output = model.predict(x_query).numerical
        if task == "regression":
            output = output.mean(dim=-1)
        cached[mode] = output.detach().cpu().numpy()
        model.clear()

    member_recipe = default_recipe()
    member_recipe.output = TargetDecode()
    member_direct: dict[str, np.ndarray] = {}
    member_cached: dict[str, np.ndarray] = {}
    for mode in ("parallel", "sequential"):
        member_direct[mode] = (
            model(
                x_context,
                target,
                x_query,
                recipe=member_recipe,
                num_estimators=8,
                ensemble_mode=mode,
                generator=torch.Generator().manual_seed(42),
            )
            .numerical.detach()
            .cpu()
            .numpy()
        )
        model.fit(
            x_context,
            target,
            recipe=member_recipe,
            num_estimators=8,
            ensemble_mode=mode,
            generator=torch.Generator().manual_seed(42),
        )
        member_cached[mode] = (
            model.predict(x_query).numerical.detach().cpu().numpy()
        )
        model.clear()

    member_atol = 3e-4 if task == "classification" else 2e-4
    member_rtol = 4e-4 if task == "classification" else 8e-4
    path_atol = 1e-5 if task == "classification" else 5e-5
    path_rtol = 1e-5 if task == "classification" else 3e-4
    for mode in ("parallel", "sequential"):
        for name, candidate_members in (
            ("direct", member_direct[mode]),
            ("cached", member_cached[mode]),
        ):
            np.testing.assert_allclose(
                candidate_members[list(reference_to_candidate)],
                reference_members,
                atol=member_atol,
                rtol=member_rtol,
                err_msg=(
                    f"first divergence: member-level {name} ({mode}, {task})"
                ),
            )
        np.testing.assert_allclose(
            member_direct[mode],
            member_cached[mode],
            atol=path_atol,
            rtol=path_rtol,
        )
    np.testing.assert_allclose(
        member_direct["parallel"],
        member_direct["sequential"],
        atol=path_atol,
        rtol=path_rtol,
    )
    np.testing.assert_allclose(
        member_cached["parallel"],
        member_cached["sequential"],
        atol=path_atol,
        rtol=path_rtol,
    )

    atol = 3e-5 if task == "classification" else 1e-4
    for mode in ("parallel", "sequential"):
        np.testing.assert_allclose(
            direct[mode],
            expected,
            atol=atol,
            rtol=1e-4,
            err_msg=f"first divergence: public forward ({mode})",
        )
        np.testing.assert_allclose(
            cached[mode],
            expected,
            atol=atol,
            rtol=1e-4,
            err_msg=f"first divergence: fit/predict ({mode})",
        )
    np.testing.assert_allclose(
        direct["parallel"],
        direct["sequential"],
        atol=1e-5,
        rtol=1e-5,
    )

    del reference, model
    gc.collect()
    torch.cuda.empty_cache()
