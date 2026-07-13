from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from .harness import Task

RUN_GPU_PARITY = os.environ.get("SDM_RUN_TABICLV2_GPU_PARITY") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_GPU_PARITY or not torch.cuda.is_available(),
    reason="set SDM_RUN_TABICLV2_GPU_PARITY=1 on a CUDA host",
)

EXPECTED_SHA256 = {
    "classification": (
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
    ),
    "regression": (
        "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
    ),
}


def _checkpoint(task: str) -> Path:
    variable = f"TABICLV2_{task.upper()}_CHECKPOINT"
    configured = os.environ.get(variable)
    if configured is not None:
        path = Path(configured)
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        variant = "classifier" if task == "classification" else "regressor"
        try:
            path = Path(
                hf_hub_download(
                    repo_id="jingang/TabICL",
                    filename=f"tabicl-{variant}-v2-20260212.ckpt",
                    local_files_only=True,
                )
            )
        except LocalEntryNotFoundError as error:
            pytest.skip(f"pinned checkpoint is not cached: {error}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256[task]
    return path


def _input(task: str) -> tuple[Tensor, Tensor]:
    features = torch.tensor(
        [
            [
                [0.1, -1.0, 2.0],
                [0.2, -0.5, 1.5],
                [-0.1, 0.0, 1.0],
                [0.4, 0.5, 0.5],
                [-0.3, 1.0, 0.0],
                [0.6, 1.5, -0.5],
                [0.0, 2.0, -1.0],
                [0.8, 2.5, -1.5],
            ]
        ],
        device="cuda",
    )
    if task == "classification":
        target = torch.tensor(
            [[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]],
            device="cuda",
        )
    else:
        target = torch.tensor(
            [[-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]],
            device="cuda",
        )
    return features, target


def _capture(module: torch.nn.Module, output: dict[str, Tensor], key: str):
    def hook(
        owner: torch.nn.Module,
        inputs: tuple[Any, ...],
        value: Tensor,
    ) -> None:
        del owner, inputs
        output[key] = value.detach()

    return module.register_forward_hook(hook)


def _max_errors(expected: Tensor, actual: Tensor) -> tuple[float, float]:
    absolute = (expected - actual).abs()
    relative = absolute / expected.abs().clamp_min(
        torch.finfo(expected.dtype).tiny
    )
    return absolute.max().item(), relative.max().item()


def _models(task: str):
    from sdm.models.tabiclv2.model import TabICLv2, _remap_ckpt
    from tabicl import TabICL

    checkpoint = torch.load(
        _checkpoint(task),
        map_location="cpu",
        weights_only=True,
    )
    reference = TabICL(**checkpoint["config"]).eval().cuda()
    reference.load_state_dict(checkpoint["state_dict"])
    candidate = TabICLv2(pretrained=False, device="cuda").eval()
    candidate_core = (
        candidate.cls_model
        if task == "classification"
        else candidate.reg_model
    )
    candidate_core.load_state_dict(
        _remap_ckpt(
            checkpoint["state_dict"],
            is_classifier=task == "classification",
        )
    )
    return reference, candidate_core


def _inference_config():
    from tabicl import InferenceConfig
    from tabicl.model.inference_config import MgrConfig

    def manager() -> MgrConfig:
        return MgrConfig(
            device="cuda",
            use_amp=False,
            use_fa3=False,
            offload="gpu",
        )

    return InferenceConfig(
        COL_CONFIG=manager(),
        ROW_CONFIG=manager(),
        ICL_CONFIG=manager(),
    )


@pytest.mark.parametrize(
    ("task", "row_atol", "icl_atol", "head_atol"),
    [
        ("classification", 7e-6, 6e-7, 4e-6),
        ("regression", 9e-6, 4e-6, 5e-6),
    ],
)
def test_pinned_core_model_stage_parity_on_gpu(
    task: str,
    row_atol: float,
    icl_atol: float,
    head_atol: float,
) -> None:
    from sdm.models.tabiclv2.model import TabICLv2, _remap_ckpt
    from tabicl import TabICL

    checkpoint = torch.load(
        _checkpoint(task),
        map_location="cpu",
        weights_only=True,
    )
    reference = TabICL(**checkpoint["config"]).eval().cuda()
    reference.load_state_dict(checkpoint["state_dict"])
    candidate = TabICLv2(pretrained=False, device="cuda").eval()
    candidate_core = (
        candidate.cls_model
        if task == "classification"
        else candidate.reg_model
    )
    candidate_core.load_state_dict(
        _remap_ckpt(
            checkpoint["state_dict"],
            is_classifier=task == "classification",
        )
    )

    reference_stages: dict[str, Tensor] = {}
    candidate_stages: dict[str, Tensor] = {}
    handles = [
        _capture(reference.row_interactor, reference_stages, "row"),
        _capture(reference.icl_predictor.ln, reference_stages, "icl"),
        _capture(reference.icl_predictor.decoder, reference_stages, "head"),
        _capture(candidate_core.row_embedding, candidate_stages, "row"),
        _capture(candidate_core.icl_block.norm, candidate_stages, "icl"),
        _capture(candidate_core.head, candidate_stages, "head"),
    ]
    features, float_target = _input(task)
    config = _inference_config()

    with torch.inference_mode():
        reference_output = reference(
            X=features,
            y_train=float_target,
            return_logits=True,
            inference_config=config,
        )
        candidate_output = candidate_core(
            x=features,
            y=(
                float_target.long()
                if task == "classification"
                else float_target
            ),
        )
    for handle in handles:
        handle.remove()

    # The reference ICL/head hooks expose all rows; its public output and SDM
    # expose the two query rows only.
    reference_stages["icl"] = reference_stages["icl"][:, -2:]
    reference_stages["head"] = reference_stages["head"][:, -2:]
    tolerances = {"row": row_atol, "icl": icl_atol, "head": head_atol}
    for stage, atol in tolerances.items():
        expected = reference_stages[stage]
        actual = candidate_stages[stage]
        max_absolute, max_relative = _max_errors(expected, actual)
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=1e-5,
            msg=(
                f"{stage}: max_absolute={max_absolute}, "
                f"max_relative={max_relative}"
            ),
        )

    if task == "classification":
        candidate_output = candidate_output[..., : reference_output.size(-1)]
    torch.testing.assert_close(
        candidate_output,
        reference_output,
        atol=head_atol,
        rtol=1e-5,
    )

    candidate_target = (
        float_target.long() if task == "classification" else float_target
    )
    with torch.inference_mode():
        non_cached = candidate(features, candidate_target)
        candidate.fit(features[:, :6], candidate_target)
        cached = candidate.predict(features[:, 6:])
        repeated_prediction = candidate.predict(features[:, 6:])
        candidate.fit(features[:, :6], candidate_target)
        repeated_fit = candidate.predict(features[:, 6:])

    torch.testing.assert_close(
        cached,
        non_cached,
        atol=5e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        repeated_prediction,
        cached,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        repeated_fit,
        cached,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_pinned_single_member_pipeline_end_to_end(task: Task) -> None:
    import numpy as np
    import pandas as pd
    from sdm import TableTensor
    from sdm.models.tabiclv2.recipe import default_recipe
    from sdm.processing import InvertibleMixin
    from sklearn.preprocessing import StandardScaler

    from .harness import (
        AGGREGATED_OUTPUT,
        FINAL_MODEL_INPUT,
        FINAL_OUTPUT,
        RAW_MODEL_OUTPUT,
        TARGET_INVERSE_OUTPUT,
        TARGET_TRANSFORMATION,
        USER_PREDICTION,
        EnsembleMemberPlan,
        record_output_stages,
        trace_reference_member,
        trace_sdm_member,
    )

    train = pd.DataFrame(
        {
            "a": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "b": [3.0, 1.0, 2.0, 0.0, -1.0, -2.0],
            "c": [0.0, 1.0, 0.5, 1.5, -0.5, 2.0],
        }
    )
    test = pd.DataFrame(
        {"a": [0.25, 4.0], "b": [0.25, -3.0], "c": [0.75, 2.5]}
    )
    target_values = (
        np.array(["a", "b", "c", "a", "b", "c"])
        if task == "classification"
        else np.array([-4.0, -1.0, 0.0, 1.0, 4.0, 9.0])
    )
    member = EnsembleMemberPlan(
        member_index=0,
        normalization="none",
        feature_permutation=(0, 1, 2),
        class_permutation=(0, 1, 2) if task == "classification" else None,
        target_transformation=(
            "label_encode+class_permute"
            if task == "classification"
            else "standard_scale"
        ),
        processor_seeds=(("random_state", 42),),
        model_output_type=(
            "logits" if task == "classification" else "raw_quantiles"
        ),
        aggregation_space=(
            "canonical_logits"
            if task == "classification"
            else "original_target"
        ),
    )
    reference_trace = trace_reference_member(
        train_features=train,
        test_features=test,
        target=target_values,
        member=member,
        task=task,
        seed=42,
    )
    all_features = TableTensor.from_pandas(
        pd.concat([train, test], ignore_index=True),
        dict.fromkeys(train.columns, "numerical"),
    )
    target_table = TableTensor.from_pandas(
        pd.DataFrame({"target": target_values}),
        {
            "target": (
                "categorical" if task == "classification" else "numerical"
            )
        },
    )
    candidate_trace, fitted_recipe = trace_sdm_member(
        recipe=default_recipe(),
        features=all_features,
        target=target_table,
        member=member,
    )

    reference_x = reference_trace.snapshots[FINAL_MODEL_INPUT].values
    candidate_x = candidate_trace.snapshots[FINAL_MODEL_INPUT].values
    assert reference_x is not None
    assert candidate_x is not None
    torch.testing.assert_close(
        candidate_x,
        reference_x,
        atol=2e-7,
        rtol=1e-7,
    )
    reference_y = reference_trace.snapshots[TARGET_TRANSFORMATION].values
    candidate_y = candidate_trace.snapshots[TARGET_TRANSFORMATION].values
    assert reference_y is not None
    assert candidate_y is not None

    reference, candidate_core = _models(task)
    with torch.inference_mode():
        reference_raw = reference(
            X=reference_x.unsqueeze(0).cuda(),
            y_train=reference_y.float().unsqueeze(0).cuda(),
            return_logits=True,
            inference_config=_inference_config(),
        ).squeeze(0)
        candidate_raw = candidate_core(
            x=candidate_x.unsqueeze(0).cuda(),
            y=(
                candidate_y.long().unsqueeze(0).cuda()
                if task == "classification"
                else candidate_y.unsqueeze(0).cuda()
            ),
        ).squeeze(0)

    if task == "classification":
        reference_canonical = reference_raw
    else:
        scaler = StandardScaler().fit(target_values.reshape(-1, 1))
        reference_canonical = reference_raw * float(scaler.scale_[0]) + float(
            scaler.mean_[0]
        )
    if task == "classification":
        # Classification is reconstructed from the fitted target categories;
        # it is deliberately not passed through target.inverse_transform.
        candidate_canonical = candidate_raw[..., : len(target_values) // 2]
    else:
        fitted_recipe.target.to("cuda")
        target_inverse = cast(InvertibleMixin, fitted_recipe.target)
        candidate_canonical = target_inverse.inverse_transform(
            TableTensor.from_tensor(candidate_raw)
        ).numerical
    reference_aggregate = reference_canonical
    candidate_aggregate = candidate_canonical
    if task == "classification":
        reference_final = (reference_aggregate / 0.9).softmax(dim=-1)
        reference_user = reference_final.argmax(dim=-1)
    else:
        reference_final = reference_aggregate
        reference_user = reference_final
    fitted_recipe.output.to("cuda")
    candidate_final = fitted_recipe.output.transform(
        TableTensor.from_tensor(candidate_aggregate)
    ).numerical
    candidate_user = (
        candidate_final.argmax(dim=-1)
        if task == "classification"
        else candidate_final
    )
    record_output_stages(
        reference_trace,
        raw_model_output=reference_raw,
        target_inverse_output=reference_canonical,
        aggregated_output=reference_aggregate,
        final_output=reference_final,
        user_prediction=reference_user,
    )
    record_output_stages(
        candidate_trace,
        raw_model_output=candidate_raw,
        target_inverse_output=candidate_canonical,
        aggregated_output=candidate_aggregate,
        final_output=candidate_final,
        user_prediction=candidate_user,
    )

    if task == "classification":
        assert reference_trace.snapshots[RAW_MODEL_OUTPUT].shape == (2, 3)
        assert candidate_trace.snapshots[RAW_MODEL_OUTPUT].shape == (2, 10)
    else:
        torch.testing.assert_close(
            candidate_raw,
            reference_raw,
            atol=1e-5,
            rtol=1e-5,
        )
    for stage in (
        TARGET_INVERSE_OUTPUT,
        AGGREGATED_OUTPUT,
        FINAL_OUTPUT,
        USER_PREDICTION,
    ):
        expected = reference_trace.snapshots[stage].values
        actual = candidate_trace.snapshots[stage].values
        assert expected is not None
        assert actual is not None
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-5,
            rtol=2e-5,
        )
