from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import TabICLv2
from sdm.models.tabiclv2.model import _remap_ckpt
from sdm.models.tabiclv2.recipe import default_recipe

Task = Literal["classification", "regression"]

EXPECTED_SHA256 = {
    "classification": (
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
    ),
    "regression": (
        "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
    ),
}
EXPECTED_TABICL_VERSION = "2.0.0"
EXPECTED_TABICL_COMMIT = "f719c886a586ed4a29236345e319ac1ea596c478"


def strict_parity_enabled() -> bool:
    return os.environ.get("SDM_RUN_TABICLV2_GPU_PARITY") == "1"


def require_reference() -> ModuleType:
    """Load and verify the optional pinned TabICLv2 reference."""
    try:
        tabicl = importlib.import_module("tabicl")
    except ImportError as error:
        if strict_parity_enabled():
            pytest.fail(
                f"strict parity requires the 'tabicl' package: {error}"
            )
        pytest.skip("optional 'tabicl' reference is not installed")

    version = importlib.metadata.version("tabicl")
    if version != EXPECTED_TABICL_VERSION:
        pytest.fail(
            "strict parity requires TabICL "
            f"{EXPECTED_TABICL_VERSION}, found {version}"
        )

    if strict_parity_enabled():
        module_file = tabicl.__file__
        if module_file is None:
            pytest.fail("strict parity requires a filesystem-backed reference")
        source = Path(module_file).resolve()
        root = next(
            (
                parent
                for parent in source.parents
                if (parent / ".git").exists()
            ),
            None,
        )
        if root is None:
            pytest.fail(
                "strict parity requires an editable checkout of the pinned "
                "TabICLv2 reference"
            )
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != EXPECTED_TABICL_COMMIT:
            pytest.fail(
                "strict parity requires TabICLv2 commit "
                f"{EXPECTED_TABICL_COMMIT}, found {commit}"
            )
    return tabicl


@dataclass(frozen=True)
class ParityData:
    x_context: np.ndarray
    x_query: np.ndarray
    target: np.ndarray


def parity_data(task: Task) -> ParityData:
    """Return a small table with four retained, skewed features."""
    x_context = np.array(
        [
            [-8.0, 0.0, 1.0, 2.0],
            [-3.0, 1.0, 1.5, 4.0],
            [-1.0, 2.0, 2.0, 8.0],
            [0.0, 4.0, 3.0, 16.0],
            [1.0, 8.0, 5.0, 32.0],
            [3.0, 16.0, 8.0, 64.0],
            [8.0, 32.0, 13.0, 128.0],
            [21.0, 64.0, 21.0, 256.0],
        ],
        dtype=np.float32,
    )
    x_query = np.array(
        [[-2.0, 3.0, 2.5, 12.0], [13.0, 48.0, 18.0, 192.0]],
        dtype=np.float32,
    )
    target = (
        np.array([10, 20, 30, 10, 20, 30, 10, 20])
        if task == "classification"
        else np.array(
            [-4.0, -1.0, 0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
            dtype=np.float32,
        )
    )
    return ParityData(x_context, x_query, target)


def candidate_tables(
    task: Task,
    *,
    device: torch.device | str,
) -> tuple[TableTensor, TableTensor, TableTensor]:
    """Convert the canonical parity data to SDM tables."""
    data = parity_data(task)
    columns = ("x0", "x1", "x2", "x3")
    x_context = TableTensor.from_tensor(
        torch.tensor(data.x_context, device=device),
        columns=columns,
    )
    x_query = TableTensor.from_tensor(
        torch.tensor(data.x_query, device=device),
        columns=columns,
    )
    if task == "regression":
        target = TableTensor.from_tensor(
            torch.tensor(data.target, device=device).unsqueeze(-1)
        )
    else:
        target = TableTensor(
            columns={Stype.categorical: ("target",)},
            categorical=CategoricalTensor(
                code=torch.tensor(
                    [1, 2, 0, 1, 2, 0, 1, 2],
                    device=device,
                ).unsqueeze(-1),
                categories=(torch.tensor([30, 10, 20, 40], device=device),),
            ),
        )
    return x_context, target, x_query


def assert_preprocessing_parity(task: Task) -> tuple[int, ...]:
    """Compare all eight member inputs at their first semantic divergence."""
    require_reference()
    # These optional packages are installed only for strict parity runs.
    from sklearn.preprocessing import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        LabelEncoder,
        StandardScaler,
    )
    from tabicl.sklearn.preprocessing import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        EnsembleGenerator,
    )

    data = parity_data(task)
    if task == "classification":
        reference_target = LabelEncoder().fit_transform(data.target)
    else:
        reference_target = StandardScaler().fit_transform(
            data.target.reshape(-1, 1)
        )[:, 0]
    reference = EnsembleGenerator(
        classification=task == "classification",
        n_estimators=8,
        norm_methods=["none", "power"],
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        random_state=42,
    ).fit(data.x_context.astype(np.float64), reference_target)
    reference_data = reference.transform(
        data.x_query.astype(np.float64),
        mode="both",
    )

    x_context, target, x_query = candidate_tables(task, device="cpu")
    recipe = default_recipe()
    candidate_context, candidate_target, _ = recipe.fit_transform(
        x_context,
        target,
        num_members=8,
        generator=torch.Generator().manual_seed(42),
    )
    candidate_query, _ = recipe.transform(x_query)

    matched: set[int] = set()
    reference_to_candidate: list[int] = []
    for normalization, (features, targets) in reference_data.items():
        for local, (permutation, _) in enumerate(
            reference.ensemble_configs_[normalization]
        ):
            expected_target = torch.from_numpy(targets[local])
            candidates = []
            for member in range(8):
                member_normalization = "none" if member % 2 == 0 else "power"
                columns = candidate_context[member].columns[Stype.numerical]
                member_permutation = tuple(
                    int(column.removeprefix("x")) for column in columns
                )
                member_target = (
                    candidate_target[member].categorical.code.squeeze(-1)
                    if task == "classification"
                    else candidate_target[member].numerical.squeeze(-1)
                )
                target_matches = torch.allclose(
                    member_target,
                    expected_target.to(member_target.dtype),
                    atol=2e-6 if task == "regression" else 0,
                    rtol=2e-6 if task == "regression" else 0,
                )
                if (
                    member not in matched
                    and member_normalization == normalization
                    and member_permutation == tuple(permutation)
                    and target_matches
                ):
                    candidates.append(member)
            assert len(candidates) == 1
            member = candidates[0]
            matched.add(member)
            reference_to_candidate.append(member)

            actual_features = torch.cat(
                (
                    candidate_context[member].numerical,
                    candidate_query[member].numerical,
                ),
                dim=-2,
            )
            tolerance = 2e-7 if normalization == "none" else 2e-4
            torch.testing.assert_close(
                actual_features,
                torch.from_numpy(features[local]).to(torch.float32),
                atol=tolerance,
                rtol=4e-4 if normalization == "power" else 2e-7,
                msg=(
                    "first divergence: feature preprocessing for "
                    f"{normalization=}, {permutation=}"
                ),
            )
    assert matched == set(range(8))
    return tuple(reference_to_candidate)


def checkpoint(task: Task) -> Path:
    """Resolve and verify the pinned local TabICLv2 checkpoint."""
    # Checkpoint tooling is optional outside strict parity runs.
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.utils import (  # noqa: PLC0415
        LocalEntryNotFoundError,
    )

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
        pytest.fail(f"strict parity checkpoint is not cached: {error}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256[task]
    return path


def candidate_model(task: Task, path: Path) -> TabICLv2:
    """Load one pinned checkpoint into the corresponding SDM core."""
    state = torch.load(path, map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    model = TabICLv2(pretrained=False, device="cuda")
    classification = task == "classification"
    core = model.cls_model if classification else model.reg_model
    core.load_state_dict(_remap_ckpt(state, is_classifier=classification))
    return model
