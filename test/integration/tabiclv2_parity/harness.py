from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import torch
from sdm import Stype, TableTensor
from sdm.processing import (
    CategoryShuffle,
    Choice,
    FeaturePermute,
    Identity,
    Power,
    Quantile,
    Recipe,
    Sequential,
    TargetDispatch,
)
from torch import Tensor

Task = Literal["classification", "regression"]

RAW_FEATURE_INPUT = "raw_feature_input"
ENCODED_NUMERICAL_FEATURES = "encoded_numerical_features"
FEATURE_PIPELINE_OUTPUT = "feature_pipeline_output"
FEATURE_PERMUTATION = "feature_permutation"
TARGET_ENCODING = "target_encoding"
TARGET_TRANSFORMATION = "target_transformation"
FINAL_MODEL_INPUT = "final_model_input"
RAW_MODEL_OUTPUT = "raw_model_output"
TARGET_INVERSE_OUTPUT = "output_after_target_inverse"
AGGREGATED_OUTPUT = "aggregated_output"
FINAL_OUTPUT = "final_output_postprocessing"
USER_PREDICTION = "user_facing_prediction"


@dataclass(frozen=True)
class EnsembleMemberPlan:
    member_index: int
    normalization: str
    feature_permutation: tuple[int, ...]
    class_permutation: tuple[int, ...] | None
    target_transformation: str
    processor_seeds: tuple[tuple[str, int | None], ...]
    model_output_type: str
    aggregation_space: str

    @property
    def semantic_id(
        self,
    ) -> tuple[str, tuple[int, ...], tuple[int, ...] | None]:
        return (
            self.normalization,
            self.feature_permutation,
            self.class_permutation,
        )


@dataclass(frozen=True)
class EnsemblePlan:
    seed: int
    members: tuple[EnsembleMemberPlan, ...]
    source: str

    def canonical_members(self) -> tuple[EnsembleMemberPlan, ...]:
        return tuple(
            sorted(self.members, key=lambda member: member.semantic_id)
        )


@dataclass(frozen=True)
class StageSnapshot:
    name: str
    shape: tuple[int, ...]
    dtype: str
    values: Tensor | None
    columns: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_table(
        cls,
        name: str,
        table: TableTensor,
        *,
        metadata: dict[str, object] | None = None,
    ) -> StageSnapshot:
        if table.numerical.size(-1) == table.size(-1):
            values = table.numerical.detach().cpu().clone()
            columns = tuple(table.columns[Stype.numerical])
            dtype = str(values.dtype)
        elif table.categorical.size(-1) == table.size(-1):
            values = table.categorical.as_tensor().detach().cpu().clone()
            columns = tuple(table.columns[Stype.categorical])
            dtype = str(values.dtype)
        elif table.size(-1) > 0:
            values = None
            columns = tuple(
                column for stype in Stype for column in table.columns[stype]
            )
            dtype = ",".join(
                f"{stype.value}={getattr(table, stype.value).dtype}"
                for stype in Stype
                if getattr(table, stype.value).size(-1) > 0
            )
        else:
            values = None
            columns = ()
            dtype = "empty"
        return cls(
            name=name,
            shape=tuple(table.size()),
            dtype=dtype,
            values=values,
            columns=columns,
            metadata=_freeze_metadata(metadata),
        )

    @classmethod
    def from_array(
        cls,
        name: str,
        array: np.ndarray | Tensor,
        *,
        columns: Sequence[str] = (),
        metadata: dict[str, object] | None = None,
        model_dtype: torch.dtype | None = None,
    ) -> StageSnapshot:
        if isinstance(array, Tensor):
            values = array.detach().cpu().clone()
            dtype = str(values.dtype)
            shape = tuple(values.shape)
        else:
            shape = tuple(array.shape)
            dtype = str(array.dtype)
            if array.dtype.kind in {"b", "i", "u", "f", "c"}:
                values = torch.from_numpy(np.asarray(array)).clone()
            else:
                values = None
        if values is not None and model_dtype is not None:
            values = values.to(model_dtype)
            dtype = str(values.dtype)
        return cls(
            name=name,
            shape=shape,
            dtype=dtype,
            values=values,
            columns=tuple(columns),
            metadata=_freeze_metadata(metadata),
        )


@dataclass
class StageTrace:
    implementation: str
    member: EnsembleMemberPlan
    snapshots: OrderedDict[str, StageSnapshot] = field(
        default_factory=OrderedDict
    )

    def record(self, snapshot: StageSnapshot) -> None:
        self.snapshots[snapshot.name] = snapshot


@dataclass(frozen=True)
class StageDifference:
    stage: str
    reason: str
    max_absolute: float | None = None
    max_relative: float | None = None


def compare_traces(
    reference: StageTrace,
    candidate: StageTrace,
    *,
    stages: Iterable[str],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> list[StageDifference]:
    differences: list[StageDifference] = []
    for stage in stages:
        expected = reference.snapshots[stage]
        actual = candidate.snapshots[stage]
        if expected.shape != actual.shape:
            differences.append(
                StageDifference(
                    stage,
                    f"shape {expected.shape} != {actual.shape}",
                )
            )
            continue
        if expected.dtype != actual.dtype:
            differences.append(
                StageDifference(
                    stage,
                    f"dtype {expected.dtype} != {actual.dtype}",
                )
            )
            continue
        if expected.columns != actual.columns:
            differences.append(
                StageDifference(
                    stage,
                    f"columns {expected.columns} != {actual.columns}",
                )
            )
            continue
        if expected.metadata != actual.metadata:
            differences.append(
                StageDifference(
                    stage,
                    f"metadata {expected.metadata} != {actual.metadata}",
                )
            )
            continue
        if expected.values is None or actual.values is None:
            if expected.values is not actual.values:
                differences.append(
                    StageDifference(stage, "only one trace has values")
                )
            continue
        if expected.values.dtype.is_floating_point:
            close = torch.isclose(
                expected.values,
                actual.values,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
        else:
            close = expected.values == actual.values
        if bool(close.all()):
            continue
        max_absolute, max_relative = _max_errors(
            expected.values,
            actual.values,
        )
        differences.append(
            StageDifference(
                stage,
                "values differ",
                max_absolute=max_absolute,
                max_relative=max_relative,
            )
        )
    return differences


def first_divergence(
    reference: StageTrace,
    candidate: StageTrace,
    *,
    stages: Iterable[str],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> StageDifference | None:
    differences = compare_traces(
        reference,
        candidate,
        stages=stages,
        atol=atol,
        rtol=rtol,
    )
    return differences[0] if differences else None


def record_output_stages(
    trace: StageTrace,
    *,
    raw_model_output: np.ndarray | Tensor,
    target_inverse_output: np.ndarray | Tensor,
    aggregated_output: np.ndarray | Tensor,
    final_output: np.ndarray | Tensor,
    user_prediction: np.ndarray | Tensor,
) -> None:
    """Record the output half of one member's observable pipeline."""
    trace.record(
        StageSnapshot.from_array(
            RAW_MODEL_OUTPUT,
            raw_model_output,
            metadata={"member_index": trace.member.member_index},
        )
    )
    trace.record(
        StageSnapshot.from_array(
            TARGET_INVERSE_OUTPUT,
            target_inverse_output,
            metadata={"space": trace.member.aggregation_space},
        )
    )
    trace.record(
        StageSnapshot.from_array(
            AGGREGATED_OUTPUT,
            aggregated_output,
            metadata={"space": trace.member.aggregation_space},
        )
    )
    trace.record(StageSnapshot.from_array(FINAL_OUTPUT, final_output))
    trace.record(StageSnapshot.from_array(USER_PREDICTION, user_prediction))


def reference_ensemble_plan(
    *,
    n_features: int,
    n_classes: int | None,
    n_estimators: int,
    task: Task,
    seed: int,
    norm_methods: list[str] | None = None,
    feature_shuffle: str = "latin",
    class_shuffle: str = "shift",
) -> EnsemblePlan:
    from tabicl.sklearn.preprocessing import EnsembleGenerator

    rows = max(4, n_classes or 2)
    x = np.arange(rows * n_features, dtype=np.float64).reshape(
        rows,
        n_features,
    )
    if task == "classification":
        assert n_classes is not None
        y = np.arange(rows) % n_classes
    else:
        y = np.linspace(-1.0, 1.0, rows)
    generator = EnsembleGenerator(
        classification=task == "classification",
        n_estimators=n_estimators,
        norm_methods=norm_methods,
        feat_shuffle_method=feature_shuffle,
        class_shuffle_method=class_shuffle,
        random_state=seed,
    ).fit(x, y)

    members: list[EnsembleMemberPlan] = []
    for normalization, configurations in generator.ensemble_configs_.items():
        for feature_permutation, class_permutation in configurations:
            members.append(
                EnsembleMemberPlan(
                    member_index=len(members),
                    normalization=normalization,
                    feature_permutation=tuple(
                        int(index) for index in feature_permutation
                    ),
                    class_permutation=(
                        None
                        if class_permutation is None
                        else tuple(int(index) for index in class_permutation)
                    ),
                    target_transformation=(
                        "label_encode+class_permute"
                        if task == "classification"
                        else "standard_scale"
                    ),
                    processor_seeds=(("random_state", seed),),
                    model_output_type=(
                        "logits"
                        if task == "classification"
                        else "raw_quantiles"
                    ),
                    aggregation_space=(
                        "canonical_logits"
                        if task == "classification"
                        else "original_target"
                    ),
                )
            )
    return EnsemblePlan(seed=seed, members=tuple(members), source="tabicl")


def sdm_ensemble_plan(
    *,
    recipe: Recipe,
    features: TableTensor,
    target: TableTensor,
    n_estimators: int,
    seed: int,
) -> EnsemblePlan:
    torch.manual_seed(seed)
    context = features[: target.size(-2)]
    members: list[EnsembleMemberPlan] = []
    for member_index in range(n_estimators):
        member_recipe = deepcopy(recipe)
        member_recipe.features.fit(context)
        transformed_target = member_recipe.target.fit_transform(target)
        del transformed_target

        choice = _one_module(member_recipe.features, Choice)
        permutation = _one_module(member_recipe.features, FeaturePermute)
        target_dispatch = _one_module(member_recipe.target, TargetDispatch)
        class_permutation: tuple[int, ...] | None = None
        if target_dispatch.get_extra_state() == "classification":
            shuffle = cast(CategoryShuffle, target_dispatch.selected)
            class_permutation = tuple(
                int(index) for index in shuffle.permutations.tolist()
            )
            task: Task = "classification"
        else:
            task = "regression"

        selected = choice.selected
        normalization = {
            Identity: "none",
            Power: "power",
            Quantile: "quantile",
        }.get(type(selected), selected.__class__.__name__.lower())
        seeds: list[tuple[str, int | None]] = [
            ("torch_global", seed),
            ("choice", None),
            ("feature_permute", None),
            ("category_shuffle", None),
        ]
        if isinstance(selected, Quantile):
            seeds.append(("quantile", selected.random_state))
        members.append(
            EnsembleMemberPlan(
                member_index=member_index,
                normalization=normalization,
                feature_permutation=tuple(
                    int(index) for index in permutation.permutation.tolist()
                ),
                class_permutation=class_permutation,
                target_transformation=(
                    "category_shuffle"
                    if task == "classification"
                    else "standard_scale"
                ),
                processor_seeds=tuple(seeds),
                model_output_type=(
                    "logits" if task == "classification" else "raw_quantiles"
                ),
                aggregation_space=(
                    "canonical_logits"
                    if task == "classification"
                    else "original_target"
                ),
            )
        )
    return EnsemblePlan(seed=seed, members=tuple(members), source="sdm")


def trace_sdm_member(
    *,
    recipe: Recipe,
    features: TableTensor,
    target: TableTensor,
    member: EnsembleMemberPlan,
) -> tuple[StageTrace, Recipe]:
    fitted = deepcopy(recipe)
    n_train = target.size(-2)
    full = features
    context = features[:n_train]
    trace = StageTrace("sdm", member)
    trace.record(StageSnapshot.from_table(RAW_FEATURE_INPUT, full))

    feature_pipeline_recorded = False
    assert isinstance(fitted.features, Sequential)
    for step in fitted.features.steps:
        if isinstance(step, Choice):
            index = _choice_index(step, member.normalization)
            step._index = index
            step.selected.fit(context)
            step._fitted = True
        elif isinstance(step, FeaturePermute):
            step.permutation = torch.tensor(
                member.feature_permutation,
                device=context.device,
            )
            step._fitted = True
        else:
            step.fit(context)

        context = step.transform(context)
        full = step.transform(full)
        if not trace.snapshots.get(ENCODED_NUMERICAL_FEATURES) and (
            full.categorical.size(-1) == 0
        ):
            trace.record(
                StageSnapshot.from_table(ENCODED_NUMERICAL_FEATURES, full)
            )
        if isinstance(step, FeaturePermute):
            trace.record(StageSnapshot.from_table(FEATURE_PERMUTATION, full))
        elif full.categorical.size(-1) == 0:
            trace.snapshots[FEATURE_PIPELINE_OUTPUT] = (
                StageSnapshot.from_table(FEATURE_PIPELINE_OUTPUT, full)
            )
            feature_pipeline_recorded = True

    if not feature_pipeline_recorded:
        trace.record(StageSnapshot.from_table(FEATURE_PIPELINE_OUTPUT, full))

    trace.record(
        StageSnapshot.from_array(
            TARGET_ENCODING,
            _target_tensor(target),
            metadata={"space": "label_codes_or_original_target"},
        )
    )
    transformed_target = fitted.target.fit_transform(target)
    dispatch = _one_module(fitted.target, TargetDispatch)
    if member.class_permutation is not None:
        shuffle = cast(CategoryShuffle, dispatch.selected)
        shuffle.permutations = torch.tensor(
            member.class_permutation,
            device=target.device,
        )
        transformed_target = fitted.target.transform(target)
    trace.record(
        StageSnapshot.from_array(
            TARGET_TRANSFORMATION,
            _target_tensor(transformed_target),
            metadata={
                "space": (
                    "permuted_label_codes"
                    if dispatch.get_extra_state() == "classification"
                    else "standardized_target"
                ),
                "task": dispatch.get_extra_state(),
            },
        )
    )

    model_y = _target_tensor(transformed_target)
    trace.record(
        StageSnapshot.from_array(
            FINAL_MODEL_INPUT,
            full.numerical,
            columns=full.columns[Stype.numerical],
            metadata={
                "target_shape": tuple(model_y.shape),
                "target_dtype": model_y.dtype,
            },
        )
    )
    return trace, fitted


def trace_reference_member(
    *,
    train_features: Any,
    test_features: Any,
    target: np.ndarray,
    member: EnsembleMemberPlan,
    task: Task,
    seed: int,
) -> StageTrace:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from tabicl.sklearn.preprocessing import (
        EnsembleGenerator,
        TransformToNumerical,
    )

    if hasattr(train_features, "columns"):
        raw_columns = tuple(str(column) for column in train_features.columns)
        all_features = pd.concat(
            [train_features, test_features],
            ignore_index=True,
        )
    else:
        raw_columns = tuple(
            str(index) for index in range(np.asarray(train_features).shape[1])
        )
        all_features = np.concatenate(
            [train_features, test_features],
            axis=0,
        )

    trace = StageTrace("tabicl", member)
    raw_array = np.asarray(all_features)
    trace.record(
        StageSnapshot.from_array(
            RAW_FEATURE_INPUT,
            raw_array,
            columns=raw_columns,
        )
    )

    encoder = TransformToNumerical().fit(train_features)
    encoded_train = np.asarray(encoder.transform(train_features))
    encoded_all = np.asarray(encoder.transform(all_features))
    if hasattr(train_features, "columns"):
        categorical = list(
            train_features.select_dtypes(
                include=["string", "object", "category", "boolean"]
            ).columns
        )
        numerical = list(
            train_features.select_dtypes(include="number").columns
        )
        encoded_columns = tuple(
            str(column) for column in [*categorical, *numerical]
        )
    else:
        encoded_columns = raw_columns
    trace.record(
        StageSnapshot.from_array(
            ENCODED_NUMERICAL_FEATURES,
            encoded_all,
            columns=encoded_columns,
        )
    )

    if task == "classification":
        target_encoder = LabelEncoder().fit(target)
        encoded_target = target_encoder.transform(target)
        target_encoding = encoded_target
    else:
        target_encoder = StandardScaler().fit(target.reshape(-1, 1))
        encoded_target = target_encoder.transform(
            target.reshape(-1, 1)
        ).reshape(-1)
        target_encoding = target
    trace.record(
        StageSnapshot.from_array(
            TARGET_ENCODING,
            np.asarray(target_encoding),
            metadata={"space": "label_codes_or_original_target"},
        )
    )

    generator = EnsembleGenerator(
        classification=task == "classification",
        n_estimators=1,
        norm_methods=[member.normalization],
        feat_shuffle_method="none",
        class_shuffle_method="none",
        outlier_threshold=4.0,
        random_state=seed,
    ).fit(encoded_train, encoded_target)

    normalization = member.normalization
    feature_permutation = np.asarray(member.feature_permutation)
    class_permutation = (
        None
        if member.class_permutation is None
        else np.asarray(member.class_permutation)
    )
    generator.ensemble_configs_ = OrderedDict(
        {
            normalization: [
                (feature_permutation, class_permutation),
            ]
        }
    )
    generator.feature_shuffles_ = OrderedDict(
        {normalization: [feature_permutation]}
    )
    if task == "classification":
        assert class_permutation is not None
        generator.class_shuffles_ = OrderedDict(
            {normalization: [class_permutation]}
        )

    filtered_all = generator.unique_filter_.transform(encoded_all)
    preprocessor = generator.preprocessors_[normalization]
    processed_all = np.concatenate(
        [
            preprocessor.X_transformed_,
            preprocessor.transform(filtered_all[len(encoded_train) :]),
        ],
        axis=0,
    )
    retained_indices = np.flatnonzero(
        generator.unique_filter_.features_to_keep_
    )
    retained_columns = tuple(
        encoded_columns[index] for index in retained_indices
    )
    trace.record(
        StageSnapshot.from_array(
            FEATURE_PIPELINE_OUTPUT,
            processed_all,
            columns=retained_columns,
        )
    )
    permuted = processed_all[:, feature_permutation]
    permuted_columns = tuple(
        retained_columns[index] for index in feature_permutation
    )
    trace.record(
        StageSnapshot.from_array(
            FEATURE_PERMUTATION,
            permuted,
            columns=permuted_columns,
        )
    )

    transformed_target = (
        encoded_target
        if class_permutation is None
        else class_permutation[encoded_target.astype(np.int64)]
    )
    trace.record(
        StageSnapshot.from_array(
            TARGET_TRANSFORMATION,
            np.asarray(transformed_target),
            metadata={
                "space": (
                    "permuted_label_codes"
                    if task == "classification"
                    else "standardized_target"
                ),
                "task": task,
            },
        )
    )
    trace.record(
        StageSnapshot.from_array(
            FINAL_MODEL_INPUT,
            permuted,
            columns=permuted_columns,
            metadata={
                "target_shape": tuple(transformed_target.shape),
                "target_dtype": torch.float32,
            },
            model_dtype=torch.float32,
        )
    )
    return trace


def _choice_index(choice: Choice, normalization: str) -> int:
    for index, option in enumerate(choice.options):
        if normalization == "none" and isinstance(option, Identity):
            return index
        if normalization == "quantile" and isinstance(option, Quantile):
            return index
        if normalization == "power" and isinstance(option, Power):
            return index
    raise ValueError(
        f"SDM recipe has no option for normalization '{normalization}'"
    )


def _one_module(root: torch.nn.Module, cls: type[Any]) -> Any:
    modules = [module for module in root.modules() if isinstance(module, cls)]
    if len(modules) != 1:
        raise AssertionError(
            f"Expected one {cls.__name__}, found {len(modules)}"
        )
    return modules[0]


def _target_tensor(target: TableTensor) -> Tensor:
    if target.categorical.size(-1) == 1:
        categorical = target.categorical
        return categorical.as_tensor().squeeze(-1)
    return target.numerical.squeeze(-1)


def _freeze_metadata(
    metadata: dict[str, object] | None,
) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    return tuple(sorted((key, str(value)) for key, value in metadata.items()))


def _max_errors(expected: Tensor, actual: Tensor) -> tuple[float, float]:
    expected_float = expected.to(torch.float64)
    actual_float = actual.to(torch.float64)
    absolute = (expected_float - actual_float).abs()
    denominator = expected_float.abs().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    relative = absolute / denominator
    return float(absolute.max().item()), float(relative.max().item())
