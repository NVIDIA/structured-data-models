r"""Evaluate SDM tabular model results on BeyondArena."""

from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata

from benchmark.tabular.system import MODEL_CONFIGS, SDMModel

benchmark_dir = Path(__file__).parent.parent
result_root = benchmark_dir / "beyondarena_out"
output_root = benchmark_dir / "beyondarena_evals"

runs = []
for model_config in MODEL_CONFIGS.values():
    result_dir = result_root / model_config.name
    if next(result_dir.rglob("results.pkl"), None) is not None:
        runs.append((model_config, result_dir))

if not runs:
    raise FileNotFoundError(
        f"No BeyondArena results found under {result_root}"
    )

base_context = BeyondArenaContext()
methods = []
method_names = []
for model_config, result_dir in runs:
    output_dir = output_root / model_config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    method_metadata = MethodMetadata.config(
        method=model_config.name,
        ag_key=SDMModel.ag_key,
        model_key=model_config.name,
        config_default=f"{SDMModel.ag_name}_c1_BAG_L1",
        can_hpo=False,
        is_bag=True,
        compute="gpu",
        artifact_dir=output_dir / "artifacts",
    )
    processed = EndToEnd.from_path_raw(
        path_raw=result_dir,
        method_metadata=method_metadata,
        task_metadata=base_context.task_metadata_collection,
        backend="native",
    )
    results = processed.get_results(new_result_prefix="[SDM] ")
    results.to_csv(
        output_dir / "method_results_per_split.csv",
        index=False,
    )
    methods.extend(
        processed.to_method_metadata_lst(new_result_prefix="[SDM] ")
    )
    method_names.extend(results["method"].unique().tolist())

context = BeyondArenaContext.from_new_methods(methods)
leaderboard = context.compare(
    output_dir=output_root,
    only_valid_tasks=method_names,
)
website = context.leaderboard_to_website_format(leaderboard)
print(website.to_string(index=False))
