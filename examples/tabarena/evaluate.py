r"""Evaluate an SDM tabular model's results on TabArena."""

import argparse
from pathlib import Path

from model import MODEL_CONFIGS
from tabarena.contexts import TabArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    choices=tuple(MODEL_CONFIGS),
    default="tabiclv2",
    help="SDM model results to evaluate.",
)
args = parser.parse_args()

model_config = MODEL_CONFIGS[args.model]
example_dir = Path(__file__).parent.parent
result_dir = example_dir / "tabarena_out" / model_config.name
output_dir = example_dir / "evals" / model_config.name
output_dir.mkdir(parents=True, exist_ok=True)

base_context = TabArenaContext()
method_metadata = MethodMetadata.baseline(
    method=model_config.method_name,
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
results.to_csv(output_dir / "method_results_per_split.csv", index=False)

methods = processed.to_method_metadata_lst(
    new_result_prefix="[SDM] ",
)
context = TabArenaContext.from_new_methods(methods)
leaderboard = context.compare(output_dir=output_dir)
website = context.leaderboard_to_website_format(leaderboard)
print(website.to_string(index=False))
