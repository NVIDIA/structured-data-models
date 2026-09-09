"""Kumo model submission using RelArena's shared validation tuner.

Use the same declared search space for every task. The candidate policy is
carried from stopped V12 development; complete runtime has not been certified.

Run one task from the repository root, after the setup in README.relarena.md::

    from examples.kumo.relational.rel_arena import KumoModel
    from relarena.runner import run_model_experiment

    result = run_model_experiment(
        KumoModel,
        "rel-f1",
        "driver-position",
        n_trials=30,
        seed=0,
        cache_dir=".cache/relarena/rel-f1-driver-position",
    )
    print(result.tuned.test_score)

cache_dir is optional. Point it at precompute_text.py's output directory to
reuse frozen text embeddings; missing documents are encoded and cached. Omit
it to encode text on demand without a persistent cache. PCA remains fitted
separately on each training context in both cases.

RelArena scores every candidate on full validation, then refits and scores the
winner and default on TEST. The complete per-task budget includes preprocessing
and all trials/refits; a per-trial time limit is not a whole-task limit.
"""

from examples.kumo.relational._relarena.adapter import KumoPredictor
from examples.kumo.relational._relarena.search_space import SEARCH_SPACE
from relarena.registry import register_model

KumoModel = register_model(search_space=SEARCH_SPACE)(KumoPredictor)
