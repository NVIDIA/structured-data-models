"""Two-hop neighbor, target, and context-pool grid with Qwen/PCA64.

Pass this file with --search-space and use --n-trials 16 for the model
script. The system script evaluates every entry. The default is included
first, not added as a seventeenth trial. Target transformation is inactive
for classification, so its sixteen entries have eight distinct settings.
"""

from itertools import product

from examples.kumo.relational.rel_arena_search_space import (
    DEFAULT as BASE_DEFAULT,
)
from relarena.search_space import SearchSpace

DEFAULT = {**BASE_DEFAULT, "text_pca_components": 64}

SEARCH_SPACE = SearchSpace(
    default_overrides=DEFAULT,
    fixed_grid=[
        {
            **DEFAULT,
            "num_neighbors": [neighbors, neighbors],
            "regression_transform": transform,
            "recent_context_pool": recent,
        }
        for neighbors, transform, recent in product(
            (16, 1, 4, 32),
            ("standard", "quantile"),
            (False, True),
        )
    ],
)
