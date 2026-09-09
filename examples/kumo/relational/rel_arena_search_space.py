"""Shared RelArena policy: text OFF or context-fitted PCA32."""

import argparse
import runpy
from typing import cast

from relarena.search_space import SearchSpace

DEFAULT = {
    "num_neighbors": [16, 16],
    "text_pca_components": 0,
    "regression_transform": "standard",
    "cache_text": True,
    "text_cache_max_bytes": 32 * 1024**3,
    "recent_context_pool": False,
    "max_keys": 20_000,
    "entity_timestamps": "retain",
    "temporal_strategy": "last",
    "raw_event_lags": True,
}

# Default first: the two candidates differ only in whether text is used.
SEARCH_SPACE = SearchSpace(
    default_overrides=DEFAULT,
    fixed_grid=[
        {**DEFAULT, "text_pca_components": components}
        for components in (0, 32)
    ],
)


def parse_search_space(argv: list[str]) -> tuple[SearchSpace, list[str]]:
    """Load an optional Python policy and retain native RelArena arguments."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--search-space",
        metavar="FILE.py",
        help="Python file exporting SEARCH_SPACE (default: bundled policy)",
    )
    args, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    space = (
        SEARCH_SPACE
        if args.search_space is None
        else cast(
            SearchSpace, runpy.run_path(args.search_space)["SEARCH_SPACE"]
        )
    )
    return space, remaining
