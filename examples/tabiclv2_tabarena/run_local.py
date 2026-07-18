"""Run local TabICLv2 on the complete TabArena suite in one process."""

from __future__ import annotations

import argparse

from examples.tabiclv2_tabarena.runner import (
    add_common_arguments,
    config_from_args,
    run,
)


def main() -> None:
    """Run the local, in-process TabArena example."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    run(config_from_args(parser.parse_args()), debug_mode=True)


if __name__ == "__main__":
    main()
