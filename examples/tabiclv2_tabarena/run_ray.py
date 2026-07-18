"""Run local TabICLv2 on TabArena with Ray-backed bagged fold fitting."""

from __future__ import annotations

import argparse

from examples.tabiclv2_tabarena.runner import (
    add_common_arguments,
    config_from_args,
    run,
)


def main() -> None:
    """Run the Ray-backed TabArena example."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--ray-address")
    args = parser.parse_args()

    import ray

    owns_ray_session = not ray.is_initialized()
    if owns_ray_session:
        ray.init(address=args.ray_address)
    try:
        run(config_from_args(args), debug_mode=False)
    finally:
        if owns_ray_session:
            ray.shutdown()


if __name__ == "__main__":
    main()
