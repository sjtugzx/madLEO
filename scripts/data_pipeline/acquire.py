"""Unified data acquisition CLI.

Dispatches to the download orchestrators in `downloaders/`. Run from the
repository root, e.g. `python scripts/acquire.py orbit-bulk --execute`.
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "ids": "downloaders.download_ids_maneuver_histories",
    "tle": "downloaders.download_batch_tle_history",
    "orbit-bulk": "downloaders.download_reference_orbit_bulk",
    "orbit-windows": "downloaders.download_reference_orbit_windows",
    "orbit-samples": "downloaders.download_reference_orbit_samples",
    "slr": "downloaders.download_reference_slr",
    "benchmark-data": "downloaders.download_benchmark_data",
    "starlink": "downloaders.starlink_acquisition",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS), help="acquisition task to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to the task")
    args = parser.parse_args()

    module = importlib.import_module(COMMANDS[args.command])
    sys.argv = [args.command, *args.args]
    result = module.main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
