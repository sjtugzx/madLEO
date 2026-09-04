"""Stage 2: Data Processing (paper Methods: Data Processing).

Parses raw provider files into structured records under the unified schema
(ISO-8601 UTC timestamps ending in Z, `_m`/`_mps`/`_rad` suffixes,
`mean_motion_rad_per_min`). The parsing itself lives in `processors/` and
`benchmarking/normalization.py`; this stage only sequences the per-subset
normalization entry points. All output roots come from the environment so
tests can point them at a sandbox.
"""

from __future__ import annotations

import os
import sys


def run_processing(subset: str, dry_run: bool = True) -> list[list[str]]:
    """Return (and optionally run) the processing command plan for a subset."""
    if subset == "reference":
        plan = [
            ["python", "scripts/data_pipeline/process.py", "normalize-annotations"],
        ]
    elif subset == "operational":
        plan = [
            ["python", "scripts/data_pipeline/process.py", "normalize-starlink"],
        ]
    else:
        raise ValueError(f"unknown subset: {subset}")

    for command in plan:
        print(("[dry-run] " if dry_run else "[run] ") + " ".join(command), flush=True)
    if not dry_run:
        import subprocess

        for command in plan:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"processing step failed: {' '.join(command)}")
    return plan


def main() -> int:
    subset = os.environ.get("SUBSET", "reference")
    dry_run = os.environ.get("DRY_RUN", "1") != "0"
    run_processing(subset, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
