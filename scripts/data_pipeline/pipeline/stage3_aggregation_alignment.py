"""Stage 3: Data Aggregation and Alignment (paper Methods).

Aggregation merges processed records per satellite; alignment evaluates
TLE/precise-orbit/SLR coverage over every event window and assigns the
confidence tier. Applies to the reference subset only in the alignment part;
the operational subset only aggregates (rolling ephemeris overwrite rule).
"""

from __future__ import annotations

import os
import sys


def run_aggregation_alignment(subset: str, dry_run: bool = True) -> list[list[str]]:
    """Return (and optionally run) the aggregation/alignment command plan."""
    if subset == "reference":
        tables_dir = os.environ.get("TABLES_DIR", "results/tables")
        plan = [
            [
                "python", "scripts/data_pipeline/process.py", "audit-windows",
                "--scope", "all", "--local-only",
                "--output", f"{tables_dir}/reference_all_window_alignment_status.csv",
            ],
            ["python", "scripts/data_pipeline/process.py", "annotation-alignment", "--scope", "all", "--batch", "A", "--batch", "B"],
        ]
    elif subset == "operational":
        # Aggregation for Starlink is part of normalization (latest-wins
        # overwrite of overlapping ephemeris points, stage 2); alignment is
        # skipped because the operational subset carries no reported events.
        plan = []
    else:
        raise ValueError(f"unknown subset: {subset}")

    if not plan:
        print(f"[skip] no aggregation/alignment steps for subset={subset}", flush=True)
    for command in plan:
        print(("[dry-run] " if dry_run else "[run] ") + " ".join(command), flush=True)
    if not dry_run:
        import subprocess

        for command in plan:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"aggregation/alignment step failed: {' '.join(command)}")
    return plan


def main() -> int:
    subset = os.environ.get("SUBSET", "reference")
    dry_run = os.environ.get("DRY_RUN", "1") != "0"
    run_aggregation_alignment(subset, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
