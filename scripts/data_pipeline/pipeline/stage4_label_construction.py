"""Stage 4: Label Construction (paper Methods). Reference subset only.

Builds the public label vocabulary: event windows with confidence tiers,
stable (no_event) windows via the deterministic grid procedure, and the
release-facing public label export. The operational subset has no reported
events and therefore skips this stage entirely.
"""

from __future__ import annotations

import os
import sys


def run_label_construction(subset: str, dry_run: bool = True) -> list[list[str]]:
    """Return (and optionally run) the label-construction command plan."""
    if subset == "operational":
        print("[skip] operational subset carries no reported events; label construction does not apply", flush=True)
        return []
    if subset != "reference":
        raise ValueError(f"unknown subset: {subset}")

    plan = [
        ["python", "scripts/experiments/run_experiments.py", "stable-windows"],
        ["python", "scripts/data_pipeline/process.py", "export-labels"],
    ]
    for command in plan:
        print(("[dry-run] " if dry_run else "[run] ") + " ".join(command), flush=True)
    if not dry_run:
        import subprocess

        for command in plan:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"label-construction step failed: {' '.join(command)}")
    return plan


def main() -> int:
    subset = os.environ.get("SUBSET", "reference")
    dry_run = os.environ.get("DRY_RUN", "1") != "0"
    run_label_construction(subset, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
