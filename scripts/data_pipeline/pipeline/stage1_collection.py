"""Stage 1: Data Collection (paper Methods: Data Collection).

Thin orchestration over the acquisition CLIs. Which targets and sources are
collected is driven entirely by environment variables (see params/*.env),
never by hard-coded lists here. Credentials are read from the process
environment, which run_*.sh populates from the git-ignored .env file.
"""

from __future__ import annotations

import os
import sys


def run_collection(subset: str, dry_run: bool = True) -> list[list[str]]:
    """Return (and optionally run) the collection command plan for a subset."""
    if subset == "reference":
        targets = os.environ.get("REFERENCE_TARGETS", "").split()
        plan = [
            ["python", "scripts/data_pipeline/acquire.py", "ids", "--execute"],
            ["python", "scripts/data_pipeline/acquire.py", "tle", "--execute"],
            ["python", "scripts/data_pipeline/acquire.py", "orbit-bulk", "--execute"],
            ["python", "scripts/data_pipeline/acquire.py", "slr", "--execute"],
        ]
        if targets:
            for command in plan[1:]:
                command.extend(["--include", *targets])
    elif subset == "operational":
        plan = [
            ["python", "scripts/data_pipeline/acquire.py", "starlink"],
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
                raise RuntimeError(f"collection step failed: {' '.join(command)}")
    return plan


def main() -> int:
    subset = os.environ.get("SUBSET", "reference")
    dry_run = os.environ.get("DRY_RUN", "1") != "0"
    run_collection(subset, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
