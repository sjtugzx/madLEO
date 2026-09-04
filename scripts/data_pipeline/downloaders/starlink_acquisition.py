"""Helpers and CLI for Starlink raw acquisition into the operational layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path


def build_starlink_target_paths() -> Dict[str, Path]:
    """Return the canonical raw/intermediate Starlink acquisition paths."""
    return {
        "tle_archive": REPO_ROOT / "data/raw/operational_constellation/starlink/tle_archive",
        "ephemeris": REPO_ROOT / "data/raw/operational_constellation/starlink/ephemeris",
        "candidate_labels": REPO_ROOT / "data/raw/operational_constellation/starlink/candidate_labels",
        "metadata": REPO_ROOT / "data/validation/download_status/benchmark/starlink/acquisition",
    }


def copy_candidate_labels_if_present(source_path: Path | None, target_dir: Path) -> Path | None:
    """Copy a legacy Starlink candidate-label file into the canonical raw layout."""
    if source_path is None or not source_path.exists():
        return None
    ensure_directory(target_dir)
    target_path = target_dir / source_path.name
    if source_path.resolve() == target_path.resolve():
        return target_path
    shutil.copy2(source_path, target_path)
    return target_path


def write_probe_metadata(path: Path, payload: Dict) -> None:
    """Persist acquisition probe metadata as JSON."""
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Starlink raw acquisition layout")
    parser.add_argument(
        "--legacy-candidate-labels",
        default=None,
        help="Optional legacy candidate-label file to copy into the canonical raw layout",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only create directories and write probe metadata placeholders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_starlink_target_paths()
    for path in paths.values():
        ensure_directory(path)

    copied_candidate = copy_candidate_labels_if_present(
        resolve_repo_path(args.legacy_candidate_labels) if args.legacy_candidate_labels else None,
        paths["candidate_labels"],
    )

    payload = {
        "sync_probe": {"status": "pending"},
        "spacetrack_probe": {"status": "pending"},
        "candidate_labels": {
            "legacy_source": args.legacy_candidate_labels or "",
            "copied_to": str(copied_candidate) if copied_candidate else "",
        },
        "probe_only": bool(args.probe_only),
    }
    write_probe_metadata(paths["metadata"] / "probe_metadata.json", payload)

    print("Prepared Starlink acquisition layout:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    if copied_candidate:
        print(f"Copied legacy candidate labels to {copied_candidate}")
    else:
        print("No legacy candidate labels copied")


if __name__ == "__main__":
    main()
