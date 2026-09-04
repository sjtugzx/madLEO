"""Unified data processing CLI for the MAD-LEO pipeline.

Dispatches to the processing/alignment modules in `alignment/`. Run from the
repository root, e.g.
`python scripts/data_pipeline/process.py normalize-annotations`.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Flat-package bootstrap: make data_pipeline/ importable as the package root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = {
    "normalize-annotations": "alignment.normalize_maneuver_annotations",
    "normalize-starlink": "alignment.normalize_starlink_operational_subset",
    "audit-windows": "alignment.audit_reference_window_alignment",
    "audit-sources": "alignment.audit_batch_a_evidence_sources",
    "annotation-alignment": "alignment.generate_reference_annotation_alignment",
    "align-evidence": "alignment.align_maneuver_evidence",
    "canonicalize": "alignment.canonicalize_release_schema",
    "export-sources": "alignment.export_normalized_sources",
    "export-labels": "alignment.public_labels",
    "export-release": "alignment.release_export",
    # Release self-consistency gate: recompute coverage from the shipped
    # evidence and diff against the annotation tables; also writes the
    # evidence registry (evidence_sources.csv) and the sentinel-6a GNSS
    # coverage manifest on demand.
    "selfcheck": "alignment.release_selfcheck",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS), help="processing task to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to the task")
    args = parser.parse_args()

    module = importlib.import_module(COMMANDS[args.command])
    sys.argv = [args.command, *args.args]
    result = module.main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
