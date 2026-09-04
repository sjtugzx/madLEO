"""Apply the stable-window tail flag to the shipped label tables.

Rule (single definition: ``benchmarking.experiment_params
.STABLE_TAIL_FLAG_THRESHOLD_M``): a stable no-event window whose TLE
bracket |delta_sma| response, as published in
``experiments/validation/stable_window_response_validation.csv``, exceeds
20 m carries the ``suspect_unreported_maneuver`` token in
``quality_flags``.  Flagged windows are retained, not dropped.

This is a release-table mutation kept separate from ``public_labels.py``
because the flag depends on the quantitative response layer (computed by
the stable-windows experiment from the released TLE evidence), while the
label export itself only reshapes the alignment tables.  Run once after
the stable-windows experiment, then rebuild the manifest:

    python scripts/data_pipeline/alignment/apply_stable_tail_flags.py
    python scripts/data_pipeline/process.py export-release --stage manifest --output-dir dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarking.config import REPO_ROOT  # noqa: E402
from benchmarking.experiment_params import STABLE_TAIL_FLAG_THRESHOLD_M  # noqa: E402

FLAG = "suspect_unreported_maneuver"
RESPONSE = REPO_ROOT / "experiments" / "validation" / "stable_window_response_validation.csv"
ANNOTATIONS = REPO_ROOT / "dataset" / "mission_reported" / "annotations"
TABLES = ("stable_windows.csv", "stable_windows_matched.csv")


def main() -> int:
    response = pd.read_csv(RESPONSE)
    flagged = set(
        response.loc[response["delta_sma_km"].abs() * 1000.0 > STABLE_TAIL_FLAG_THRESHOLD_M, "annotation_id"]
    )
    print(f"flag rule: |delta_sma| > {STABLE_TAIL_FLAG_THRESHOLD_M} m -> {len(flagged)} windows", flush=True)
    for name in TABLES:
        path = ANNOTATIONS / name
        table = pd.read_csv(path, dtype=str).fillna("")
        hit = table["annotation_id"].isin(flagged)
        flags = table["quality_flags"]
        add = hit & ~flags.str.contains(FLAG, regex=False)
        table.loc[add, "quality_flags"] = [
            f"{old},{FLAG}" if old else FLAG for old in flags[add]
        ]
        table.to_csv(path, index=False)
        print(f"{name}: {int(hit.sum())} flagged ({int(add.sum())} newly)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
