"""Apply the shipped evidence QC-flag columns to an already-built release tree.

The final pre-submission review found two shipped-evidence anomalies that
must carry explicit flags (rows are MARKED, never dropped -- Ruling-17):

- 20 SLR normal points with implausible one-way ranges / cross-target
  contamination (sentinel-3a: 5 COMPASS-I6B rows + 6 station-7825 rows on
  2024-09-03; sentinel-3b: 9 station-7825 rows on 2024-09-03) -- the record
  triage already marks them ``qc_rejected`` internally, but the flag was
  stripped at parquet write; this script resolves them into the shipped
  ``qc_status`` vocabulary via the single shared implementation
  (``processors.physical_qc.slr_qc_status``).
- 5,861 Starlink ephemeris states (3 actively deorbiting objects) with
  |r| below the Earth's radius -- flagged ``below_surface`` in a new
  ``quality_flag`` column.

A full pipeline re-run reproduces these columns natively (the staging paths
now write them); this script is the one-shot patch for the already-built
tree so the release does not need a multi-day re-parse.  It also refreshes
``schema_audit.csv`` (ephemeris flag disclosure) and regenerates
``manifest.json`` via ``release_export.stage_manifest``.

Idempotent: re-running on an already-patched tree recomputes identical
columns (and a new manifest timestamp).  Run from the repository root::

    python3 scripts/data_pipeline/alignment/patch_evidence_flags.py \
        --root dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Flat-package bootstrap (same as the other alignment modules).
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from benchmarking.experiment_params import EARTH_RADIUS_KM  # noqa: E402
from processors.physical_qc import slr_qc_status  # noqa: E402

SLR_ROOT = Path("mission_reported") / "evidence" / "slr"
STARLINK_ROOT = Path("operational") / "starlink"


def patch_slr(root: Path) -> list[tuple[str, int, int, int]]:
    """Add/refresh ``qc_status`` on every SLR evidence parquet.

    Returns per-file ``(file, ok, range_implausible, cross_target)`` counts.
    The status is recomputed from the shipped range/target columns with the
    shared implementation -- no triage state is needed (no other Class-A
    assertion fires on the current release), so the patch and a full
    re-stage produce identical values.
    """
    rows: list[tuple[str, int, int, int]] = []
    for path in sorted((root / SLR_ROOT).glob("*.parquet")):
        frame = pd.read_parquet(path)
        status = slr_qc_status(frame)
        if "qc_status" in frame.columns and frame["qc_status"].equals(status):
            counts = status.value_counts()
            rows.append((path.stem, int(counts.get("ok", 0)),
                         int(counts.get("range_implausible", 0)),
                         int(counts.get("cross_target", 0))))
            print(f"  slr {path.stem}: qc_status already present, unchanged")
            continue
        frame["qc_status"] = status
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
        tmp.replace(path)
        counts = status.value_counts()
        flagged = len(frame) - int(counts.get("ok", 0))
        rows.append((path.stem, int(counts.get("ok", 0)),
                     int(counts.get("range_implausible", 0)),
                     int(counts.get("cross_target", 0))))
        print(f"  slr {path.stem}: {len(frame)} rows, {flagged} flagged "
              f"(range_implausible={counts.get('range_implausible', 0)}, "
              f"cross_target={counts.get('cross_target', 0)})")
    return rows


def _starlink_flag(table: pa.Table) -> pa.Array:
    """below_surface flag array for an ephemeris table (Arrow, no pandas copy)."""
    radius2 = pc.add(
        pc.add(pc.multiply(table["x_m"], table["x_m"]),
               pc.multiply(table["y_m"], table["y_m"])),
        pc.multiply(table["z_m"], table["z_m"]),
    )
    radius = pc.sqrt(radius2)
    below = pc.less(radius, pa.scalar(EARTH_RADIUS_KM * 1000.0))
    return pc.if_else(below, pa.scalar("below_surface"), pa.scalar(""))


def patch_starlink(root: Path) -> None:
    """Add/refresh ``quality_flag`` on the Starlink ephemeris parquets."""
    for name in ("ephemeris_state.parquet", "ephemeris_state_sample25.parquet"):
        path = root / STARLINK_ROOT / name
        if not path.exists():
            continue
        table = pq.read_table(path)
        flag = _starlink_flag(table)
        if "quality_flag" in table.column_names:
            if table["quality_flag"].equals(flag):
                print(f"  starlink {name}: quality_flag already present, unchanged "
                      f"({pc.sum(pc.equal(flag, 'below_surface')).as_py()} flagged)")
                continue
            table = table.drop_columns(["quality_flag"])
        table = table.append_column("quality_flag", flag)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(path)
        flagged = int(pc.sum(pc.equal(flag, "below_surface")).as_py())
        print(f"  starlink {name}: {table.num_rows} rows, {flagged} below_surface")


def patch_schema_audit(root: Path) -> None:
    """Refresh the ephemeris row of experiments/starlink/schema_audit.csv."""
    # The audit table lives with the code (tracked), not inside the dataset
    # tree; anchor it to the repository root, not the --root argument.
    repo_root = _PIPELINE_ROOT.parent.parent
    audit_path = repo_root / "experiments" / "starlink" / "schema_audit.csv"
    if not audit_path.exists():
        print(f"  schema_audit: {audit_path} not found, skipped")
        return
    audit = pd.read_csv(audit_path)
    eph = pq.read_table(root / STARLINK_ROOT / "ephemeris_state.parquet")
    flagged = int(pc.sum(pc.equal(eph["quality_flag"], "below_surface")).as_py())
    mask = audit["table"] == "ephemeris_state.parquet"
    if "quality_flag_rows" not in audit.columns:
        audit["quality_flag_rows"] = 0
    audit.loc[mask, "quality_flag_rows"] = flagged
    audit.loc[mask, "quality_flags"] = (
        "ok" if flagged == 0 else f"ok ({flagged} rows flagged; see quality_flag column)"
    )
    audit.to_csv(audit_path, index=False)
    print(f"  schema_audit.csv: ephemeris quality_flag_rows={flagged}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="dataset", help="release tree root (default: dataset)")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"release root not found: {root}")

    print(f"[patch] SLR qc_status under {root / SLR_ROOT}")
    patch_slr(root)
    print(f"[patch] Starlink quality_flag under {root / STARLINK_ROOT}")
    patch_starlink(root)
    patch_schema_audit(root)

    print("[patch] regenerating manifest.json")
    from alignment.release_export import stage_manifest
    stage_manifest(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
