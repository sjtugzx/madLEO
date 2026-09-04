"""Cross-format SLR validation: independent formats must agree on the same passes.

Purpose (WP4 / root-cause fix A4): the MERIT-II monthly merged files are
parsed by a spec-driven parser (``processors.slr_formats.parse_merit_monthly_file``).
This experiment validates that parser against *independently specified*
SLR products covering the same period: for every normal-point record of the
reference family, find the nearest same-station record of the MERIT family
and compare ranges.

Design note (measured on the legacy raw archive, 2026-08-28): for jason-1
the planned CRD-vs-MERIT 2008 pairing does not exist -- the MERIT family
ends 2007-11 and the CRD family starts 2008-08 -- so the executable
dual-format comparison in this archive is MERIT-II full-rate vs the
historic pre-CRD ILRS '99999' normal-point format over their 37 shared
months.  The comparison is exact-epoch for most stations (the NP epoch is
the window's anchor shot, present in the full-rate stream), so the median
|delta range| is the NP compression scatter: mm level.  The script detects
family coverage first and reports the executed (and the unavailable)
pairings honestly.

Output: ``results/tables/slr_crossformat_validation.csv`` with one row per
(compared family pair, station) plus summary rows, and a PASS/FAIL verdict
against the acceptance bound (median |delta range| <= 0.05 m).

Example invocation from the repository root::

    python scripts/experiments/run_experiments.py slr-crossformat-check \
        --raw-root data/raw/reference_validation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data_pipeline"))

from benchmarking.config import REPO_ROOT  # noqa: E402
from processors.slr_formats import detect_slr_format  # noqa: E402
from processors.slr_processor import slr_to_dataframe  # noqa: E402

DEFAULT_RAW_ROOT = Path(
    os.environ.get(
        "MADLEO_RAW_ARCHIVE",
        str(REPO_ROOT / "data" / "raw" / "reference_validation"),
    )
)
DEFAULT_SAT = "jason-1"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "slr_crossformat_validation.csv"
ACCEPTANCE_MEDIAN_ABS_DELTA_RANGE_M = 0.05  # 5 cm; measured medians are 2-13 mm
MATCH_TOLERANCE_S = 5.0
EXACT_MATCH_S = 1.0e-3


def _classify_files(slr_dir: Path) -> Dict[str, List[Path]]:
    """Group an SLR directory's files by CONTENT family."""
    families: Dict[str, List[Path]] = {}
    for path in sorted(slr_dir.glob("*")):
        if not path.is_file():
            continue
        try:
            with open(path, "r", errors="replace") as handle:
                first_lines = [handle.readline() for _ in range(10)]
        except OSError:
            continue
        families.setdefault(detect_slr_format(first_lines), []).append(path)
    return families


def _load_family(files: List[Path]) -> pd.DataFrame:
    """Parse a family's files into a slim epoch/station/range frame."""
    frames = []
    for path in files:
        df, _ = slr_to_dataframe(str(path))
        if df.empty:
            continue
        slim = pd.DataFrame({
            "station_id": df["station_id"].astype(str),
            "epoch": pd.to_datetime(df["epoch"]),
            "range_m": pd.to_numeric(df["range_m"], errors="coerce"),
            "record_type": df["record_type"].astype(str),
        })
        frames.append(slim.dropna(subset=["range_m"]))
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values("epoch").reset_index(drop=True)


def _month_of(frame: pd.DataFrame) -> pd.Series:
    return frame["epoch"].dt.strftime("%Y%m")


def _nearest_match_deltas(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    """Nearest same-station candidate for each reference record.

    Returns a frame indexed like ``reference`` with ``dt_s`` and
    ``abs_delta_range_m`` (NaN where the station has no candidate data).
    """
    dt_out = np.full(len(reference), np.nan)
    dr_out = np.full(len(reference), np.nan)
    ref_stations = reference["station_id"].to_numpy()
    ref_times = reference["epoch"].astype("int64").to_numpy()
    ref_ranges = reference["range_m"].to_numpy()
    for station in np.unique(ref_stations):
        cand = candidate[candidate["station_id"] == station]
        if cand.empty:
            continue
        cand = cand.sort_values("epoch")
        cand_times = cand["epoch"].astype("int64").to_numpy()
        cand_ranges = cand["range_m"].to_numpy()
        sel = ref_stations == station
        rt = ref_times[sel]
        idx = np.searchsorted(cand_times, rt)
        idx = np.clip(idx, 1, len(cand_times) - 1)
        choose_left = np.abs(cand_times[idx - 1] - rt) < np.abs(cand_times[idx] - rt)
        idx = idx - choose_left.astype(int)
        dt_out[sel] = np.abs(cand_times[idx] - rt) / 1e9
        dr_out[sel] = np.abs(cand_ranges[idx] - ref_ranges[sel])
    return pd.DataFrame({"dt_s": dt_out, "abs_delta_range_m": dr_out})


def _pair_summary(
    left_family: str,
    right_family: str,
    months: List[str],
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> Tuple[List[dict], dict]:
    """Station-level rows + overall verdict for one family pair."""
    rows: List[dict] = []
    matched = _nearest_match_deltas(right, left)
    keep = matched["dt_s"] <= MATCH_TOLERANCE_S
    exact = matched["dt_s"] <= EXACT_MATCH_S
    for station in sorted(set(right["station_id"]) & set(left["station_id"])):
        sel = (right["station_id"] == station).to_numpy() & keep.to_numpy()
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        dr = matched["abs_delta_range_m"].to_numpy()[sel]
        dt = matched["dt_s"].to_numpy()[sel]
        ex_sel = (right["station_id"] == station).to_numpy() & exact.to_numpy()
        rows.append({
            "comparison": f"{right_family}_vs_{left_family}",
            "months": ";".join(months),
            "station_id": station,
            "reference_records": int((right["station_id"] == station).sum()),
            "matched_records": n_sel,
            "exact_epoch_records": int(ex_sel.sum()),
            "median_dt_s": float(np.median(dt)),
            "median_abs_delta_range_m": float(np.median(dr)),
            "max_abs_delta_range_m": float(np.max(dr)),
        })
    if not rows:
        return rows, {"executed": False}
    all_dr = matched["abs_delta_range_m"].to_numpy()[keep.to_numpy()]
    overall = {
        "executed": True,
        "comparison": f"{right_family}_vs_{left_family}",
        "matched_records": int(keep.sum()),
        "median_dt_s": float(np.median(matched["dt_s"].to_numpy()[keep.to_numpy()])),
        "median_abs_delta_range_m": float(np.median(all_dr)),
        "max_abs_delta_range_m": float(np.max(all_dr)),
    }
    overall["verdict"] = (
        "PASS"
        if overall["median_abs_delta_range_m"] <= ACCEPTANCE_MEDIAN_ABS_DELTA_RANGE_M
        else "FAIL"
    )
    return rows, overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                        help="legacy raw archive root (default: $OLD_RAW)")
    parser.add_argument("--sat", default=DEFAULT_SAT, help="satellite directory name")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output CSV path")
    parser.add_argument("--months", nargs="*", default=None,
                        help="restrict to YYYYMM months (default: all shared months)")
    args = parser.parse_args()

    slr_dir = args.raw_root / args.sat / "slr"
    if not slr_dir.is_dir():
        print(f"SLR directory not found: {slr_dir}")
        return 1

    families = _classify_files(slr_dir)
    print(f"family inventory for {args.sat} ({slr_dir}):")
    coverage: Dict[str, dict] = {}
    for family, files in sorted(families.items()):
        if family not in {"crd", "legacy_npt", "merit_monthly"} or not files:
            continue
        frame = _load_family(files)
        months = sorted(_month_of(frame).unique())
        coverage[family] = {"frame": frame, "months": months}
        print(f"  {family:14s} {len(files):4d} files  {len(frame):9d} records  "
              f"{months[0]}..{months[-1]}")

    rows: List[dict] = []
    summaries: List[dict] = []

    # Preferred pairing per the task plan: CRD vs MERIT in a shared period.
    if "crd" in coverage and "merit_monthly" in coverage:
        shared = sorted(set(coverage["crd"]["months"]) & set(coverage["merit_monthly"]["months"]))
        if shared:
            crd = _restrict(coverage["crd"]["frame"], shared, args.months)
            merit = _restrict(coverage["merit_monthly"]["frame"], shared, args.months)
            pair_rows, overall = _pair_summary("merit_monthly", "crd", shared, merit, crd)
            rows.extend(pair_rows)
            summaries.append(overall)
        else:
            summaries.append({
                "executed": False,
                "comparison": "crd_vs_merit_monthly",
                "note": (f"no shared month in this archive: CRD covers "
                         f"{coverage['crd']['months'][0]}..{coverage['crd']['months'][-1]}, "
                         f"MERIT covers {coverage['merit_monthly']['months'][0]}.."
                         f"{coverage['merit_monthly']['months'][-1]}"),
            })
            print("NOTE: CRD and MERIT families share no month in this archive; "
                  "running MERIT vs legacy normal points instead (same-period, "
                  "independent format spec).")

    # Always run the executable independent-format comparison when both exist.
    if "merit_monthly" in coverage and "legacy_npt" in coverage:
        shared = sorted(set(coverage["merit_monthly"]["months"])
                        & set(coverage["legacy_npt"]["months"]))
        if shared:
            merit = _restrict(coverage["merit_monthly"]["frame"], shared, args.months)
            legacy = _restrict(coverage["legacy_npt"]["frame"], shared, args.months)
            pair_rows, overall = _pair_summary("merit_monthly", "legacy_npt", shared, merit, legacy)
            rows.extend(pair_rows)
            summaries.append(overall)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    print(f"\nper-station table written to {args.output}")
    for summary in summaries:
        if summary.get("executed"):
            print(f"{summary['comparison']}: matched {summary['matched_records']} records, "
                  f"median dt = {summary['median_dt_s']:.4f} s, "
                  f"median |delta range| = {summary['median_abs_delta_range_m']*1e3:.2f} mm, "
                  f"max = {summary['max_abs_delta_range_m']*1e3:.2f} mm -> {summary['verdict']}")
        else:
            print(f"{summary['comparison']}: not executed ({summary.get('note', 'no overlap')})")

    failed = [s for s in summaries if s.get("verdict") == "FAIL"]
    executed = [s for s in summaries if s.get("executed")]
    if not executed:
        print("no comparison could be executed")
        return 1
    return 1 if failed else 0


def _restrict(frame: pd.DataFrame, shared_months: List[str],
              requested: Optional[List[str]]) -> pd.DataFrame:
    months = [m for m in (requested or shared_months) if m in shared_months]
    if not months:
        return frame.iloc[0:0]
    sel = _month_of(frame).isin(months)
    return frame[sel].reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
