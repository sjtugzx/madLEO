"""Analysis-window sensitivity of the TLE SMA-shift response metric.

Reviewer question: are the response metrics sensitive to the analysis-window
definition (the release's asymmetric event_time -6 h / +24 h window vs a
symmetric +/-24 h window)?

For every one of the 1,134 shipped event windows this experiment recomputes
the TLE response exactly as the release does --

    delta_a = a(n_after) - a(n_before),   a(n) = (mu / n^2)^(1/3),

where ``n_before`` / ``n_after`` are the mean motions at the nearest TLE
epochs at-or-before the window start / at-or-after the window end -- twice:

1. RELEASE window:    [event_time - 6 h,  event_time + 24 h]  (the shipped
   ``window_start_utc`` / ``window_end_utc``),
2. SYMMETRIC window:  [event_time - 24 h, event_time + 24 h],

with the event epoch taken as ``window_start_utc + 6 h`` (verified to equal
``maneuver_annotations.csv:event_time_utc`` for all 1,134 windows).  Note the
two definitions share the same window END (event_time + 24 h); only the
before-bracket placement moves, by 18 h.

Inputs are the shipped release tables ONLY (no credentials, no ``data/``
workspace): the annotation table, the per-target TLE evidence parquets, and
the published per-window response table, against which the release-window
recomputation is verified (max abs difference reported; expected ~1e-9 km
since both derive from the same parsed mean motions).

Output: ``experiments/validation/window_sensitivity.csv`` with one row per
event window (annotation_id, sat_id, delta_sma_release_m,
delta_sma_symmetric_m, difference_m = symmetric - release, sign_agreement).
Windows whose bracket cannot be computed under either definition keep empty
cells and are counted in the summary.

Run via the experiment dispatcher:

    python scripts/experiments/run_experiments.py window-sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyzers.event_response import mean_motion_to_sma_m
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.experiment_params import (
    ANALYSIS_WINDOW_POST_HOURS,
    ANALYSIS_WINDOW_PRE_HOURS,
)

DEFAULT_EVENT_WINDOWS = REPO_ROOT / "dataset" / "mission_reported" / "annotations" / "event_windows.csv"
DEFAULT_TLE_EVIDENCE_DIR = REPO_ROOT / "dataset" / "mission_reported" / "evidence" / "tle"
DEFAULT_RESPONSE_TABLE = REPO_ROOT / "experiments" / "validation" / "maneuver_event_response_validation.csv"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "validation" / "window_sensitivity.csv"

OUTPUT_COLUMNS = [
    "annotation_id",
    "sat_id",
    "delta_sma_release_m",
    "delta_sma_symmetric_m",
    "difference_m",
    "sign_agreement",
]


def load_tle_index(tle_dir: Path, sat_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(sorted epoch-ns array, parallel mean-motion array) for one target.

    Same preparation as the release TLE path: NaN mean motions dropped,
    deduplicated by epoch, sorted (``load_tle_history`` dedup discipline).
    """
    path = Path(tle_dir) / f"{sat_id}.parquet"
    if not path.is_file():
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=float)
    frame = pd.read_parquet(path, columns=["epoch", "mean_motion_rad_per_min"])
    frame = frame.dropna(subset=["epoch", "mean_motion_rad_per_min"])
    frame = frame.drop_duplicates(subset=["epoch"]).sort_values("epoch")
    epochs_ns = frame["epoch"].astype("int64").to_numpy(dtype=np.int64)
    return epochs_ns, frame["mean_motion_rad_per_min"].to_numpy(dtype=float)


def bracket_delta_sma_m(index: tuple[np.ndarray, np.ndarray], start_ns: int, end_ns: int) -> float:
    """TLE SMA shift (m) between the nearest epochs bracketing [start, end].

    Bracketing rule identical to ``analyzers.event_response.compute_tle_response``:
    the last catalog epoch at-or-before ``start`` and the first at-or-after
    ``end``.  Returns NaN when either side of the bracket does not exist.
    """
    epochs_ns, mean_motion = index
    if len(epochs_ns) == 0:
        return float("nan")
    before_pos = int(np.searchsorted(epochs_ns, start_ns, side="right")) - 1
    after_pos = int(np.searchsorted(epochs_ns, end_ns, side="left"))
    if before_pos < 0 or after_pos >= len(epochs_ns):
        return float("nan")
    sma_before_m = mean_motion_to_sma_m(float(mean_motion[before_pos]))
    sma_after_m = mean_motion_to_sma_m(float(mean_motion[after_pos]))
    return sma_after_m - sma_before_m


def window_sensitivity(event_windows: pd.DataFrame, tle_dir: Path) -> pd.DataFrame:
    """Per-window release vs symmetric TLE SMA-shift table."""
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    pre = pd.Timedelta(hours=ANALYSIS_WINDOW_PRE_HOURS)
    post = pd.Timedelta(hours=ANALYSIS_WINDOW_POST_HOURS)
    for _, window in event_windows.iterrows():
        sat_id = str(window["sat_id"])
        if sat_id not in cache:
            cache[sat_id] = load_tle_index(tle_dir, sat_id)
        index = cache[sat_id]
        start = pd.to_datetime(window["window_start_utc"], utc=True, format="mixed")
        end = pd.to_datetime(window["window_end_utc"], utc=True, format="mixed")
        event_epoch = start + pre
        release = bracket_delta_sma_m(index, int(start.value), int(end.value))
        symmetric = bracket_delta_sma_m(
            index, int((event_epoch - post).value), int((event_epoch + post).value)
        )
        if np.isnan(release) or np.isnan(symmetric):
            agreement = ""
            difference = np.nan
        else:
            agreement = bool(np.sign(release) == np.sign(symmetric))
            difference = symmetric - release
        rows.append(
            {
                "annotation_id": window["annotation_id"],
                "sat_id": sat_id,
                "delta_sma_release_m": release,
                "delta_sma_symmetric_m": symmetric,
                "difference_m": difference,
                "sign_agreement": agreement,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def verify_against_release(frame: pd.DataFrame, response: pd.DataFrame) -> dict[str, float]:
    """Max/median abs diff (km) between the recomputed and published delta_sma_km."""
    published = response[["annotation_id", "delta_sma_km"]].copy()
    published["delta_sma_km"] = pd.to_numeric(published["delta_sma_km"], errors="coerce")
    merged = frame.merge(published, on="annotation_id", how="left")
    both = merged.dropna(subset=["delta_sma_release_m", "delta_sma_km"])
    diffs_km = (both["delta_sma_release_m"] / 1000.0 - both["delta_sma_km"]).abs()
    n_published = int(published["delta_sma_km"].notna().sum())
    return {
        "n_published_delta": n_published,
        "n_recomputed": int(frame["delta_sma_release_m"].notna().sum()),
        "n_compared": int(len(both)),
        "max_abs_difference_km": float(diffs_km.max()) if len(diffs_km) else float("nan"),
        "median_abs_difference_km": float(diffs_km.median()) if len(diffs_km) else float("nan"),
    }


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    """Headline statistics over the windows computable under both definitions."""
    both = frame.dropna(subset=["delta_sma_release_m", "delta_sma_symmetric_m"]).copy()
    abs_diff = both["difference_m"].abs()
    agreement = both["sign_agreement"].astype(bool)
    median_release = float(both["delta_sma_release_m"].abs().median())
    median_symmetric = float(both["delta_sma_symmetric_m"].abs().median())
    return {
        "n_windows": int(len(frame)),
        "n_computable_release": int(frame["delta_sma_release_m"].notna().sum()),
        "n_computable_symmetric": int(frame["delta_sma_symmetric_m"].notna().sum()),
        "n_computable_both": int(len(both)),
        "median_abs_difference_m": round(float(abs_diff.median()), 4) if len(both) else None,
        "p95_abs_difference_m": round(float(abs_diff.quantile(0.95)), 4) if len(both) else None,
        "max_abs_difference_m": round(float(abs_diff.max()), 4) if len(both) else None,
        "sign_agreement_fraction": round(float(agreement.mean()), 4) if len(both) else None,
        "median_abs_delta_sma_release_m": round(median_release, 4) if len(both) else None,
        "median_abs_delta_sma_symmetric_m": round(median_symmetric, 4) if len(both) else None,
        "median_abs_delta_sma_change_m": round(median_symmetric - median_release, 4) if len(both) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-windows", default=str(DEFAULT_EVENT_WINDOWS))
    parser.add_argument("--tle-dir", default=str(DEFAULT_TLE_EVIDENCE_DIR))
    parser.add_argument("--response-table", default=str(DEFAULT_RESPONSE_TABLE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    event_windows = pd.read_csv(resolve_repo_path(args.event_windows))
    response = pd.read_csv(resolve_repo_path(args.response_table))
    frame = window_sensitivity(event_windows, resolve_repo_path(args.tle_dir))

    out = resolve_repo_path(args.output)
    ensure_directory(out.parent)
    frame.to_csv(out, index=False)

    verification = verify_against_release(frame, response)
    summary = summarize(frame)
    print(f"wrote {out} ({len(frame)} windows)", flush=True)
    print(json.dumps({"verification": verification, "summary": summary}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
