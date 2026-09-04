"""Generate per-event t0 state estimates and harmonization experiments E1-E3."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyzers.state_estimation import (
    STATUS_COMPUTED,
    compute_target_estimates,
    propagation_residual_km,
)
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.experiment_params import PROPAGATION_BIN_EDGES_HOURS

DEFAULT_ANNOTATED_ROOT = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "reference_validation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "tables"

DEFAULT_TARGETS = [
    "sentinel-3a", "sentinel-3b", "jason-3", "sentinel-6a", "cryosat-2", "saral",
    "jason-1", "jason-2", "topex-poseidon", "hy-2a", "swot",
]

SOURCE_STATUS_COLUMNS = ["tle_state_status", "orbit_state_status", "slr_state_status"]
SOURCE_NAMES = {"tle_state_status": "tle", "orbit_state_status": "orbit", "slr_state_status": "slr"}

# Single shared bin definition (C8): the experiment-local variant
# [0, 6, 12, 24, 48, 96, 168, inf] diverged from the tv-hardening bins and
# gave the two published propagation tables different binning.
E2_BINS_HOURS = list(PROPAGATION_BIN_EDGES_HOURS)


def load_windows(annotated_root: Path, sat_id: str) -> pd.DataFrame:
    """Load one target's annotated event windows with t0 timestamps."""
    path = annotated_root / sat_id / "maneuver_annotations.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def coverage_rows(estimates: pd.DataFrame) -> list[dict]:
    """E1: per-target per-source t0-estimate availability."""
    rows = []
    for sat_id, group in estimates.groupby("sat_id"):
        row = {"sat_id": sat_id, "window_count": len(group)}
        for column, source in SOURCE_NAMES.items():
            row[f"{source}_computed_count"] = int((group[column] == STATUS_COMPUTED).sum())
        rows.append(row)
    return rows


def propagation_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    """E2: TLE-vs-POD position residual binned by propagation distance."""
    rows = []
    for _, row in estimates.iterrows():
        if row.get("tle_state_status") != STATUS_COMPUTED or row.get("orbit_state_status") != STATUS_COMPUTED:
            continue
        residual = propagation_residual_km(row.to_dict(), row.to_dict())
        if residual is None:
            continue
        rows.append(
            {
                "sat_id": row["sat_id"],
                "annotation_id": row["annotation_id"],
                "propagation_hours": row["tle_propagation_seconds"] / 3600.0,
                "residual_km": residual,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["propagation_bin_hours"] = pd.cut(frame["propagation_hours"], bins=E2_BINS_HOURS)
    summary = (
        frame.groupby("propagation_bin_hours", observed=True)["residual_km"]
        .agg(["count", "median", lambda s: s.quantile(0.75)])
        .rename(columns={"<lambda_0>": "p75"})
        .reset_index()
    )
    summary["propagation_bin_hours"] = summary["propagation_bin_hours"].astype(str)
    return summary


def interpolation_selfcheck_rows(estimates: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    """E3: decimate-then-interpolate orbit error budget per target."""
    from analyzers.event_response import build_target_context, load_orbit_file
    from analyzers.state_estimation import interpolate_orbit_state

    rows = []
    for sat_id in sorted(estimates["sat_id"].unique()):
        context = build_target_context(raw_root, sat_id)
        orbit_index = context.get("orbit_index") or []
        errors = []
        for entry in orbit_index[:10]:
            try:
                frame, _ = load_orbit_file(entry["path"])
            except Exception:
                continue
            frame = frame.copy()
            frame["epoch"] = pd.to_datetime(frame["epoch"], utc=True)
            frame = frame.sort_values("epoch").reset_index(drop=True)
            if len(frame) < 12:
                continue
            decimated = frame.iloc[::2].reset_index(drop=True)
            held_out = frame.iloc[1::2].reset_index(drop=True)
            for point in held_out["epoch"].iloc[2:-2].sample(min(5, len(held_out) - 4), random_state=42):
                state = interpolate_orbit_state(decimated, point)
                if state is None:
                    continue
                actual = frame[frame["epoch"] == point]
                if actual.empty:
                    continue
                error = np.sqrt(
                    (state["orbit_x_m"] - actual["x"].iloc[0]) ** 2
                    + (state["orbit_y_m"] - actual["y"].iloc[0]) ** 2
                    + (state["orbit_z_m"] - actual["z"].iloc[0]) ** 2
                )
                errors.append(error)
        if errors:
            rows.append(
                {
                    "sat_id": sat_id,
                    "sample_count": len(errors),
                    "interpolation_error_median_m": float(np.median(errors)),
                    "interpolation_error_p75_m": float(np.quantile(errors, 0.75)),
                    "interpolation_error_max_m": float(np.max(errors)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated-root", default=str(DEFAULT_ANNOTATED_ROOT))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include", action="append", default=None, help="sat_id filter (repeatable)")
    args = parser.parse_args()

    annotated_root = resolve_repo_path(args.annotated_root)
    raw_root = resolve_repo_path(args.raw_root)
    output_dir = ensure_directory(resolve_repo_path(args.output_dir))

    sat_ids = args.include or DEFAULT_TARGETS
    frames = []
    for sat_id in sat_ids:
        windows = load_windows(annotated_root, sat_id)
        if windows.empty:
            print(f"{sat_id}: no annotated windows, skipped", flush=True)
            continue
        estimates = compute_target_estimates(sat_id, windows, raw_root)
        frames.append(estimates)
        print(f"{sat_id}: {len(estimates)} windows estimated", flush=True)
    estimates = pd.concat(frames, ignore_index=True)
    estimates.to_csv(output_dir / "maneuver_event_state_estimates.csv", index=False)

    coverage = pd.DataFrame(coverage_rows(estimates))
    coverage.to_csv(output_dir / "harmonization_coverage.csv", index=False)

    propagation = propagation_rows(estimates)
    propagation.to_csv(output_dir / "sgp4_propagation_residuals.csv", index=False)

    selfcheck = interpolation_selfcheck_rows(estimates, raw_root)
    selfcheck.to_csv(output_dir / "orbit_interpolation_selfcheck.csv", index=False)

    print(f"wrote {len(estimates)} state estimates -> {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
