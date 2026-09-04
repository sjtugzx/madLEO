"""Extract catalog-indicated orbit-change candidates from the Starlink TLE table.

For every satellite in the released operational slice, consecutive TLE
epochs are differenced on the semi-major axis (Kepler conversion of the
catalog mean motion, mu = 3.986004418e14 m^3/s^2).  Steps whose absolute
change exceeds ``STEP_THRESHOLD_M`` (100 m) are written out as candidate
orbit-change events with the bracketing catalog epochs.

These are CANDIDATE signatures derived from catalog fitting noise and
possible maneuvers alike: at the altitudes where they concentrate
(sub-480 km deployment and transfer orbits), two days of drag move the
semi-major axis by amounts comparable to the threshold, so the table is an
audit artifact for method development, never a maneuver label (see the
claim boundary in dataset/docs/metadata.md).

Example:
    python scripts/experiments/run_experiments.py starlink-step-candidates
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TLE = REPO / "dataset" / "operational" / "starlink" / "tle_elements.parquet"
OUT = REPO / "experiments" / "starlink"
OUT_TABLE = OUT / "starlink_tle_step_candidates.csv"

MU_M3_PER_S2 = 3.986004418e14
STEP_THRESHOLD_M = 100.0


def main() -> int:
    tle = pd.read_parquet(TLE).sort_values(["sat_id", "epoch"], kind="mergesort")
    n_rad_per_s = tle["mean_motion_rad_per_min"].to_numpy() / 60.0
    a_km = (MU_M3_PER_S2 / n_rad_per_s ** 2) ** (1.0 / 3.0) / 1000.0
    tle = tle.assign(a_km=a_km)
    tle["delta_sma_m"] = tle.groupby("sat_id")["a_km"].diff() * 1000.0
    tle["median_altitude_km"] = tle.groupby("sat_id")["a_km"].transform("median") - 6378.137
    steps = tle[tle["delta_sma_m"].abs() > STEP_THRESHOLD_M].copy()
    prev_epoch = tle.groupby("sat_id")["epoch"].shift()
    steps["bracket_start_utc"] = prev_epoch[steps.index].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    steps["bracket_end_utc"] = steps["epoch"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = steps[["sat_id", "satellite_name", "bracket_start_utc", "bracket_end_utc", "delta_sma_m", "median_altitude_km"]]
    out = out.round({"delta_sma_m": 1, "median_altitude_km": 2})
    out.to_csv(OUT_TABLE, index=False)
    print(f"wrote {OUT_TABLE} ({len(out)} candidate steps over {out['sat_id'].nunique()} satellites)", flush=True)
    mag = out["delta_sma_m"].abs()
    print(
        "median |delta_a| %.1f m, p90 %.1f m, max %.1f m | median altitude of contributing satellites: %.0f km"
        % (mag.median(), mag.quantile(0.9), mag.max(), out["median_altitude_km"].median()),
        flush=True,
    )
    below = (out["median_altitude_km"] < 480).mean()
    print(f"share from sub-480 km deployment/transfer population: {below:.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
