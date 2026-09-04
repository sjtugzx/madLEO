"""Synthetic two-estimator attenuation demo on a shipped precise-orbit arc.

Reviewer question: the OLS/Deming slope of ~0.6 between the TLE-scale and the
period-averaged-orbit SMA-shift responses is attributed to (a) period
averaging of a near-impulsive change and (b) the catalog bracket spacing
(median 46.8 h).  This experiment demonstrates synthetically, on real shipped
evidence, how much attenuation those two mechanisms can produce.

Design (fully deterministic; no random draws, so the seed-42 requirement is
satisfied vacuously -- SEED is kept for provenance):

1. ARC: the longest event-window-free stretch of the shipped sentinel-3a
   precise-orbit evidence parquet, required to be sampling-continuous (no
   epoch gap > MAX_SAMPLING_GAP_SECONDS) and free of DEGRADED quality flags;
   the central ARC_DAYS days are used.
2. TRUTH: the osculating semi-major-axis series a(t) from the shipped state
   vectors (vis-viva; the shipped velocities are Earth-fixed, so the shared
   ``processors.physical_qc.ecef_velocity_to_inertial`` omega x r correction
   is applied first), detrended with a single linear fit so the injected
   signal is the only secular feature.
3. INJECTION: a finite-duration linear ramp of size delta_a over
   RAMP_DURATION_HOURS starting at t0 (the arc center).  The ramp -- not a
   pure step -- is the physically motivated ingredient: it represents the
   finite time over which a real orbit change completes and over which the
   catalog absorbs it (multi-impulse annotated operations span up to ~7 h;
   the catalog-visible mean-element transition is broader).  A pure-step
   control is run alongside and reported in the summary: under a strictly
   impulsive injection BOTH estimators recover the injection exactly, so all
   attenuation reported here comes from the ramp, not from the estimators
   themselves.
4. ESTIMATORS (per grid cell):
   a. ORBIT (the release's Algorithm 1, reused verbatim via
      ``analyzers.event_response._period_averaged_sma_m``): centered rolling
      one-nodal-period mean of the injected osculating series, then band
      medians over the 12 h bands bracketing the release window
      ([t0-18 h, t0-6 h) and (t0+24 h, t0+36 h]), difference.
   b. TLE-SCALE: the catalog is modelled as a backward-looking mean of the
      period-averaged series over one TLE fit span
      (CATALOG_FIT_SPAN_HOURS = 24 h) -- a TLE fitted over an arc straddling
      the maneuver absorbs only part of it.  Synthetic catalog epochs are
      placed with the bracket slack split symmetrically around the release
      window: t_before = t0-6 h-u, t_after = t0+24 h+u with
      u = max(0, (spacing - 30 h)/2).  Spacings below the 30 h window length
      (12 h, 24 h in the grid) are NOT realizable under the release
      bracketing rule and degenerate to the window-edge placement (u = 0).

Output: ``experiments/validation/attenuation_demo.csv``, one row per
(delta_a x bracket spacing) grid cell: injected_da_m, bracket_spacing_h,
orbit_estimator_m, tle_estimator_m, attenuation_orbit, attenuation_tle.

Honest caveats (also printed in the run summary): because the system is
linear, attenuation is essentially independent of injected_da_m except where
residual post-averaging noise competes with the smallest injections; and the
catalog fit-span model attenuates the TLE-scale estimator MORE than the orbit
estimator, i.e. the dilution produced by mechanisms (a)+(b) acts in the
OPPOSITE direction to the observed orbit-on-TLE slope of 0.59.  The observed
slope is instead quantitatively consistent with errors-in-variables
attenuation from the published TLE noise floor (24 m, vs the orbit
estimator's 1.1 m) against the event-scale signal spread; the summary prints
the implied signal std for comparison.

Run via the experiment dispatcher:

    python scripts/experiments/run_experiments.py attenuation-demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyzers.event_response import EARTH_MU_M3_PER_S2, _period_averaged_sma_m
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.experiment_params import (
    ANALYSIS_WINDOW_POST_HOURS,
    ANALYSIS_WINDOW_PRE_HOURS,
    BRACKET_BAND_HOURS,
    SMA_SHIFT_NOISE_FLOOR_M,
)
from processors.physical_qc import ecef_velocity_to_inertial

DEFAULT_EVENT_WINDOWS = REPO_ROOT / "dataset" / "mission_reported" / "annotations" / "event_windows.csv"
DEFAULT_ORBIT_EVIDENCE_DIR = REPO_ROOT / "dataset" / "mission_reported" / "evidence" / "orbit"
DEFAULT_RESPONSE_TABLE = REPO_ROOT / "experiments" / "validation" / "maneuver_event_response_validation.csv"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "validation" / "attenuation_demo.csv"

#: Target and length of the quiet demonstration arc.
ARC_SAT_ID = "sentinel-3a"
ARC_DAYS = 5.0
#: Sampling continuity / quality requirements for the selected arc.
MAX_SAMPLING_GAP_SECONDS = 120.0
#: Duration (h) of the injected finite-duration ramp (the catalog-absorption
#: ingredient; see module docstring, item 3).
RAMP_DURATION_HOURS = 40.0
#: Backward-looking span (h) over which a catalog epoch averages the
#: mean-element series (TLE fit span model; item 4b).
CATALOG_FIT_SPAN_HOURS = 24.0
#: Demonstration grid (P4-style single definition; experiment-local by design).
INJECTED_DA_GRID_M = (10.0, 20.0, 50.0, 100.0, 500.0)
BRACKET_SPACING_GRID_H = (12.0, 24.0, 46.8, 72.0)
#: Determinism provenance (no random draws in the design).
SEED = 42
#: Observed OLS slope of orbit_mean_sma_shift_m on the TLE-scale response in
#: the shipped maneuver_event_response_validation.csv (recomputed at runtime
#: from the table; this constant only anchors the docstring/summary text).
OBSERVED_OLS_SLOPE = 0.59

STATE_COLUMNS = ["epoch", "quality", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]

OUTPUT_COLUMNS = [
    "injected_da_m",
    "bracket_spacing_h",
    "orbit_estimator_m",
    "tle_estimator_m",
    "attenuation_orbit",
    "attenuation_tle",
]


def find_quiet_arc(states: pd.DataFrame, event_windows: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of the central ARC_DAYS of the longest event-free stretch.

    Candidate stretches are the gaps between consecutive event windows of
    ARC_SAT_ID (plus the mission head/tail); they are tried longest-first and
    must be sampling-continuous and free of DEGRADED quality flags.
    """
    epochs_ns = states["epoch"].astype("int64").to_numpy(dtype=np.int64)
    arc_ns = int(ARC_DAYS * 24 * 3600 * 1e9)
    windows = event_windows[event_windows["sat_id"] == ARC_SAT_ID]
    bounds: list[tuple[int, int]] = []
    for _, row in windows.iterrows():
        start = pd.to_datetime(row["window_start_utc"], utc=True, format="mixed")
        end = pd.to_datetime(row["window_end_utc"], utc=True, format="mixed")
        bounds.append((int(start.value), int(end.value)))
    bounds.sort()
    edges = [int(epochs_ns[0])] + [b for pair in bounds for b in pair] + [int(epochs_ns[-1])]
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    gaps = [(lo, hi) for lo, hi in gaps if hi - lo >= arc_ns]
    for lo, hi in sorted(gaps, key=lambda g: g[1] - g[0], reverse=True):
        center = (lo + hi) // 2
        arc_lo, arc_hi = center - arc_ns // 2, center + arc_ns // 2
        i_lo = int(np.searchsorted(epochs_ns, arc_lo, side="left"))
        i_hi = int(np.searchsorted(epochs_ns, arc_hi, side="right"))
        segment = states.iloc[i_lo:i_hi]
        max_gap_s = float(np.diff(epochs_ns[i_lo:i_hi]).max()) / 1e9 if i_hi - i_lo > 1 else float("inf")
        degraded = segment["quality"].astype(str).str.startswith("DEGRADED").any()
        if max_gap_s <= MAX_SAMPLING_GAP_SECONDS and not degraded:
            start = states["epoch"].iloc[i_lo]
            end = states["epoch"].iloc[i_hi - 1]
            return start, end
    raise RuntimeError(f"no qualifying {ARC_DAYS}-day quiet arc found for {ARC_SAT_ID}")


def osculating_sma_m(states: pd.DataFrame) -> pd.Series:
    """Osculating semi-major axis (m) via vis-viva with the ECEF->inertial fix."""
    position = states[["x_m", "y_m", "z_m"]].astype(float).to_numpy()
    velocity = states[["vx_mps", "vy_mps", "vz_mps"]].astype(float).to_numpy()
    velocity = ecef_velocity_to_inertial(velocity, position)
    radius = np.sqrt((position * position).sum(axis=1))
    speed2 = (velocity * velocity).sum(axis=1)
    energy_term = 2.0 / radius - speed2 / EARTH_MU_M3_PER_S2
    with np.errstate(divide="ignore", invalid="ignore"):
        sma = np.where(energy_term > 0, 1.0 / energy_term, np.nan)
    return pd.Series(sma, index=states.index)


def detrended_sma_m(sma: pd.Series, times_s: np.ndarray) -> pd.Series:
    """SMA series with its secular (linear) trend removed, level preserved."""
    slope, intercept = np.polyfit(times_s, sma.to_numpy(dtype=float), 1)
    return sma - (slope * times_s + intercept - float(sma.mean()))


def ramp_profile(hours_since_t0: np.ndarray, duration_h: float) -> np.ndarray:
    """Unit-magnitude linear ramp from t0 to t0+duration_h (1 afterwards)."""
    if duration_h <= 0:
        return (hours_since_t0 >= 0).astype(float)
    return np.clip(hours_since_t0 / duration_h, 0.0, 1.0)


def orbit_estimator_m(period_averaged: pd.Series, times_s: np.ndarray, t0_s: float) -> float:
    """Release Algorithm 1 band-median difference (m) around the window."""
    start_s = t0_s - ANALYSIS_WINDOW_PRE_HOURS * 3600.0
    end_s = t0_s + ANALYSIS_WINDOW_POST_HOURS * 3600.0
    band_s = BRACKET_BAND_HOURS * 3600.0
    before = period_averaged[(times_s >= start_s - band_s) & (times_s < start_s)].dropna()
    after = period_averaged[(times_s > end_s) & (times_s <= end_s + band_s)].dropna()
    if before.empty or after.empty:
        return float("nan")
    return float(after.median() - before.median())


def catalog_value_m(period_averaged: pd.Series, times_s: np.ndarray, query_s: float) -> float:
    """Catalog epoch value: backward mean of the mean-element series over one fit span."""
    span_s = CATALOG_FIT_SPAN_HOURS * 3600.0
    values = period_averaged[(times_s >= query_s - span_s) & (times_s <= query_s)].dropna()
    return float(values.mean()) if len(values) else float("nan")


def tle_estimator_m(period_averaged: pd.Series, times_s: np.ndarray, t0_s: float, spacing_h: float) -> float:
    """TLE-scale bracket difference (m) for a total catalog bracket spacing (h)."""
    window_h = ANALYSIS_WINDOW_PRE_HOURS + ANALYSIS_WINDOW_POST_HOURS
    slack_h = max(0.0, (spacing_h - window_h) / 2.0)
    t_before_s = t0_s - (ANALYSIS_WINDOW_PRE_HOURS + slack_h) * 3600.0
    t_after_s = t0_s + (ANALYSIS_WINDOW_POST_HOURS + slack_h) * 3600.0
    before = catalog_value_m(period_averaged, times_s, t_before_s)
    after = catalog_value_m(period_averaged, times_s, t_after_s)
    return after - before


def run_grid(
    base: pd.Series,
    times_s: np.ndarray,
    t0_s: float,
    orbit_frame: pd.DataFrame,
    duration_h: float,
) -> pd.DataFrame:
    """Both estimators over the (delta_a x bracket spacing) grid for one injection profile.

    Both estimators are linear in the series, so the quiet-arc baseline
    contribution cancels exactly by subtraction: each reported estimate is
    est(injected) - est(base), i.e. the response attributable to the
    injection alone (this also absorbs any residual post-detrend curvature
    of the real arc).
    """
    hours_since_t0 = (times_s - t0_s) / 3600.0
    profile = ramp_profile(hours_since_t0, duration_h)
    base_averaged = _period_averaged_sma_m(orbit_frame, base)
    base_orbit = orbit_estimator_m(base_averaged, times_s, t0_s)
    base_tle = {spacing_h: tle_estimator_m(base_averaged, times_s, t0_s, spacing_h) for spacing_h in BRACKET_SPACING_GRID_H}
    rows = []
    for delta_a in INJECTED_DA_GRID_M:
        injected = base + delta_a * profile
        period_averaged = _period_averaged_sma_m(orbit_frame, injected)
        for spacing_h in BRACKET_SPACING_GRID_H:
            orbit_est = orbit_estimator_m(period_averaged, times_s, t0_s) - base_orbit
            tle_est = tle_estimator_m(period_averaged, times_s, t0_s, spacing_h) - base_tle[spacing_h]
            rows.append(
                {
                    "injected_da_m": delta_a,
                    "bracket_spacing_h": spacing_h,
                    "orbit_estimator_m": orbit_est,
                    "tle_estimator_m": tle_est,
                    "attenuation_orbit": orbit_est / delta_a,
                    "attenuation_tle": tle_est / delta_a,
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def observed_ols_slope(response: pd.DataFrame) -> float:
    """OLS slope of orbit_mean_sma_shift_m on the TLE-scale response (shipped table)."""
    both = response.dropna(subset=["delta_sma_km", "orbit_mean_sma_shift_m"])
    x = pd.to_numeric(both["delta_sma_km"], errors="coerce").to_numpy() * 1000.0
    y = pd.to_numeric(both["orbit_mean_sma_shift_m"], errors="coerce").to_numpy()
    return float(np.polyfit(x, y, 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-windows", default=str(DEFAULT_EVENT_WINDOWS))
    parser.add_argument("--orbit-dir", default=str(DEFAULT_ORBIT_EVIDENCE_DIR))
    parser.add_argument("--response-table", default=str(DEFAULT_RESPONSE_TABLE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    orbit_path = resolve_repo_path(args.orbit_dir) / f"{ARC_SAT_ID}.parquet"
    states = pd.read_parquet(orbit_path, columns=STATE_COLUMNS)
    states = states.sort_values("epoch").reset_index(drop=True)

    event_windows = pd.read_csv(resolve_repo_path(args.event_windows))
    arc_start, arc_end = find_quiet_arc(states, event_windows)
    arc = states[(states["epoch"] >= arc_start) & (states["epoch"] <= arc_end)].reset_index(drop=True)
    print(f"quiet arc: {arc_start} .. {arc_end} ({len(arc)} states)", flush=True)

    times_s = arc["epoch"].astype("int64").to_numpy(dtype=np.int64) / 1e9
    t0_s = float(times_s[0] + times_s[-1]) / 2.0
    sma = osculating_sma_m(arc)
    base = detrended_sma_m(sma, times_s)
    orbit_frame = arc[["epoch"]].copy()

    ramp = run_grid(base, times_s, t0_s, orbit_frame, RAMP_DURATION_HOURS)
    out = resolve_repo_path(args.output)
    ensure_directory(out.parent)
    ramp.to_csv(out, index=False)

    # Pure-step control (duration 0): both estimators must recover the
    # injection exactly -- all attenuation above is ramp geometry, not the
    # estimators.
    step = run_grid(base, times_s, t0_s, orbit_frame, 0.0)
    step_control = {
        "max_abs_attenuation_orbit_deviation_from_1": float((step["attenuation_orbit"] - 1.0).abs().max()),
        "max_abs_attenuation_tle_deviation_from_1": float((step["attenuation_tle"] - 1.0).abs().max()),
    }

    response = pd.read_csv(resolve_repo_path(args.response_table))
    slope = observed_ols_slope(response)
    noise2 = SMA_SHIFT_NOISE_FLOOR_M["tle_median"] ** 2
    implied_signal_std_m = float(np.sqrt(noise2 * slope / max(1e-12, 1.0 - slope)))

    median_spacing_rows = ramp[np.isclose(ramp["bracket_spacing_h"], 46.8)]
    summary = {
        "arc": {"sat_id": ARC_SAT_ID, "start": str(arc_start), "end": str(arc_end), "n_states": int(len(arc))},
        "ramp_duration_h": RAMP_DURATION_HOURS,
        "catalog_fit_span_h": CATALOG_FIT_SPAN_HOURS,
        "median_attenuation_orbit": round(float(ramp["attenuation_orbit"].median()), 4),
        "median_attenuation_tle": round(float(ramp["attenuation_tle"].median()), 4),
        "attenuation_orbit_at_median_spacing": round(float(median_spacing_rows["attenuation_orbit"].median()), 4),
        "attenuation_tle_at_median_spacing": round(float(median_spacing_rows["attenuation_tle"].median()), 4),
        "implied_orbit_over_tle_ratio_at_median_spacing": round(
            float(median_spacing_rows["attenuation_orbit"].median() / median_spacing_rows["attenuation_tle"].median()), 4
        ),
        "pure_step_control": step_control,
        "observed_ols_slope_orbit_on_tle": round(slope, 4),
        "eiv_implied_signal_std_m_at_tle_noise_floor_24m": round(implied_signal_std_m, 2),
    }
    print(f"wrote {out} ({len(ramp)} grid rows)", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
