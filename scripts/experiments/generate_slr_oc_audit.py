"""SLR geometric O-C (observed-minus-computed) range-residual audit.

Computes the standard POD-validation quantity for SLR: the difference
between the observed one-way range (ILRS normal point) and the range
computed geometrically from the precise-orbit satellite state and the
SLRF2020 station coordinate, for every maneuver event window that has
both orbit and SLR evidence coverage. This replaces the earlier
normal-point precision (sigma_m) shift analysis, which measures station
data quality rather than orbit accuracy.

Method
------
For each event window of a target with both evidence snapshots:

1. Select SLR normal points with receive epochs in
   [window_start - 12 h, window_end + 12 h] and split them into a
   "before" side (< window_start) and an "after" side (> window_end).
2. For each observation, interpolate the satellite ITRF position at the
   reflect epoch with scipy CubicSpline over the orbit-state epochs.
   Orbit snapshots are multi-day arcs; the spline is built per contiguous
   arc (arcs split where the epoch gap exceeds max(300 s, 10x median
   cadence)) so no interpolation across data gaps occurs. Observations
   outside an arc interior are dropped.
   Light-time correction: t_reflect = t_receive - range/c, iterated once
   (tau_0 = observed range / c, then tau_1 = computed distance / c).
3. The station position is the SLRF2020 SINEX coordinate (solution valid
   at the observation epoch when the site has multiple solutions),
   linearly propagated with the published velocity from the SINEX
   reference epoch; Earth rotation over the ~10-20 ms flight time is
   neglected (sub-mm in ITRF).
4. O-C = observed_range_m - |r_sat(t_reflect) - r_station|.

Uncorrected terms (first-order audit; by design)
------------------------------------------------
The absolute O-C is dominated by terms that are deliberately NOT
corrected:

- tropospheric refraction delay: ~2.3 m zenith, up to ~20 m at low
  elevation (not corrected);
- satellite center-of-mass to retroreflector-offset: ~1-2 m for LEO
  altimetry satellites (not corrected);
- station range/instrument biases and residual station coordinate error;
- relativistic path delay (cm-level);
- the normal-point epoch is treated as the receive epoch (if the provider
  convention differs, the induced error is second order, sub-meter).

These put the absolute O-C at the meter-to-tens-of-meters level, which is
expected and acceptable: the audit looks for CHANGES of the O-C statistics
across maneuver windows (before vs after), which the quasi-constant biases
above largely cancel. Robust statistics are used: medians, and per-side
outlier clipping at |O-C - median| > 1000 m (gross blunders / time-tag
errors), which are excluded from the reported counts and moments.

Reads the shipped release snapshots (dataset/mission_reported/evidence/, read-only) and
writes results/tables/slr_oc_residuals.csv (per window) and
results/tables/slr_oc_summary.csv (per target). Targets without an orbit
snapshot (sentinel-6a) are reported in the summary with zero computable
windows.

Example
-------
python scripts/experiments/run_experiments.py slr-oc-audit
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Bootstrap: make the flat data_pipeline packages importable (mirrors the
# path setup run_experiments.py performs before importing this module).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data_pipeline"))

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path  # noqa: E402

EVIDENCE = REPO_ROOT / "dataset" / "mission_reported" / "evidence"
ORBIT_DIR = EVIDENCE / "orbit"
SLR_DIR = EVIDENCE / "slr"
EVENT_WINDOWS = REPO_ROOT / "dataset" / "mission_reported" / "annotations" / "event_windows.csv"
SINEX_PATH = REPO_ROOT / "docs" / "standards" / "ilrs-slrf2020-pos-vel-2025.05.13.snx"
TABLES = REPO_ROOT / "results" / "tables"

C_LIGHT_M_S = 299792458.0
DEFAULT_PAD_HOURS = 12.0
DEFAULT_MIN_OBS_PER_SIDE = 3
BLUNDER_CLIP_M = 1000.0
RANGE_PLAUSIBLE_M = (2.0e5, 1.0e7)  # one-way LEO slant ranges
SEGMENT_GAP_MIN_S = 300.0
SEGMENT_GAP_FACTOR = 10.0
MIN_SEGMENT_STATES = 4  # CubicSpline (not-a-knot) needs >= 4 nodes
SECONDS_PER_YEAR = 365.25 * 86400.0

RESIDUAL_COLUMNS = (
    "annotation_id",
    "sat_id",
    "n_obs_before",
    "n_obs_after",
    "oc_median_before_m",
    "oc_median_after_m",
    "oc_shift_m",
    "oc_rms_before_m",
    "oc_rms_after_m",
    "stations_used",
)


# ---------------------------------------------------------------------------
# SLRF2020 SINEX parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationSolution:
    """One (site code, point, solution) entry of the SLRF2020 SINEX file."""

    code: str
    start_s: float  # validity start, unix seconds
    end_s: float  # validity end, unix seconds (inf when open-ended)
    tref_s: float  # reference epoch of the coordinate/velocity pair
    xyz_m: tuple[float, float, float]
    vel_mps: tuple[float, float, float]


def _snx_year(yy: int) -> int:
    """Expand a SINEX two-digit year (pivot 50: >=50 -> 19xx, else 20xx)."""
    return 1900 + yy if yy >= 50 else 2000 + yy


def _snx_epoch_to_unix_s(token: str) -> float:
    """Convert 'yy:ddd:sod' SINEX epoch to unix seconds."""
    yy, doy, sod = (int(part) for part in token.split(":"))
    epoch = datetime(_snx_year(yy), 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy - 1, seconds=sod
    )
    return epoch.timestamp()


def parse_slrf_station_solutions(sinex_path: Path) -> dict[str, list[StationSolution]]:
    """Parse STAX/STAY/STAZ and VELX/VELY/VELZ from an SLRF SINEX file.

    Returns station code -> list of solutions. Solution validity windows
    come from SOLUTION/EPOCHS (00:000:00000 as an end epoch is open-ended).
    """
    epochs: dict[tuple[str, str, str], tuple[float, float]] = {}
    estimates: dict[tuple[str, str, str], dict[str, float]] = {}
    ref_epochs: dict[tuple[str, str, str], float] = {}
    block: str | None = None
    for line in Path(sinex_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("+SOLUTION/EPOCHS"):
            block = "epochs"
            continue
        if line.startswith("+SOLUTION/ESTIMATE"):
            block = "estimate"
            continue
        if line.startswith("-SOLUTION"):
            block = None
            continue
        if block is None or not line.strip() or line.startswith("*"):
            continue
        tokens = line.split()
        if block == "epochs":
            # code pt soln T start end mean_epoch
            key = (tokens[0], tokens[1], tokens[2])
            start_s = _snx_epoch_to_unix_s(tokens[4])
            end_s = (
                np.inf if tokens[5] == "00:000:00000" else _snx_epoch_to_unix_s(tokens[5])
            )
            epochs[key] = (start_s, end_s)
        else:
            # index type code pt soln ref_epoch unit constraint value std
            key = (tokens[2], tokens[3], tokens[4])
            estimates.setdefault(key, {})[tokens[1]] = float(tokens[8])
            ref_epochs[key] = _snx_epoch_to_unix_s(tokens[5])

    solutions: dict[str, list[StationSolution]] = {}
    for key, values in estimates.items():
        code = key[0]
        if not {"STAX", "STAY", "STAZ"} <= values.keys():
            continue
        vel = tuple(values.get(name, 0.0) for name in ("VELX", "VELY", "VELZ"))
        sol = StationSolution(
            code=code,
            start_s=epochs.get(key, (-np.inf, np.inf))[0],
            end_s=epochs.get(key, (-np.inf, np.inf))[1],
            tref_s=ref_epochs[key],
            xyz_m=(values["STAX"], values["STAY"], values["STAZ"]),
            vel_mps=vel,  # type: ignore[arg-type]
        )
        solutions.setdefault(code, []).append(sol)
    for sols in solutions.values():
        sols.sort(key=lambda s: s.start_s)
    return solutions


def station_positions(
    solutions: list[StationSolution], t_s: np.ndarray
) -> np.ndarray:
    """Station ITRF positions at observation epochs (velocity-propagated).

    For each epoch the solution whose validity window covers the epoch is
    used; epochs before the first solution use the earliest solution and
    epochs after the last validity end use the latest one.
    """
    pos = np.full((t_s.size, 3), np.nan)
    unassigned = np.arange(t_s.size)
    for sol in solutions:
        if unassigned.size == 0:
            break
        times = t_s[unassigned]
        inside = (times >= sol.start_s) & (times <= sol.end_s)
        if inside.any():
            idx = unassigned[inside]
            years = (t_s[idx] - sol.tref_s) / SECONDS_PER_YEAR
            pos[idx] = np.asarray(sol.xyz_m)[None, :] + np.asarray(sol.vel_mps)[
                None, :
            ] * years[:, None]
            unassigned = unassigned[~inside]
    if unassigned.size and solutions:
        times = t_s[unassigned]
        early = unassigned[times < solutions[0].start_s]
        late = np.setdiff1d(unassigned, early)
        for idx, sol in ((early, solutions[0]), (late, solutions[-1])):
            if idx.size:
                years = (t_s[idx] - sol.tref_s) / SECONDS_PER_YEAR
                pos[idx] = np.asarray(sol.xyz_m)[None, :] + np.asarray(sol.vel_mps)[
                    None, :
                ] * years[:, None]
    return pos


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _utc_seconds(series: pd.Series) -> np.ndarray:
    """UTC datetime series -> float unix seconds."""
    if series.dt.tz is not None:
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series.to_numpy().astype("datetime64[ns]").astype(np.int64).astype(np.float64) / 1e9


# ---------------------------------------------------------------------------
# O-C computation
# ---------------------------------------------------------------------------


def _orbit_segments(t_s: np.ndarray, xyz: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split the orbit states into contiguous arcs (no interpolation over gaps).

    Arcs break where the epoch spacing exceeds max(300 s, 10x median spacing).
    """
    diffs = np.diff(t_s)
    median_dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else SEGMENT_GAP_MIN_S
    gap_threshold = max(SEGMENT_GAP_MIN_S, SEGMENT_GAP_FACTOR * median_dt)
    break_idx = np.flatnonzero(diffs > gap_threshold)
    bounds = np.concatenate(([0], break_idx + 1, [t_s.size]))
    segments = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo >= MIN_SEGMENT_STATES:
            segments.append((t_s[lo:hi], xyz[lo:hi]))
    return segments


def compute_target_oc(
    orbit_path: Path, slr_path: Path, station_solutions: dict[str, list[StationSolution]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per-observation O-C residuals for one target.

    Returns a frame with columns t_s (unix seconds), oc_m, station_id, plus
    accounting counters (unknown stations, blunders, observations outside
    orbit arcs).
    """
    orbit = pd.read_parquet(
        orbit_path, columns=["epoch", "x_m", "y_m", "z_m"]
    ).sort_values("epoch")
    orbit = orbit.dropna(subset=["x_m", "y_m", "z_m"]).drop_duplicates(subset="epoch")
    orb_t = _utc_seconds(orbit["epoch"])
    orb_xyz = orbit[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)

    slr = pd.read_parquet(slr_path, columns=["epoch", "range_m", "station_id"])
    slr = slr.sort_values("epoch").reset_index(drop=True)
    counters = {"n_slr_total": int(len(slr)), "n_station_unknown": 0, "n_blunder": 0}
    slr = slr[slr["range_m"].between(*RANGE_PLAUSIBLE_M)].reset_index(drop=True)
    slr["station_id"] = slr["station_id"].astype(str)
    known = slr["station_id"].isin(station_solutions)
    counters["n_station_unknown"] = int((~known).sum())
    slr = slr[known].reset_index(drop=True)

    t_s = _utc_seconds(slr["epoch"])
    rng = slr["range_m"].to_numpy(dtype=float)
    oc = np.full(t_s.size, np.nan)
    station_ids = slr["station_id"].to_numpy()

    for seg_t, seg_xyz in _orbit_segments(orb_t, orb_xyz):
        margin = float(np.median(np.diff(seg_t)))  # keep clear of arc edges
        lo = int(np.searchsorted(t_s, seg_t[0] + margin, side="left"))
        hi = int(np.searchsorted(t_s, seg_t[-1] - margin, side="right"))
        if hi <= lo:
            continue
        spline = CubicSpline(seg_t, seg_xyz, axis=0, extrapolate=False)
        idx = np.arange(lo, hi)
        times = t_s[idx]
        obs_rng = rng[idx]
        st_pos = np.full((times.size, 3), np.nan)
        for station in np.unique(station_ids[idx]):
            sel = station_ids[idx] == station
            st_pos[sel] = station_positions(station_solutions[station], times[sel])
        valid = np.isfinite(st_pos).all(axis=1) & np.isfinite(obs_rng)
        idx_v = idx[valid]
        if idx_v.size == 0:
            continue
        t_valid = t_s[idx_v]
        st_valid = st_pos[valid]
        r_valid = obs_rng[valid]
        # Light-time iteration (one step beyond the observed-range estimate).
        t_reflect = t_valid - r_valid / C_LIGHT_M_S
        sat_pos = spline(t_reflect)
        dist = np.linalg.norm(sat_pos - st_valid, axis=1)
        t_reflect = t_valid - dist / C_LIGHT_M_S
        sat_pos = spline(t_reflect)  # NaN outside the arc -> dropped below
        dist = np.linalg.norm(sat_pos - st_valid, axis=1)
        computed = np.where(np.isfinite(dist), r_valid - dist, np.nan)
        oc[idx_v] = computed

    frame = pd.DataFrame({"t_s": t_s, "oc_m": oc, "station_id": station_ids})
    return frame, counters


def _side_stats(oc_values: np.ndarray) -> tuple[int, float, float]:
    """(count, median, rms) of one window side after blunder clipping."""
    finite = oc_values[np.isfinite(oc_values)]
    if finite.size == 0:
        return 0, np.nan, np.nan
    center = float(np.median(finite))
    kept = finite[np.abs(finite - center) <= BLUNDER_CLIP_M]
    if kept.size == 0:
        return 0, np.nan, np.nan
    return (
        int(kept.size),
        float(np.median(kept)),
        float(np.sqrt(np.mean(kept**2))),
    )


def audit_windows(
    sat_id: str,
    windows: pd.DataFrame,
    oc_frame: pd.DataFrame,
    pad_hours: float,
    min_obs_per_side: int,
) -> list[dict]:
    """Per-window before/after O-C statistics for one target."""
    pad_s = pad_hours * 3600.0
    t_s = oc_frame["t_s"].to_numpy()
    oc = oc_frame["oc_m"].to_numpy()
    stations = oc_frame["station_id"].to_numpy()
    rows = []
    for window in windows.itertuples(index=False):
        lo = int(np.searchsorted(t_s, window.start_s - pad_s, side="left"))
        hi = int(np.searchsorted(t_s, window.end_s + pad_s, side="right"))
        if hi <= lo:
            continue
        times = t_s[lo:hi]
        before = oc[lo:hi][times < window.start_s]
        after = oc[lo:hi][times > window.end_s]
        n_b, med_b, rms_b = _side_stats(before)
        n_a, med_a, rms_a = _side_stats(after)
        if n_b + n_a == 0:
            continue
        side_stations = np.unique(
            np.concatenate(
                (
                    stations[lo:hi][(times < window.start_s) & np.isfinite(oc[lo:hi])],
                    stations[lo:hi][(times > window.end_s) & np.isfinite(oc[lo:hi])],
                )
            )
        )
        shift = med_a - med_b if (n_b >= min_obs_per_side and n_a >= min_obs_per_side) else np.nan
        rows.append(
            {
                "annotation_id": window.annotation_id,
                "sat_id": sat_id,
                "n_obs_before": n_b,
                "n_obs_after": n_a,
                "oc_median_before_m": round(med_b, 3) if np.isfinite(med_b) else np.nan,
                "oc_median_after_m": round(med_a, 3) if np.isfinite(med_a) else np.nan,
                "oc_shift_m": round(shift, 3) if np.isfinite(shift) else np.nan,
                "oc_rms_before_m": round(rms_b, 3) if np.isfinite(rms_b) else np.nan,
                "oc_rms_after_m": round(rms_a, 3) if np.isfinite(rms_a) else np.nan,
                "stations_used": ";".join(side_stations),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-windows", default=str(EVENT_WINDOWS))
    parser.add_argument("--orbit-dir", default=str(ORBIT_DIR))
    parser.add_argument("--slr-dir", default=str(SLR_DIR))
    parser.add_argument("--sinex", default=str(SINEX_PATH))
    parser.add_argument("--pad-hours", type=float, default=DEFAULT_PAD_HOURS)
    parser.add_argument("--min-obs-per-side", type=int, default=DEFAULT_MIN_OBS_PER_SIDE)
    parser.add_argument("--output-dir", default=str(TABLES))
    args = parser.parse_args()

    windows_path = Path(resolve_repo_path(args.event_windows))
    orbit_dir = Path(resolve_repo_path(args.orbit_dir))
    slr_dir = Path(resolve_repo_path(args.slr_dir))
    sinex_path = Path(resolve_repo_path(args.sinex))
    output = ensure_directory(Path(resolve_repo_path(args.output_dir)))

    station_solutions = parse_slrf_station_solutions(sinex_path)
    print(f"parsed {len(station_solutions)} SLRF2020 station sites from {sinex_path.name}", flush=True)

    windows = pd.read_csv(windows_path)
    windows["start_s"] = _utc_seconds(pd.to_datetime(windows["window_start_utc"], utc=True, format="mixed"))
    windows["end_s"] = _utc_seconds(pd.to_datetime(windows["window_end_utc"], utc=True, format="mixed"))

    residual_rows: list[dict] = []
    summary_rows: list[dict] = []
    all_oc: list[np.ndarray] = []
    for sat_id in sorted(windows["sat_id"].unique()):
        sat_windows = windows[windows["sat_id"] == sat_id]
        orbit_path = orbit_dir / f"{sat_id}.parquet"
        slr_path = slr_dir / f"{sat_id}.parquet"
        if not (orbit_path.exists() and slr_path.exists()):
            reason = "no orbit snapshot" if not orbit_path.exists() else "no slr snapshot"
            print(f"{sat_id}: skipped ({reason}); {len(sat_windows)} windows not computable", flush=True)
            summary_rows.append(
                {"sat_id": sat_id, "n_windows_computable": 0, "median_oc_shift_m": np.nan, "median_oc_rms_m": np.nan}
            )
            continue
        oc_frame, counters = compute_target_oc(orbit_path, slr_path, station_solutions)
        rows = audit_windows(sat_id, sat_windows, oc_frame, args.pad_hours, args.min_obs_per_side)
        residual_rows.extend(rows)
        computable = [r for r in rows if np.isfinite(r["oc_shift_m"])]
        rms_values = [
            r[col]
            for r in computable
            for col in ("oc_rms_before_m", "oc_rms_after_m")
            if np.isfinite(r[col])
        ]
        summary_rows.append(
            {
                "sat_id": sat_id,
                "n_windows_computable": len(computable),
                "median_oc_shift_m": round(float(np.median([r["oc_shift_m"] for r in computable])), 3)
                if computable
                else np.nan,
                "median_oc_rms_m": round(float(np.median(rms_values)), 3) if rms_values else np.nan,
            }
        )
        oc_used = oc_frame["oc_m"].to_numpy()
        oc_used = oc_used[np.isfinite(oc_used)]
        all_oc.append(oc_used)
        print(
            f"{sat_id}: {len(rows)} windows with observations, "
            f"{len(computable)} computable (>={args.min_obs_per_side} obs per side); "
            f"{oc_used.size} O-C residuals, median {np.median(np.abs(oc_used)):.2f} m absolute"
            if oc_used.size
            else f"{sat_id}: no computable O-C residuals",
            flush=True,
        )

    residuals = (
        pd.DataFrame(residual_rows, columns=list(RESIDUAL_COLUMNS))
        .sort_values(["sat_id", "annotation_id"])
        .reset_index(drop=True)
    )
    residuals.to_csv(output / "slr_oc_residuals.csv", index=False)
    print(f"wrote slr_oc_residuals.csv ({len(residuals)} rows)", flush=True)

    summary = pd.DataFrame(summary_rows, columns=["sat_id", "n_windows_computable", "median_oc_shift_m", "median_oc_rms_m"]).sort_values("sat_id")
    summary.to_csv(output / "slr_oc_summary.csv", index=False)
    print(f"wrote slr_oc_summary.csv ({len(summary)} rows)", flush=True)

    pooled = np.concatenate(all_oc) if all_oc else np.array([])
    if pooled.size:
        med_abs = float(np.median(np.abs(pooled)))
        frac_100m = float(np.mean(np.abs(pooled) > 100.0))
        frac_1km = float(np.mean(np.abs(pooled) > 1000.0))
        print(
            f"pooled O-C sanity: median |O-C| = {med_abs:.2f} m, "
            f"{100 * frac_100m:.2f}% beyond 100 m, {100 * frac_1km:.4f}% beyond 1 km",
            flush=True,
        )
        if med_abs > 100.0:
            print(
                "WARNING: pooled median |O-C| exceeds 100 m -- check interpolation "
                "and frame handling before using these tables",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
