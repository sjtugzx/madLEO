"""Event-epoch orbit state estimation (harmonization layer A1).

For every mission-reported maneuver window this module propagates each
evidence source to the IDS first-impulse epoch ``t0``:

- TLE: SGP4 propagation of the bracketing TLE epochs to ``t0`` (TEME), with
  an astropy TEME->ITRS transform for comparison against precise orbits.
  Propagation distance (seconds) is recorded as uncertainty metadata.
- Precise orbit: cubic-spline interpolation of the parsed SP3/EOF/OGDR
  series to ``t0`` (positions in meters, velocities in m/s).
- SLR: nearest-pass residual statistics around ``t0``.

The per-source estimates feed the harmonization experiments E1 (coverage),
E2 (SGP4 propagation residual curve), and E3 (interpolation self-check)
reported in the paper's Technical Validation section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

from analyzers.event_response import (
    build_target_context,
    load_orbit_file,
    load_slr_records,
    select_orbit_files,
)
from benchmarking.normalization import parse_tle_line_pairs

STATUS_COMPUTED = "computed"
STATUS_MISSING = "source_missing"
STATUS_INSUFFICIENT = "insufficient_coverage"
STATUS_PROPAGATION_FAR = "propagation_too_far"
STATUS_PARSE = "parse_unavailable"

MAX_PROPAGATION_SECONDS = 7 * 86400.0


# ---------------------------------------------------------------------------
# TLE line history + SGP4 propagation
# ---------------------------------------------------------------------------


def load_tle_line_history(tle_dir: str | Path | None) -> pd.DataFrame:
    """Parse raw TLE files keeping line1/line2 for SGP4 propagation.

    Delegates to the canonical parser
    ``benchmarking.normalization.parse_tle_line_pairs`` (checksum + NORAD
    cross-check with counted rejections; REVIEW_FINDINGS 6.3.1) -- no
    private line-pairing loop here.
    """
    if tle_dir is None:
        return pd.DataFrame()
    directory = Path(tle_dir)
    if not directory.is_dir():
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".tle", ".txt"} or path.name.startswith("."):
            continue
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        for row in parse_tle_line_pairs(lines).itertuples(index=False):
            records.append(
                {
                    "epoch": row.epoch,
                    "line1": row.line1,
                    "line2": row.line2,
                }
            )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["epoch"] = pd.to_datetime(frame["epoch"], utc=True)
    return frame.drop_duplicates(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)


def propagate_tle(line1: str, line2: str, epoch: pd.Timestamp) -> dict[str, Any] | None:
    """Propagate one TLE pair to ``epoch``; returns TEME and ITRS state (m, m/s)."""
    sat = Satrec.twoline2rv(line1, line2)
    ts = pd.to_datetime(epoch, utc=True).to_pydatetime()
    jd, fr = jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second + ts.microsecond / 1e6)
    error, position_km, velocity_km_s = sat.sgp4(jd, fr)
    if error != 0:
        return None
    state = {
        "teme_x_m": position_km[0] * 1000.0,
        "teme_y_m": position_km[1] * 1000.0,
        "teme_z_m": position_km[2] * 1000.0,
        "teme_vx_mps": velocity_km_s[0] * 1000.0,
        "teme_vy_mps": velocity_km_s[1] * 1000.0,
        "teme_vz_mps": velocity_km_s[2] * 1000.0,
    }
    itrs = teme_to_itrs(position_km, velocity_km_s, ts)
    if itrs is not None:
        state.update(itrs)
    return state


#: Module-level failure ledger for :func:`teme_to_itrs` (exception type ->
#: count).  The transform returns ``None`` instead of raising -- a missing
#: frame-data table or astropy import failure must not kill the pipeline --
#: but such failures are COUNTED here rather than silently swallowed
#: (REVIEW_FINDINGS silent-except ruling).  Behavior is unchanged.
TEME_ITRS_FAILURES: dict[str, int] = {}


def teme_to_itrs(position_km: Any, velocity_km_s: Any, ts: Any) -> dict[str, float] | None:
    """Convert TEME state to ITRS via astropy; None when frame data unavailable.

    A ``None`` return is always mirrored by a counted entry in
    :data:`TEME_ITRS_FAILURES` (keyed by exception type) -- the failure is
    visible, never silent.
    """
    try:
        import astropy.units as u
        from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
        from astropy.time import Time

        obstime = Time(ts)
        rep = CartesianRepresentation(np.asarray(position_km) * u.km)
        dif = CartesianDifferential(np.asarray(velocity_km_s) * u.km / u.s)
        itrs = TEME(rep.with_differentials(dif), obstime=obstime).transform_to(ITRS(obstime=obstime))
        pos = itrs.cartesian.xyz.to(u.m).value
        vel = itrs.velocity.d_xyz.to(u.m / u.s).value
        return {
            "itrs_x_m": float(pos[0]),
            "itrs_y_m": float(pos[1]),
            "itrs_z_m": float(pos[2]),
            "itrs_vx_mps": float(vel[0]),
            "itrs_vy_mps": float(vel[1]),
            "itrs_vz_mps": float(vel[2]),
        }
    except Exception as exc:
        TEME_ITRS_FAILURES[type(exc).__name__] = TEME_ITRS_FAILURES.get(type(exc).__name__, 0) + 1
        return None


def estimate_tle_state(tle_lines: pd.DataFrame, t0: pd.Timestamp) -> dict[str, Any]:
    """SGP4-propagate the nearest TLE epoch before t0 forward to t0."""
    result: dict[str, Any] = {"tle_state_status": STATUS_MISSING}
    if tle_lines is None or tle_lines.empty:
        return result
    before = tle_lines[tle_lines["epoch"] <= t0]
    if before.empty:
        result["tle_state_status"] = STATUS_INSUFFICIENT
        return result
    nearest = before.iloc[-1]
    propagation_seconds = float((t0 - nearest["epoch"]).total_seconds())
    result["tle_propagation_seconds"] = propagation_seconds
    result["tle_epoch_used_utc"] = nearest["epoch"].isoformat().replace("+00:00", "Z")
    if propagation_seconds > MAX_PROPAGATION_SECONDS:
        result["tle_state_status"] = STATUS_PROPAGATION_FAR
        return result
    state = propagate_tle(nearest["line1"], nearest["line2"], t0)
    if state is None:
        result["tle_state_status"] = STATUS_PARSE
        return result
    result.update(state)
    result["tle_state_status"] = STATUS_COMPUTED
    return result


# ---------------------------------------------------------------------------
# Precise-orbit spline interpolation
# ---------------------------------------------------------------------------


def interpolate_orbit_state(orbit_df: pd.DataFrame, t0: pd.Timestamp, edge_tolerance_seconds: float = 900.0) -> dict[str, Any] | None:
    """Interpolate a parsed orbit series to t0 (meters, m/s).

    Dense series (<=5 s sampling, e.g. EOF/OGDR at 1 Hz) use a cubic spline.
    Sparse series (SP3 at 30-60 s+) use a 9-point Lagrange window, following
    the IGS practice that high-order polynomials are required for cm-level
    recovery at coarse sampling. Targets within ``edge_tolerance_seconds``
    of the series edges are extrapolated with the edge polynomial and flagged
    via ``orbit_extrapolation_seconds`` — SP3 arcs routinely end a few
    minutes before/after the event epoch.
    """
    from scipy.interpolate import CubicSpline

    df = orbit_df.copy()
    df["epoch"] = pd.to_datetime(df["epoch"], utc=True)
    df = df.sort_values("epoch").reset_index(drop=True)
    if len(df) < 4:
        return None
    seconds = (df["epoch"] - df["epoch"].iloc[0]).dt.total_seconds().to_numpy()
    target = (t0 - df["epoch"].iloc[0]).total_seconds()
    extrapolation = 0.0
    if target < seconds[0]:
        extrapolation = seconds[0] - target
    elif target > seconds[-1]:
        extrapolation = target - seconds[-1]
    if extrapolation > edge_tolerance_seconds:
        return None
    eval_target = target
    median_step = float(np.median(np.diff(seconds)))
    sparse = median_step > 5.0 and len(df) >= 10
    state: dict[str, Any] = {}
    for column, prefix in (("x", "orbit_x_m"), ("y", "orbit_y_m"), ("z", "orbit_z_m")):
        values = df[column].to_numpy(dtype=float)
        if sparse:
            # Clamp the fit window to the series edge, but evaluate at the
            # true target (true extrapolation within the tolerance).
            center = int(np.clip(np.searchsorted(seconds, eval_target), 5, len(seconds) - 5))
            lo = max(0, min(center - 5, len(seconds) - 10))
            window = slice(lo, lo + 10)
            coeffs = np.polyfit(seconds[window] - eval_target, values[window], 9)
            state[prefix] = float(np.polyval(coeffs, 0.0))
            deriv = np.polyder(coeffs)
            state[_velocity_key(prefix)] = float(np.polyval(deriv, 0.0))
        else:
            spline = CubicSpline(seconds, values)
            state[prefix] = float(spline(eval_target))
            state[_velocity_key(prefix)] = float(spline.derivative()(eval_target))
    if extrapolation:
        state["orbit_extrapolation_seconds"] = extrapolation
    return state


def _velocity_key(prefix: str) -> str:
    return prefix.replace("orbit_x_m", "orbit_vx_mps").replace("orbit_y_m", "orbit_vy_mps").replace("orbit_z_m", "orbit_vz_mps")


def estimate_orbit_state(context: dict[str, Any], t0: pd.Timestamp, margin_hours: float = 6.0) -> dict[str, Any]:
    """Interpolate the best-covering precise-orbit file to t0."""
    result: dict[str, Any] = {"orbit_state_status": STATUS_MISSING}
    orbit_index = context.get("orbit_index") or []
    if not orbit_index:
        return result
    start = t0 - pd.Timedelta(hours=margin_hours)
    end = t0 + pd.Timedelta(hours=margin_hours)
    candidates = select_orbit_files(orbit_index, start, end, margin_hours)
    for candidate in candidates:
        try:
            frame, _ = load_orbit_file(candidate["path"])
        except Exception:
            continue
        try:
            state = interpolate_orbit_state(frame, t0)
        except Exception:
            # C16: isolate interpolation failures (e.g. duplicate epochs
            # breaking the CubicSpline construction) to THIS candidate file;
            # they must invalidate one arc, never abort the whole
            # state-estimates run.
            continue
        if state is not None:
            result.update(state)
            result["orbit_file_used"] = str(candidate["path"])
            result["orbit_state_status"] = STATUS_COMPUTED
            return result
    result["orbit_state_status"] = STATUS_INSUFFICIENT if candidates else STATUS_MISSING
    return result


# ---------------------------------------------------------------------------
# SLR nearest-pass summary
# ---------------------------------------------------------------------------


def estimate_slr_state(context: dict[str, Any], t0: pd.Timestamp, margin_hours: float = 24.0) -> dict[str, Any]:
    """Summarize the SLR passes nearest to t0."""
    result: dict[str, Any] = {"slr_state_status": STATUS_MISSING}
    slr_df = context.get("slr_df")
    if slr_df is None or slr_df.empty:
        return result
    df = slr_df.copy()
    # Ruling-17: rows marked qc_rejected by the record triage (Class-A
    # physics violations) are counted rejections -- they never contribute
    # to the state-estimate metrics.
    if "qc_rejected" in df.columns:
        df = df[~df["qc_rejected"].fillna(False).astype(bool)]
    df["epoch"] = pd.to_datetime(df["epoch"], utc=True)
    before = df[df["epoch"] <= t0]
    after = df[df["epoch"] > t0]
    if before.empty or after.empty:
        result["slr_state_status"] = STATUS_INSUFFICIENT
        return result
    nearest_before = before.iloc[-1]
    nearest_after = after.iloc[0]
    result["slr_nearest_before_seconds"] = float((t0 - nearest_before["epoch"]).total_seconds())
    result["slr_nearest_after_seconds"] = float((nearest_after["epoch"] - t0).total_seconds())
    metric = "residual_m" if "residual_m" in df.columns and df["residual_m"].notna().any() else "sigma_m"
    if metric not in df.columns:
        result["slr_state_status"] = STATUS_PARSE
        return result
    band = df[(df["epoch"] >= t0 - pd.Timedelta(hours=margin_hours)) & (df["epoch"] <= t0 + pd.Timedelta(hours=margin_hours))]
    result["slr_band_point_count"] = int(len(band))
    if len(band) == 0:
        result["slr_state_status"] = STATUS_INSUFFICIENT
        return result
    result["slr_band_median_m"] = float(band[metric].median())
    result["slr_band_rms_m"] = float(np.sqrt(np.nanmean(band[metric].astype(float) ** 2)))
    result["slr_metric_used"] = metric
    result["slr_state_status"] = STATUS_COMPUTED
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_state_context(raw_root: str | Path, sat_id: str) -> dict[str, Any]:
    """Build the per-target context once (orbit index, SLR records, TLE lines)."""
    context = build_target_context(raw_root, sat_id)
    slr_paths = [entry["path"] for entry in (context.get("slr_index") or [])]
    context["slr_df"] = load_slr_records(slr_paths)
    context["tle_lines"] = load_tle_line_history(Path(raw_root) / sat_id / "tle")
    return context


def compute_window_estimates(window: pd.Series, context: dict[str, Any]) -> dict[str, Any]:
    """Compute per-source t0 state estimates for one event window."""
    t0 = pd.to_datetime(window.get("event_time_utc") or window.get("window_start_utc"), utc=True)
    row: dict[str, Any] = {
        "annotation_id": window.get("annotation_id", ""),
        "sat_id": window.get("sat_id", ""),
        "t0_utc": t0.isoformat().replace("+00:00", "Z"),
    }
    row.update(estimate_tle_state(context.get("tle_lines"), t0))
    row.update(estimate_orbit_state(context, t0))
    row.update(estimate_slr_state(context, t0))
    return row


def compute_target_estimates(sat_id: str, windows: pd.DataFrame, raw_root: str | Path) -> pd.DataFrame:
    """Compute t0 state estimates for all windows of one target."""
    context = build_state_context(raw_root, sat_id)
    rows = [compute_window_estimates(window, context) for _, window in windows.iterrows()]
    return pd.DataFrame(rows)


def propagation_residual_km(tle_row: dict[str, Any], orbit_row: dict[str, Any]) -> float | None:
    """Position residual between TLE-propagated (ITRS) and orbit-splined states."""
    try:
        diff = np.array(
            [
                tle_row["itrs_x_m"] - orbit_row["orbit_x_m"],
                tle_row["itrs_y_m"] - orbit_row["orbit_y_m"],
                tle_row["itrs_z_m"] - orbit_row["orbit_z_m"],
            ]
        )
    except KeyError:
        return None
    return float(np.linalg.norm(diff) / 1000.0)
