"""Quantitative per-event-window source response computation.

For each mission-reported maneuver window this module computes numeric
response magnitudes from the three independent local sources:

- TLE: deltas of mean motion (rad/min), eccentricity, inclination (rad),
  and BSTAR between the nearest catalog epochs before/after the window,
  plus the derived semi-major-axis shift (km).  Bracketing-epoch selection
  and angular unwrapping reuse ``align_maneuver_evidence`` logic.
- Precise orbit: pre/post window median geocentric-radius shift (m) from
  position samples, and a vis-viva osculating semi-major-axis shift (m)
  when velocity components are available (EOF, OGDR, SWOT POR).  SP3
  products are position-only, so only the radius shift is reported there.
- SLR: normal-point residual/sigma RMS before/during/after the window and
  the absolute bias shift (m), mirroring ``extract_slr_support``'s
  residual_m-vs-sigma_m column preference.

Per-source status codes: computed / insufficient_coverage /
source_missing / parse_unavailable.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alignment.align_maneuver_evidence import _angle_delta
from alignment.audit_reference_window_alignment import infer_orbit_product_spans, infer_slr_file_spans
from benchmarking.experiment_params import BRACKET_BAND_HOURS
from benchmarking.normalization import parse_raw_tle_file
from processors import physical_qc as _physical_qc
from processors.jason3_ogdr_processor import process_jason3_ogdr_file
from processors.physical_qc import (
    PhysicalQCViolation,
    assert_orbit_records,
    assert_slr_records,
    ecef_velocity_to_inertial,
)
from processors.pod_processor import eof_to_dataframe, parse_sp3_epochs, parse_sp3_header
from processors.slr_formats import mission_window_for_sat
from processors.slr_processor import SPEED_OF_LIGHT, slr_to_dataframe
from processors.swot_por_processor import process_swot_por_file
from processors.timebase import sp3_time_scale_shift_seconds, sp3_time_system


EARTH_MU_M3_PER_S2 = 3.986004418e14
# P4 single source: the 12-h sampling margin IS the TLE bracket band width
# (benchmarking.experiment_params.BRACKET_BAND_HOURS); aliased here only for
# the keyword-arg sites that predate the shared module.
DEFAULT_MARGIN_HOURS = BRACKET_BAND_HOURS
MIN_BAND_SAMPLES = 2

STATUS_COMPUTED = "computed"
STATUS_INSUFFICIENT = "insufficient_coverage"
STATUS_MISSING = "source_missing"
STATUS_PARSE_UNAVAILABLE = "parse_unavailable"

PER_WINDOW_COLUMNS = [
    "annotation_id",
    "sat_id",
    "batch",
    "confidence_tier",
    "window_start_utc",
    "window_end_utc",
    "tle_status",
    "tle_nearest_before_utc",
    "tle_nearest_after_utc",
    "delta_mean_motion_rad_per_min",
    "delta_eccentricity",
    "delta_inclination_rad",
    "delta_bstar",
    "tle_sma_before_km",
    "tle_sma_after_km",
    "delta_sma_km",
    "orbit_status",
    "orbit_source",
    "orbit_file_count",
    "orbit_parse_failure_count",
    "orbit_parse_failure_detail",
    "orbit_samples_before",
    "orbit_samples_after",
    "orbit_radius_before_m",
    "orbit_radius_after_m",
    "orbit_radius_shift_m",
    "orbit_sma_before_m",
    "orbit_sma_after_m",
    "orbit_sma_shift_m",
    "orbit_mean_sma_before_m",
    "orbit_mean_sma_after_m",
    "orbit_mean_sma_shift_m",
    "slr_status",
    "slr_metric_column",
    "slr_obs_before",
    "slr_obs_during",
    "slr_obs_after",
    "slr_rms_before_m",
    "slr_rms_during_m",
    "slr_rms_after_m",
    "slr_bias_shift_m",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_utc(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _iso(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    ts = pd.to_datetime(value, utc=True)
    if ts.microsecond:
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def mean_motion_to_sma_m(mean_motion_rad_per_min: float) -> float:
    """Convert mean motion (rad/min) to semi-major axis (m) via Kepler's third law."""
    n_rad_per_s = float(mean_motion_rad_per_min) / 60.0
    return (EARTH_MU_M3_PER_S2 / (n_rad_per_s * n_rad_per_s)) ** (1.0 / 3.0)


def confidence_tier(tle_status: str, slr_status: str, orbit_status: str) -> str:
    """Coverage-confidence tier, same rule as the annotated event-window tables."""
    has_tle = tle_status == "covered"
    has_slr = slr_status == "covered"
    has_orbit = orbit_status == "covered"
    if has_tle and has_slr and has_orbit:
        return "A"
    if has_tle and has_orbit:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# TLE response
# ---------------------------------------------------------------------------


def load_tle_history(tle_dir: str | Path | None) -> pd.DataFrame:
    """Parse and merge every raw TLE file in a target directory (deduped by epoch).

    Per-file parse errors are COUNTED, not silently dropped (same ledger
    discipline as the orbit path): each failure is noted on stderr and the
    count lands in the returned frame's ``attrs["tle_parse_failure_count"]``
    (with per-file detail under ``attrs["tle_parse_failures"]``).
    """
    if tle_dir is None:
        return pd.DataFrame()
    directory = Path(tle_dir)
    if not directory.is_dir():
        return pd.DataFrame()
    frames = []
    parse_failures: list[dict[str, str]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".tle", ".txt"} or path.name.startswith("."):
            continue
        try:
            frame = parse_raw_tle_file(path)
        except Exception as exc:
            # Counted, not silent (same ledger discipline as the orbit
            # path): stderr note per file + the failure count lands in the
            # returned frame's ``attrs["tle_parse_failure_count"]``.
            detail = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            parse_failures.append({"path": str(path), "error": detail})
            print(f"tle parse failure ({path.name}): {detail}", file=sys.stderr)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        empty = pd.DataFrame()
        if parse_failures:
            empty.attrs["tle_parse_failure_count"] = len(parse_failures)
            empty.attrs["tle_parse_failures"] = parse_failures
        return empty
    merged = pd.concat(frames, ignore_index=True)
    merged["epoch"] = pd.to_datetime(merged["epoch"], utc=True)
    merged = merged.drop_duplicates(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)
    merged.attrs["tle_parse_failure_count"] = len(parse_failures)
    if parse_failures:
        merged.attrs["tle_parse_failures"] = parse_failures
    return merged


def compute_tle_response(tle_df: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """TLE element deltas between the nearest epochs bracketing the window.

    Same bracketing rule and angular unwrap as ``align_maneuver_evidence``.
    """
    result: dict[str, Any] = {"tle_status": STATUS_MISSING}
    if tle_df is None or tle_df.empty:
        return result

    tle = tle_df.copy()
    tle["epoch"] = pd.to_datetime(tle["epoch"], utc=True)
    tle = tle.sort_values("epoch").reset_index(drop=True)
    before = tle[tle["epoch"] <= start]
    after = tle[tle["epoch"] >= end]
    if before.empty or after.empty:
        result["tle_status"] = STATUS_INSUFFICIENT
        return result

    nearest_before = before.iloc[-1]
    nearest_after = after.iloc[0]
    result["tle_nearest_before_utc"] = _iso(nearest_before["epoch"])
    result["tle_nearest_after_utc"] = _iso(nearest_after["epoch"])

    scalar_fields = {
        "mean_motion": "delta_mean_motion_rad_per_min",
        "eccentricity": "delta_eccentricity",
        "bstar": "delta_bstar",
    }
    for field, output_key in scalar_fields.items():
        if field in tle.columns and pd.notna(nearest_before.get(field)) and pd.notna(nearest_after.get(field)):
            result[output_key] = float(nearest_after[field]) - float(nearest_before[field])
    if "inclination" in tle.columns and pd.notna(nearest_before.get("inclination")) and pd.notna(nearest_after.get("inclination")):
        delta, _ = _angle_delta(float(nearest_after["inclination"]), float(nearest_before["inclination"]))
        result["delta_inclination_rad"] = delta

    if "mean_motion" in tle.columns and pd.notna(nearest_before.get("mean_motion")) and pd.notna(nearest_after.get("mean_motion")):
        sma_before_m = mean_motion_to_sma_m(float(nearest_before["mean_motion"]))
        sma_after_m = mean_motion_to_sma_m(float(nearest_after["mean_motion"]))
        result["tle_sma_before_km"] = sma_before_m / 1000.0
        result["tle_sma_after_km"] = sma_after_m / 1000.0
        result["delta_sma_km"] = (sma_after_m - sma_before_m) / 1000.0

    result["tle_status"] = STATUS_COMPUTED
    return result


# ---------------------------------------------------------------------------
# Precise-orbit response
# ---------------------------------------------------------------------------


def _source_priority(row: dict) -> int:
    """Prefer precise orbit over pod, and POE over MOE within pod files."""
    name = Path(row["path"]).name.upper()
    priority = 0 if row.get("source_type") == "precise_orbit" else 1
    if "POEORB" in name:
        priority -= 0.5
    elif "MOEORB" in name:
        priority += 0.5
    return priority


def _decompress_unix_z(path: Path) -> str:
    """Decompress a unix-compress (.Z) file to text via gzip's LZW support."""
    completed = subprocess.run(
        ["gzip", "-dc", str(path)],
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8", errors="ignore")


# SP3 time-scale conversion (TAI/GPS -> UTC) is metadata driven: the time
# system is read from the product's own %c header line via
# processors.timebase (ERFA-authoritative leap seconds, fix C6/D11).  No
# filename-prefix heuristics and no hand-maintained leap tables here.


def _load_sp3_file(path: Path) -> tuple[pd.DataFrame, bool]:
    """Parse an SP3 file (plain or .Z-compressed) keeping the dominant vehicle.

    Returns (DataFrame, rotating_frame); velocities in ITRF-family products
    are ECEF and need the rotation correction before vis-viva use.
    """
    if path.name.endswith(".Z"):
        text = _decompress_unix_z(path)
        lines = [line.rstrip("\n") for line in text.splitlines()]
    else:
        lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    header = parse_sp3_header(lines)
    records = parse_sp3_epochs(lines, header)
    df = pd.DataFrame(records)
    if df.empty:
        return df, False
    if df["sat_id"].nunique() > 1:
        dominant = df["sat_id"].value_counts().idxmax()
        df = df[df["sat_id"] == dominant].copy()
    df["epoch"] = pd.to_datetime(df["epoch"], utc=True)
    if not df.empty:
        time_system = sp3_time_system(header.header_lines)
        shift_seconds = sp3_time_scale_shift_seconds(time_system, df["epoch"].iloc[0])
        if shift_seconds:
            df["epoch"] = df["epoch"] + pd.Timedelta(seconds=shift_seconds)
    rotating = header.coord_system.upper().startswith(("ITR", "ECEF", "WGS"))
    return df, rotating


def load_orbit_file(path: str | Path) -> tuple[pd.DataFrame, bool]:
    """Load one orbit product file, then run the L2 physics assertions.

    Returns (DataFrame with epoch/x/y/z[/vx/vy/vz], rotating_frame) where
    rotating_frame flags ECEF products whose velocities need the omega x r
    correction before vis-viva use.  Raises on unsupported/broken formats.

    Every non-empty frame passes through ``processors.physical_qc.
    assert_orbit_records`` (radius band, omega x r-corrected vis-viva
    band, strictly increasing epochs; REVIEW_FINDINGS 6.2.4) before it is
    returned; a clean frame carries its QC ledger row in
    ``df.attrs["orbit_qc"]``.  A physics-identity violation is a COUNTED
    FILE-LEVEL REJECTION (Ruling-17): the function returns an EMPTY
    DataFrame carrying ``attrs["orbit_qc_rejection"] = {"assertion",
    "row_index", "value"}`` instead of raising -- the caller counts the
    file as parse-unavailable, so a defective product never enters
    results and never kills the pipeline (known for NRT OGDR gaps and
    legacy data outliers).
    """
    path = Path(path)
    df, rotating = _load_orbit_records(path)
    if not df.empty:
        sat_id = str(df["sat_id"].iloc[0]) if "sat_id" in df.columns else path.stem
        try:
            df.attrs["orbit_qc"] = assert_orbit_records(
                df, sat_id, source=path.name, rotating_frame=rotating
            )
        except PhysicalQCViolation as exc:
            # Ruling-17 counted file-level rejection: refuse the file
            # (empty frame + structured rejection metadata), never
            # silently pass it, never raise to kill the pipeline.
            print(f"orbit QC rejection ({path.name}): {exc}", file=sys.stderr)
            rejected = pd.DataFrame()
            rejected.attrs["orbit_qc_rejection"] = {
                "assertion": str(exc.assertion),
                "row_index": exc.row_index,
                "value": exc.value,
            }
            return rejected, rotating
    return df, rotating


def _load_orbit_records(path: str | Path) -> tuple[pd.DataFrame, bool]:
    """Format dispatch for one orbit product file (no physics assertions)."""
    path = Path(path)
    name = path.name.upper()
    suffix = path.suffix.lower()

    if suffix in {".eof", ".xml"}:
        # Sentinel EOF/POD OSVs are Earth-fixed (verified: vis-viva SMA
        # scatter collapses only after the rotation correction).
        df, _ = eof_to_dataframe(str(path))
        return df, True
    if suffix == ".z" or suffix in {".sp3", ".eph"}:
        return _load_sp3_file(path)
    if suffix == ".gz" or ".RNX" in name:
        raise ValueError("gnss_rinex_observation_not_satellite_ephemeris")
    if suffix == ".nc":
        if name.startswith(("SWOT_POR", "SWOT_VOR")):
            return process_swot_por_file(str(path)), True
        return process_jason3_ogdr_file(str(path)), True
    raise ValueError(f"unsupported_orbit_format:{path.name}")


def select_orbit_files(orbit_index: list[dict], start: pd.Timestamp, end: pd.Timestamp, margin_hours: float) -> list[dict]:
    """Select best-priority orbit files overlapping the window plus margin."""
    margin = pd.Timedelta(hours=margin_hours)
    span_start, span_end = start - margin, end + margin
    matches = []
    for row in orbit_index:
        row_start = _parse_utc(row["start_utc"])
        row_end = _parse_utc(row["end_utc"])
        if row_start < span_end and row_end > span_start:
            matches.append(row)
    if not matches:
        return []
    best = min(_source_priority(row) for row in matches)
    selected = [row for row in matches if _source_priority(row) == best]
    return sorted(selected, key=lambda row: row["start_utc"])


def _vis_viva_sma_m(df: pd.DataFrame, rotating_frame: bool) -> pd.Series:
    """Osculating semi-major axis (m) from state vectors (vis-viva).

    The ECEF -> inertial velocity correction (v_i = v_ecef + omega x r) is
    the shared ``processors.physical_qc.ecef_velocity_to_inertial``
    implementation (single-implementation rule, REVIEW_FINDINGS 6.2.2).
    """
    x = df["x"].astype(float).to_numpy()
    y = df["y"].astype(float).to_numpy()
    z = df["z"].astype(float).to_numpy()
    velocity = df[["vx", "vy", "vz"]].astype(float).to_numpy()
    if rotating_frame:
        velocity = ecef_velocity_to_inertial(velocity, df[["x", "y", "z"]].astype(float).to_numpy())
    radius = np.sqrt(x * x + y * y + z * z)
    speed2 = np.sum(velocity * velocity, axis=1)
    energy_term = 2.0 / radius - speed2 / EARTH_MU_M3_PER_S2
    with np.errstate(divide="ignore", invalid="ignore"):
        sma = np.where(energy_term > 0, 1.0 / energy_term, np.nan)
    return pd.Series(sma, index=df.index)


def compute_orbit_response(
    orbit_df: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    margin_hours: float = DEFAULT_MARGIN_HOURS,
    rotating_frame: bool = False,
) -> dict[str, Any]:
    """Pre/post window orbit response from state-vector samples.

    Metrics: median geocentric-radius shift (position-only, robust across
    formats) and, when velocities are present, median osculating SMA shift.
    """
    result: dict[str, Any] = {"orbit_status": STATUS_MISSING}
    if orbit_df is None or orbit_df.empty:
        return result

    orbit = orbit_df.copy()
    orbit["epoch"] = pd.to_datetime(orbit["epoch"], utc=True)
    orbit = orbit.sort_values("epoch").reset_index(drop=True)
    margin = pd.Timedelta(hours=margin_hours)
    before = orbit[(orbit["epoch"] >= start - margin) & (orbit["epoch"] < start)]
    after = orbit[(orbit["epoch"] > end) & (orbit["epoch"] <= end + margin)]
    result["orbit_samples_before"] = int(len(before))
    result["orbit_samples_after"] = int(len(after))
    if len(before) < MIN_BAND_SAMPLES or len(after) < MIN_BAND_SAMPLES:
        result["orbit_status"] = STATUS_INSUFFICIENT
        return result

    for band, label in ((before, "before"), (after, "after")):
        radius = np.sqrt(band["x"].astype(float) ** 2 + band["y"].astype(float) ** 2 + band["z"].astype(float) ** 2)
        result[f"orbit_radius_{label}_m"] = float(radius.median())
    result["orbit_radius_shift_m"] = result["orbit_radius_after_m"] - result["orbit_radius_before_m"]

    has_velocity = {"vx", "vy", "vz"}.issubset(orbit.columns)
    if has_velocity:
        sma = _vis_viva_sma_m(orbit, rotating_frame)
        sma_before = sma.loc[before.index].dropna()
        sma_after = sma.loc[after.index].dropna()
        if len(sma_before) >= MIN_BAND_SAMPLES and len(sma_after) >= MIN_BAND_SAMPLES:
            result["orbit_sma_before_m"] = float(sma_before.median())
            result["orbit_sma_after_m"] = float(sma_after.median())
            result["orbit_sma_shift_m"] = result["orbit_sma_after_m"] - result["orbit_sma_before_m"]
            mean_sma = _period_averaged_sma_m(orbit, sma)
            if mean_sma is not None:
                mean_before = mean_sma.loc[before.index].dropna()
                mean_after = mean_sma.loc[after.index].dropna()
                if len(mean_before) >= 1 and len(mean_after) >= 1:
                    result["orbit_mean_sma_before_m"] = float(mean_before.median())
                    result["orbit_mean_sma_after_m"] = float(mean_after.median())
                    result["orbit_mean_sma_shift_m"] = result["orbit_mean_sma_after_m"] - result["orbit_mean_sma_before_m"]

    result["orbit_status"] = STATUS_COMPUTED
    return result


def _period_averaged_sma_m(orbit: pd.DataFrame, sma: pd.Series) -> pd.Series | None:
    """Suppress J2 short-periodic oscillation by averaging over one nodal period.

    Returns a rolling one-period mean of the osculating SMA series, aligned to
    the original index (NaN at the edges), or None when the series is too
    sparse. This makes the orbit response comparable to TLE mean-element
    deltas (same dimension, same scale).
    """
    valid = sma.dropna()
    if len(valid) < 8:
        return None
    a = float(valid.median())
    period_s = 2.0 * np.pi * np.sqrt(a**3 / EARTH_MU_M3_PER_S2)
    times = pd.to_datetime(orbit["epoch"], utc=True).astype("int64") / 1e9
    step_s = float(np.median(np.diff(times)))
    if step_s <= 0:
        return None
    window = max(2, int(round(period_s / step_s)))
    return sma.rolling(window=window, center=True, min_periods=max(2, window // 2)).mean()


# ---------------------------------------------------------------------------
# SLR response
# ---------------------------------------------------------------------------


def _apply_slr_record_triage(
    slr_df: pd.DataFrame,
    sat_id: str = "slr",
    mission_span: tuple | None = None,
) -> pd.DataFrame:
    """Record-level SLR QC triage (REVIEW_FINDINGS section 8.4).

    Three classes with three different dispositions:

    - **A -- hard assertions (identity physics), counted row-level
      rejection (Ruling-17).** Delegated to
      ``processors.physical_qc.assert_slr_records`` -- the single shared
      implementation (T5 unification): TOF/range identity
      ``|c * tof / 2 - range| <= SLR_TOF_IDENTITY_TOLERANCE_M``, the
      one-way range band, epoch presence, and the mission-span gate when
      ``mission_span`` is given.  ``range_m`` is derived from the TOF in
      every supported format, so a violation can only be a parsing
      defect -- but a defective ROW is marked, not dropped: offending
      rows carry ``qc_rejected=True`` (rows are NEVER removed), the
      counts land in ``attrs["slr_qc_rejections"]`` (assertion ->
      row count), and metric consumers skip the marked rows.  Nothing
      raises for a violation the shared logic can identify (626k
      measured rows: 0 violations).
    - **B -- source-semantics marker (never a failure).** Normal-point
      records whose source left ``sigma`` / ``num_returns`` unfilled (== 0)
      get a ``source_zero_fields`` tag.  Structural zeros are "source did
      not fill the field", not measurements; full-rate records naturally
      lack these fields.  Rows are kept unconditionally -- dropping them
      would be a category error (Little & Rubin structural zeros).
    - **C -- counted alert.** ``sigma > 1 m`` rows are real but poor
      quality source values; counted in ``slr_qc_alerts`` for adjudication,
      rows kept.

    The physical-QC ledger row lands in ``slr_df.attrs["slr_qc"]``, the
    counted Class-A rejections in ``slr_df.attrs["slr_qc_rejections"]``,
    and the triage alert counts in ``slr_df.attrs["slr_qc_alerts"]``
    (DataFrame attrs survive the immediate return only -- callers that
    need the ledger should read it right after loading).
    """
    alerts: dict[str, int] = {}

    # Class A: shared physical assertions (physical_qc is the one
    # implementation of the TOF identity -- constant, exception type, and
    # NaN semantics all come from there; no local re-derivation).
    qc_rejections: dict[str, int] = {}
    qc_rejected = np.zeros(len(slr_df), dtype=bool)
    try:
        qc = assert_slr_records(slr_df, sat_id=sat_id, mission_span=mission_span)
    except PhysicalQCViolation as exc:
        # Ruling-17 counted row-level rejection: enumerate EVERY offending
        # row through the shared assertion logic (report mode -- no
        # regex-parsing of the exception, no drop-and-retry), mark the
        # rows, and count them; the rows themselves stay in the frame.
        reported = _physical_qc.assert_slr_records(
            slr_df, sat_id=sat_id, mission_span=mission_span, mode="report"
        )
        qc_rejections = dict(reported.get("slr_qc_rejections") or {})
        qc_rejected[reported.get("qc_rejected_index") or []] = True
        if not qc_rejections:
            # The raising gate's verdict cannot be reproduced as row
            # indices by the shared physics logic (e.g. a replacement
            # gate without a report mode): surface it for the caller to
            # count (the staging path records tof_assert_green=False and
            # writes no parquet) -- never silently passed.
            raise
        qc = reported
        print(
            f"slr QC rejection ({sat_id}, {qc_rejections} rows marked): {exc}",
            file=sys.stderr,
        )
    slr_df["qc_rejected"] = qc_rejected
    slr_df.attrs["slr_qc"] = qc
    slr_df.attrs["slr_qc_rejections"] = qc_rejections

    # Class B: source-zero semantics markers (rows are never dropped).
    sigma: pd.Series | None = None
    sigma_zero = np.zeros(len(slr_df), dtype=bool)
    returns_zero = np.zeros(len(slr_df), dtype=bool)
    if "sigma_m" in slr_df.columns:
        sigma = pd.to_numeric(slr_df["sigma_m"], errors="coerce")
        sigma_zero = sigma.eq(0).to_numpy(dtype=bool, na_value=False)
        alerts["source_zero_sigma"] = int(sigma_zero.sum())
    if "num_returns" in slr_df.columns:
        returns = pd.to_numeric(slr_df["num_returns"], errors="coerce")
        returns_zero = returns.eq(0).to_numpy(dtype=bool, na_value=False)
        alerts["source_zero_num_returns"] = int(returns_zero.sum())
    slr_df["source_zero_fields"] = np.where(
        sigma_zero & returns_zero, "sigma+num_returns",
        np.where(sigma_zero, "sigma", np.where(returns_zero, "num_returns", "")),
    )

    # Class C: sigma>1 m counted alert (rows kept, adjudication elsewhere).
    if sigma is not None:
        alerts["sigma_above_1m"] = int((sigma > 1.0).sum())

    slr_df.attrs["slr_qc_alerts"] = alerts
    return slr_df


def load_slr_records(
    paths: list[str | Path],
    mission_window: tuple | None = None,
) -> pd.DataFrame:
    """Parse SLR files, dedupe overlapping monthly/daily products, run QC triage.

    Content routing happens inside ``slr_to_dataframe`` (root-cause fix A4):
    a file whose content matches no supported SLR format raises
    ``UnsupportedSLRFormatError`` -- there is no fallback guess parser and
    unrecognized input is a hard file-level failure, never a silent skip.

    ``mission_window`` (optional ``(start, end)``) gates the MERIT-II
    two-digit-year resolution in ``slr_to_dataframe`` AND the frame-level
    mission-span assertion in the triage; callers that know the satellite
    should pass ``mission_window_for_sat(sat_id)`` (``None`` when unknown
    -- then MERIT files fall back to the parser's launch-year anchor and
    are counted under the ``window_unavailable_files`` alert, never a
    crash).

    The merged frame is deduplicated on (station_id, epoch) -- overlapping
    monthly MERIT and daily CRD products of the same pass collapse to one
    row per observation -- then passed through
    ``_apply_slr_record_triage`` (shared physical assertions plus
    ``source_zero_fields`` / sigma>1 m markers and alerts).  A row that
    violates a Class-A physics assertion is a counted row-level rejection
    (Ruling-17): it stays in the frame marked ``qc_rejected=True`` with
    the counts in ``attrs["slr_qc_rejections"]`` -- metric consumers
    (``compute_slr_response``) skip the marked rows; nothing raises.
    """
    frames = []
    window_unavailable_files = 0
    for path in paths:
        df, validation = slr_to_dataframe(str(path), mission_window=mission_window)
        if validation.get("window_unavailable"):
            window_unavailable_files += 1
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["epoch"] = pd.to_datetime(merged["epoch"], utc=True)
    dedup_keys = [key for key in ("station_id", "epoch") if key in merged.columns]
    if dedup_keys:
        merged = merged.drop_duplicates(subset=dedup_keys)
    merged = merged.sort_values("epoch").reset_index(drop=True)
    result = _apply_slr_record_triage(merged, mission_span=mission_window)
    if window_unavailable_files:
        result.attrs["slr_qc_alerts"]["window_unavailable_files"] = window_unavailable_files
    return result


def compute_slr_response(
    slr_df: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    margin_hours: float = DEFAULT_MARGIN_HOURS,
) -> dict[str, Any]:
    """SLR RMS before/during/after the window plus absolute bias shift (m).

    Metric column preference matches ``extract_slr_support``: residual_m
    when available, otherwise sigma_m.
    """
    result: dict[str, Any] = {"slr_status": STATUS_MISSING}
    if slr_df is None or slr_df.empty:
        return result

    slr = slr_df.copy()
    # Ruling-17: rows marked qc_rejected by the record triage (Class-A
    # physics violations) are counted rejections -- they never contribute
    # to response metrics.
    if "qc_rejected" in slr.columns:
        slr = slr[~slr["qc_rejected"].fillna(False).astype(bool)]
    slr["epoch"] = pd.to_datetime(slr["epoch"], utc=True)
    margin = pd.Timedelta(hours=margin_hours)
    before = slr[(slr["epoch"] >= start - margin) & (slr["epoch"] < start)]
    during = slr[(slr["epoch"] >= start) & (slr["epoch"] <= end)]
    after = slr[(slr["epoch"] > end) & (slr["epoch"] <= end + margin)]
    result["slr_obs_before"] = int(len(before))
    result["slr_obs_during"] = int(len(during))
    result["slr_obs_after"] = int(len(after))

    metric_col = "residual_m" if "residual_m" in slr.columns and slr["residual_m"].notna().any() else "sigma_m"
    result["slr_metric_column"] = metric_col
    if metric_col not in slr.columns or not slr[metric_col].notna().any():
        result["slr_status"] = STATUS_INSUFFICIENT
        return result

    before_values = before[metric_col].astype(float).dropna()
    after_values = after[metric_col].astype(float).dropna()
    during_values = during[metric_col].astype(float).dropna()
    if len(before_values) < MIN_BAND_SAMPLES or len(after_values) < MIN_BAND_SAMPLES:
        result["slr_status"] = STATUS_INSUFFICIENT
        return result

    result["slr_rms_before_m"] = float(np.sqrt(np.mean(before_values**2)))
    result["slr_rms_after_m"] = float(np.sqrt(np.mean(after_values**2)))
    if len(during_values) > 0:
        result["slr_rms_during_m"] = float(np.sqrt(np.mean(during_values**2)))
    result["slr_bias_shift_m"] = float(abs(after_values.mean() - before_values.mean()))
    result["slr_status"] = STATUS_COMPUTED
    return result


# ---------------------------------------------------------------------------
# Window/target drivers
# ---------------------------------------------------------------------------


def build_target_context(raw_root: str | Path, sat_id: str) -> dict[str, Any]:
    """Build per-target file indexes and the merged TLE history exactly once."""
    target_dir = Path(raw_root) / sat_id
    tle_df = load_tle_history(target_dir / "tle")
    orbit_index = infer_orbit_product_spans([target_dir / "pod", target_dir / "precise_orbit"])
    slr_index = infer_slr_file_spans(target_dir / "slr")
    return {
        "sat_id": sat_id,
        "tle_df": tle_df,
        "orbit_index": orbit_index,
        "slr_index": slr_index,
        "has_orbit_dir": (target_dir / "pod").is_dir() or (target_dir / "precise_orbit").is_dir(),
        "has_slr_dir": (target_dir / "slr").is_dir(),
    }


def _orbit_parse_failure_detail(parse_failures: list[dict[str, str]]) -> str:
    """Compact per-file parse-failure detail for the window record (L4).

    "; "-joined ``<filename>:<ErrorType>: <message>`` entries so the
    emitted per-window row carries the counted errors' identity, not just
    their number (REVIEW_FINDINGS 6.1 L4 accounting; fix-round-1).
    """
    return "; ".join(
        f"{Path(entry['path']).name}:{entry['error']}" for entry in parse_failures
    )


def _orbit_response_for_window(context: dict, start: pd.Timestamp, end: pd.Timestamp, margin_hours: float) -> dict[str, Any]:
    selected = select_orbit_files(context["orbit_index"], start, end, margin_hours)
    if not selected:
        status = STATUS_MISSING if not context["has_orbit_dir"] else STATUS_INSUFFICIENT
        return {"orbit_status": status, "orbit_file_count": 0}

    frames = []
    rotating_frame: bool | None = None
    parse_error: str | None = None
    parse_failures: list[dict[str, str]] = []
    for row in selected:
        try:
            df, rotating = load_orbit_file(row["path"])
        except PhysicalQCViolation:
            # The production loader never lands here (it counts its own
            # QC rejections via the empty-frame return below); a
            # replacement loader that still raises keeps the blocking
            # contract (Ruling-14 compatibility, same semantics as
            # release_export._orbit_evidence_frames).
            raise
        except Exception as exc:
            # Other parse errors are COUNTED per file, not silent drops.
            parse_error = str(exc).splitlines()[0]
            parse_failures.append(
                {"path": str(row["path"]), "error": f"{type(exc).__name__}: {parse_error}"}
            )
            continue
        if not df.empty:
            frames.append(df)
            # Every selected file of one satellite must agree on the frame
            # convention: concatenated rows are analyzed under ONE flag, so a
            # mixed-frame selection must abort loudly, never average out.
            if rotating_frame is None:
                rotating_frame = rotating
            elif rotating_frame != rotating:
                raise ValueError(
                    "mixed_frame_orbit_products: selected files disagree on "
                    "rotating-frame convention (vis-viva SMA would be computed "
                    "on inconsistent velocities)"
                )
        else:
            # Ruling-17 counted file-level rejection: the loader refused
            # the file (physics identity failed) and returned empty +
            # ``orbit_qc_rejection`` -- count it exactly like any other
            # parse-unavailable file, never a silent drop.
            rejection = df.attrs.get("orbit_qc_rejection")
            if rejection:
                parse_error = f"PhysicalQCViolation: {rejection['assertion']}"
                parse_failures.append(
                    {
                        "path": str(row["path"]),
                        "error": (
                            f"PhysicalQCViolation: {rejection['assertion']} "
                            f"row_index={rejection['row_index']!r} "
                            f"value={rejection['value']!r}"
                        ),
                    }
                )
    if not frames:
        status = STATUS_PARSE_UNAVAILABLE if parse_error else STATUS_INSUFFICIENT
        return {
            "orbit_status": status,
            "orbit_file_count": int(len(selected)),
            "orbit_source": selected[0].get("source_type", ""),
            "orbit_parse_failure_count": len(parse_failures),
            "orbit_parse_failure_detail": _orbit_parse_failure_detail(parse_failures),
        }

    orbit_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["epoch"]).sort_values("epoch")
    result = compute_orbit_response(orbit_df, start, end, margin_hours=margin_hours, rotating_frame=rotating_frame)
    result["orbit_file_count"] = int(len(selected))
    result["orbit_parse_failure_count"] = len(parse_failures)
    result["orbit_parse_failure_detail"] = _orbit_parse_failure_detail(parse_failures)
    first_name = Path(selected[0]["path"]).name.upper()
    suffix = Path(selected[0]["path"]).suffix.lower().lstrip(".")
    product = "poe" if "POEORB" in first_name else "moe" if "MOEORB" in first_name else suffix
    result["orbit_source"] = f"{selected[0].get('source_type', 'orbit')}:{product}"
    return result


def _slr_response_for_window(context: dict, start: pd.Timestamp, end: pd.Timestamp, margin_hours: float) -> dict[str, Any]:
    margin = pd.Timedelta(hours=margin_hours)
    span_start, span_end = start - margin, end + margin
    paths = []
    for row in context["slr_index"]:
        row_start = _parse_utc(row["start_utc"])
        row_end = _parse_utc(row["end_utc"])
        if row_start < span_end and row_end > span_start:
            paths.append(row["path"])
    if not paths:
        status = STATUS_MISSING if not context["has_slr_dir"] else STATUS_INSUFFICIENT
        return {"slr_status": status}
    slr_df = load_slr_records(
        paths,
        mission_window=mission_window_for_sat(context.get("sat_id", "")),
    )
    if slr_df.empty:
        return {"slr_status": STATUS_PARSE_UNAVAILABLE}
    return compute_slr_response(slr_df, start, end, margin_hours=margin_hours)


def compute_window_response(window: pd.Series, context: dict, margin_hours: float = DEFAULT_MARGIN_HOURS) -> dict[str, Any]:
    """Compute the combined per-source response record for one event window."""
    start = _parse_utc(window["window_start_utc"])
    end = _parse_utc(window["window_end_utc"])
    record: dict[str, Any] = {
        "annotation_id": window.get("annotation_id", ""),
        "sat_id": context["sat_id"],
        "batch": window.get("batch", ""),
        "confidence_tier": confidence_tier(
            str(window.get("tle_status", "")),
            str(window.get("slr_status", "")),
            str(window.get("orbit_status", "")),
        ),
        "window_start_utc": _iso(start),
        "window_end_utc": _iso(end),
    }
    record.update(compute_tle_response(context["tle_df"], start, end))
    record.update(_orbit_response_for_window(context, start, end, margin_hours))
    record.update(_slr_response_for_window(context, start, end, margin_hours))
    return {column: record.get(column, "") for column in PER_WINDOW_COLUMNS}


def compute_target_responses(
    windows: pd.DataFrame,
    raw_root: str | Path,
    sat_id: str,
    margin_hours: float = DEFAULT_MARGIN_HOURS,
) -> pd.DataFrame:
    """Compute response rows for all windows of one target."""
    context = build_target_context(raw_root, sat_id)
    rows = [compute_window_response(window, context, margin_hours=margin_hours) for _, window in windows.iterrows()]
    return pd.DataFrame(rows, columns=PER_WINDOW_COLUMNS)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    ("delta_mean_motion_rad_per_min", "tle"),
    ("delta_sma_km", "tle"),
    ("delta_eccentricity", "tle"),
    ("delta_inclination_rad", "tle"),
    ("orbit_radius_shift_m", "orbit"),
    ("orbit_sma_shift_m", "orbit"),
    ("slr_bias_shift_m", "slr"),
    ("slr_rms_before_m", "slr"),
    ("slr_rms_after_m", "slr"),
]

SUMMARY_COLUMNS = [
    "sat_id",
    "confidence_tier",
    "window_count",
    "tle_computed_count",
    "orbit_computed_count",
    "slr_computed_count",
    "median_abs_delta_mean_motion_rad_per_min",
    "median_abs_delta_sma_km",
    "p25_abs_delta_sma_km",
    "p75_abs_delta_sma_km",
    "median_abs_delta_eccentricity",
    "median_abs_delta_inclination_rad",
    "median_abs_orbit_radius_shift_m",
    "p25_abs_orbit_radius_shift_m",
    "p75_abs_orbit_radius_shift_m",
    "median_abs_orbit_sma_shift_m",
    "median_slr_bias_shift_m",
    "p25_slr_bias_shift_m",
    "p75_slr_bias_shift_m",
    "median_slr_rms_before_m",
    "median_slr_rms_after_m",
]


def _summary_row(group: pd.DataFrame, sat_id: str, tier: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sat_id": sat_id,
        "confidence_tier": tier,
        "window_count": int(len(group)),
        "tle_computed_count": int((group["tle_status"] == STATUS_COMPUTED).sum()),
        "orbit_computed_count": int((group["orbit_status"] == STATUS_COMPUTED).sum()),
        "slr_computed_count": int((group["slr_status"] == STATUS_COMPUTED).sum()),
    }
    numeric = group.copy()
    for column, _ in SUMMARY_METRICS:
        if column not in numeric.columns:
            numeric[column] = np.nan
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    def _abs_median(column: str) -> float:
        values = numeric[column].abs().dropna()
        return float(values.median()) if len(values) else math.nan

    def _abs_quantile(column: str, q: float) -> float:
        values = numeric[column].abs().dropna()
        return float(values.quantile(q)) if len(values) else math.nan

    row["median_abs_delta_mean_motion_rad_per_min"] = _abs_median("delta_mean_motion_rad_per_min")
    row["median_abs_delta_sma_km"] = _abs_median("delta_sma_km")
    row["p25_abs_delta_sma_km"] = _abs_quantile("delta_sma_km", 0.25)
    row["p75_abs_delta_sma_km"] = _abs_quantile("delta_sma_km", 0.75)
    row["median_abs_delta_eccentricity"] = _abs_median("delta_eccentricity")
    row["median_abs_delta_inclination_rad"] = _abs_median("delta_inclination_rad")
    row["median_abs_orbit_radius_shift_m"] = _abs_median("orbit_radius_shift_m")
    row["p25_abs_orbit_radius_shift_m"] = _abs_quantile("orbit_radius_shift_m", 0.25)
    row["p75_abs_orbit_radius_shift_m"] = _abs_quantile("orbit_radius_shift_m", 0.75)
    row["median_abs_orbit_sma_shift_m"] = _abs_median("orbit_sma_shift_m")
    row["median_slr_bias_shift_m"] = _abs_median("slr_bias_shift_m")
    row["p25_slr_bias_shift_m"] = _abs_quantile("slr_bias_shift_m", 0.25)
    row["p75_slr_bias_shift_m"] = _abs_quantile("slr_bias_shift_m", 0.75)

    before = numeric["slr_rms_before_m"].dropna()
    after = numeric["slr_rms_after_m"].dropna()
    row["median_slr_rms_before_m"] = float(before.median()) if len(before) else math.nan
    row["median_slr_rms_after_m"] = float(after.median()) if len(after) else math.nan
    return row


def summarize_responses(per_window: pd.DataFrame) -> pd.DataFrame:
    """Per-target/per-tier response summary plus ALL roll-up rows."""
    if per_window.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    for (sat_id, tier), group in per_window.groupby(["sat_id", "confidence_tier"], sort=True):
        rows.append(_summary_row(group, str(sat_id), str(tier)))
    for sat_id, group in per_window.groupby("sat_id", sort=True):
        rows.append(_summary_row(group, str(sat_id), "ALL"))
    for tier, group in per_window.groupby("confidence_tier", sort=True):
        rows.append(_summary_row(group, "ALL", str(tier)))
    rows.append(_summary_row(per_window, "ALL", "ALL"))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
