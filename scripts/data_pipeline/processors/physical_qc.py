"""L2 physics-based assertions over normalized MAD-LEO records.

Every assertion implemented here is an identity or a bound with a
theoretical derivation (REVIEW_FINDINGS.md sections 6.2.2-6.2.4, 8.4);
violations are expected to be zero and always indicate a parsing /
coordinate / unit defect, never "bad data to filter away".  A violation
raises :class:`PhysicalQCViolation`; a clean frame returns a QC summary
dict (the per-record "QC ledger row") for diagnostics.

Orbit assertion set (self-contained; configs carry no nominal
semi-major axes, so mission-nominal checks are optional caller-supplied
gates on top of the LEO radius band and the vis-viva bound form of the
energy identity):

1. ``radius_in_leo_band`` -- r = |x,y,z| in [ORBIT_RADIUS_MIN_M,
   ORBIT_RADIUS_MAX_M].  Non-finite positions trip the same assertion.
2. ``vis_viva_sma_in_leo_band`` -- a = 1/(2/r - v^2/mu) in the same
   band, computed with the Earth-rotation-corrected velocity
   ``v_inertial = v_ecef + omega x r`` for Earth-fixed products
   (``rotating_frame=True``).  This IS the energy-consistency
   assertion in bound form: specific orbital energy for a bound Kepler
   orbit is eps = -mu/(2a), so requiring the vis-viva a implied by
   (r, |v_inertial|) to stay in the mission band is equivalent to
   requiring |eps - eps0| within the band's energy width.  Feeding an
   ECEF velocity field as inertial (the B14 defect class) removes an
   |omega x r| ~ 0.46-0.53 km/s component and shifts a by hundreds of
   km (e.g. jason-2 altitude: ~436 km down-shift); the band form
   catches this only below ~814 km altitude -- the lowest altitude at
   which the down-shift no longer reaches the 6578 km band floor, at
   real inclinations -- so above ~800 km the optional nominal-a gate
   (``expected_sma_m`` / ``sma_tolerance``, assertion
   ``sma_median_off_nominal``) is the effective detector, at ALL
   altitudes.
3. ``sma_median_off_nominal`` (optional gate) -- when the caller
   supplies the mission-nominal semi-major axis, the per-frame median
   vis-viva a must match it within ``sma_tolerance`` (default 25 km,
   orders of magnitude above mm-cm arc noise, far below the hundreds
   of km the defect class shifts).
4. ``epoch_present`` / ``epoch_strictly_increasing`` -- epochs parse
   (no NaT) and strictly increase (duplicates are not strictly
   increasing).
5. The QC summary carries ``sma_median_m``, the per-satellite median
   vis-viva a, as the cross-product consistency diagnostic.

The omega x r correction is first-order but effectively exact for this
magnitude-only identity: the residual ITRF->GCRS axis rotation at the
instant is orthogonal and preserves |v|, and the neglected polar-motion
tilt of omega contributes < 1 mm/s (~1e-4 relative on v^2), orders of
magnitude below the band separation.

SLR assertions (``assert_slr_records``):

- ``slr_tof_range_identity`` -- |c*tof/2 - range| <= 1 m.  c is the SI
  exact 299792458 m/s; ``range_m`` is derived from the TOF by every
  supported format, so a violation is necessarily a parsing defect
  (626k measured rows: 0 violations; REVIEW_FINDINGS 8.4 Class A).
- ``slr_range_in_band`` -- one-way range in [5e5, 6e6] m (see
  SLR_RANGE_MIN_M derivation below).
- ``slr_epoch_in_mission_span`` -- epochs inside the mission span when
  one is provided.

``assert_slr_records`` additionally accepts ``mode="report"``: instead of
raising on the first violation it returns the QC ledger augmented with
``slr_qc_rejections`` (assertion -> offending row count) and
``qc_rejected_index`` (0-based positions of every row violating any
assertion), so callers implement counted row-level rejection (mark the
rows, count them, keep them) without re-deriving the assertion logic
(Ruling-17).

TLE epoch self-consistency (``assert_tle_epoch_selfconsistency``):
SGP4 propagated at the element set's own epoch must return error-free,
inside the LEO radius band, and within TLE_SGP4_KEPLER_TOLERANCE_M of
the closed-form Keplerian position from the same elements.

Time-scale alert (``timescale_alert``, D11): for one orbit arc, the
nearest-prior TLE is SGP4-propagated to the arc samples, rotated to
ITRS, and the median residual against the product positions is
compared with TIMESCALE_ALERT_THRESHOLD_KM.  This is deliberately
**advisory** (alert-only, never raises): a time-scale mis-tag produces
a constant along-track offset (1 s <-> ~7.4 km, 37 s <-> ~275 km) that
dwarfs the threshold, while honest TLE noise can approach the bias
floor, so hits go to human adjudication (WP9 recalibration).

Example invocation (from the repository root)::

    python3 -c "import sys; sys.path.insert(0, 'scripts/data_pipeline'); \
        from processors.physical_qc import assert_orbit_records; \
        print('module ok')"
"""

from __future__ import annotations

import math
from datetime import timezone
from typing import Any

import erfa
import numpy as np
import pandas as pd
from astropy.time import Time
from sgp4.api import Satrec, jday

# Speed of light: SI definition of the metre, exact by construction.
# Imported (not redefined) from the SLR processor so the identity has a
# single constant in the repository.
from processors.slr_processor import SPEED_OF_LIGHT

# ---------------------------------------------------------------------------
# Thresholds (all derivations inline; keep as module constants)
# ---------------------------------------------------------------------------

# WGS-84 Earth gravitational constant (m^3/s^2); same value as
# analyzers.event_response.EARTH_MU_M3_PER_S2 (kept local because
# processors must not import from analyzers).
EARTH_MU_M3_PER_S2 = 3.986004418e14

# WGS-84 Earth rotation rate (rad/s).  |omega x r| at the equator:
#   r = 6378.137 km  ->  0.465 km/s
#   r = 7178.137 km  ->  0.523 km/s  (800 km altitude reference orbit)
# i.e. the ECEF-velocity-as-inertial defect class always moves |v| by
# roughly half a km/s -- hundreds of km on the vis-viva semi-major axis.
EARTH_ROTATION_RAD_PER_S = 7.2921150e-5

# LEO radius band [6,578 km, 8,378 km]:
#   lower = WGS-84 equatorial radius 6378.137 km + 200 km, the lowest
#           altitude of sustained orbital flight (below ~200 km drag
#           deorbits within days) -> 6578.137 km, rounded down to the
#           whole km;
#   upper = 6378.137 km + 2000 km, the conventional LEO upper boundary
#           (below the inner radiation belt intensity peak) -> 8378.137
#           km, rounded down.
# Every dataset target sits inside with large margin: Starlink 550 km
# (r = 6928 km) to TOPEX/Jason 1336 km (r = 7714 km); a km/m unit mix
# moves r by three orders of magnitude, far outside the band.
ORBIT_RADIUS_MIN_M = 6_578_000.0
ORBIT_RADIUS_MAX_M = 8_378_000.0

# SP3-c/d spec clock sentinel ("clock unavailable/bad").  Counted in the
# orbit QC summary as a diagnostic; the sentinel must never be treated
# as a measurement (REVIEW_FINDINGS 6.2.4).
SP3_CLOCK_SENTINEL = 999999.999999

# SLR one-way range band [500 km, 6000 km]:
#   lower = 500 km, below the lowest possible slant range of the
#           constellation (overhead pass == altitude; lowest target
#           altitude ~540 km for the Starlink slice, ~720 km for
#           CryoSat-2 in the reference subset);
#   upper = 6000 km, above the horizon-grazing slant range of the
#           highest target (alt 1336 km: sqrt((6378.137+1336)^2 -
#           6378.137^2) = 4339 km) with margin for low-elevation
#           refraction-cutoff cutoffs.
SLR_RANGE_MIN_M = 5.0e5
SLR_RANGE_MAX_M = 6.0e6

# TOF identity tolerance: 1 m.  range_m is computed from the TOF in
# every supported format (merit/CRD/legacy all derive it), so the only
# error sources are float rounding (~1e-9 m) and a genuine field/unit
# misalignment; 1 m separates those by nine orders of magnitude.
# Measured baseline: 626k rows, 0 violations (REVIEW_FINDINGS 8.4).
SLR_TOF_IDENTITY_TOLERANCE_M = 1.0

# SGP4-at-epoch vs closed-form Keplerian position from the same
# elements: the two differ by J2 short-period + mean/osculating
# transformation terms of order J2*a ~ 7-15 km for the near-circular
# reference orbits; 100 km gives ~10x safety while a mis-parsed element
# column or unit error shifts the position by hundreds to thousands of
# km (REVIEW_FINDINGS 6.2.4 "SGP4 at the epoch time propagates position
# with ~0 residual").
TLE_SGP4_KEPLER_TOLERANCE_M = 100_000.0

# D11 timescale alert threshold, km.  Calibrated against the measured
# TLE-vs-precise-orbit bias floor (source_bias_calibration.csv: worst
# target median norm 43.6 km; well-tagged single-arc max 13.7 km):
# 20 km sits above the honest-arcs envelope (13.7 km) and ~2.2x below
# the worst-target floor, so a systematic mis-tag (19 s GPS/UTC ~>
# 140 km, 32-37 s TAI/UTC ~> 240-275 km along-track) trips
# unambiguously.  Advisory by design (never blocks a build); to be
# re-frozen after WP9 calibration.
TIMESCALE_ALERT_THRESHOLD_KM = 20.0

# Cost cap for the alert: subsample the arc to at most this many evenly
# spaced samples before SGP4 propagation (median is insensitive to the
# exact sampling).
TIMESCALE_ALERT_MAX_SAMPLES = 101


class PhysicalQCViolation(ValueError):
    """A physics identity/bound is violated: a parsing defect, never data noise.

    Attributes carry the structured ledger fields: ``file`` (source file
    name or sat id fallback), ``row_index`` (0-based position in the
    checked frame), ``assertion`` (stable assertion name), ``value``
    (the offending measured value).
    """

    def __init__(self, file: str, row_index: int | None, assertion: str, value: Any) -> None:
        self.file = file
        self.row_index = row_index
        self.assertion = assertion
        self.value = value
        super().__init__(
            f"physical_qc_violation[{assertion}] file={file!r} row_index={row_index!r} "
            f"value={value!r}"
        )


def ecef_velocity_to_inertial(velocity: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Convert Earth-fixed (ECEF) velocity components to inertial.

    Single shared implementation of the rotating-frame velocity theorem
    (REVIEW_FINDINGS 6.2.2 rule 2: exactly one implementation repository-
    wide; the retired hand-written copies lived in
    ``analyzers.event_response._vis_viva_sma_m`` and
    ``processors.coordinate_transform``):

        v_inertial = v_ecef + omega x r,  omega = (0, 0, EARTH_ROTATION_RAD_PER_S)

    i.e. vx_i = vx - omega_E*y, vy_i = vy + omega_E*x, vz_i = vz.  This is
    the first-order (z-axis) form: adequate for magnitude-only identities
    (vis-viva energy), where the neglected precession/nutation/polar-motion
    terms are orthogonal and do not change |v| (see module docstring);
    full frame transforms use astropy coordinates directly.

    ``velocity``/``position`` are (N, 3) arrays (a single (3,) vector also
    works); the input arrays are never mutated (a copy is returned).
    """
    velocity = np.array(velocity, dtype=float, copy=True)
    position = np.asarray(position, dtype=float)
    velocity[..., 0] += -EARTH_ROTATION_RAD_PER_S * position[..., 1]
    velocity[..., 1] += EARTH_ROTATION_RAD_PER_S * position[..., 0]
    return velocity


def _iso(epoch: pd.Timestamp) -> str:
    ts = pd.Timestamp(epoch)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Orbit records
# ---------------------------------------------------------------------------


# Default nominal-a tolerance, m: 25 km.  Derivation: the arc-level
# noise on the median vis-viva a is mm-cm (measurement noise) to at most
# tens of meters (J2 short-period mixing), while the ECEF-as-inertial
# defect class shifts a by hundreds of km (jason-2 altitude: ~436 km);
# 25 km separates the two by more than an order of magnitude in both
# directions and is documented as the ``sma_tolerance`` default.
SMA_NOMINAL_TOLERANCE_M = 25_000.0


def assert_orbit_records(
    df: pd.DataFrame,
    sat_id: str,
    source: str | None = None,
    *,
    rotating_frame: bool = False,
    expected_sma_m: float | None = None,
    sma_tolerance: float = SMA_NOMINAL_TOLERANCE_M,
) -> dict[str, Any]:
    """Assert L2 physics identities on one normalized orbit-product frame.

    ``df`` must carry ``epoch`` (UTC-parsable), ``x``/``y``/``z`` (m);
    ``vx``/``vy``/``vz`` (m/s) enable the vis-viva assertion.  When
    ``rotating_frame`` is True the velocity components are Earth-fixed
    (ECEF) and are corrected with ``v_inertial = v_ecef + omega x r``
    before vis-viva use (first-order, see module docstring).

    When ``expected_sma_m`` is provided, the additional
    ``sma_median_off_nominal`` gate runs: ``|sma_median_m -
    expected_sma_m| <= sma_tolerance`` (default 25 km) must hold.  This
    catches the ECEF-velocity-as-inertial defect class at ALL
    altitudes -- above ~814 km the down-shift no longer reaches the
    LEO band floor, so the band form alone cannot catch it (module
    docstring).  If velocities are absent or non-finite while the gate
    was requested, the gate trips with value ``None`` (a requested
    nominal check is never silently skipped).

    Raises :class:`PhysicalQCViolation` on the first violation and
    returns the QC ledger row (dict) when clean.
    """
    ledger_file = source if source else str(sat_id)
    if df is None or len(df) == 0:
        return {
            "sat_id": sat_id,
            "source": source,
            "n_records": 0,
            "assertions": [],
        }
    missing = [column for column in ("epoch", "x", "y", "z") if column not in df.columns]
    if missing:
        raise ValueError(f"orbit_records_missing_columns:{missing}")

    # --- assertion 3: epoch present, strictly increasing, no duplicates ---
    epochs = pd.to_datetime(df["epoch"], utc=True, errors="coerce")
    nat_rows = np.flatnonzero(epochs.isna().to_numpy())
    if len(nat_rows) > 0:
        raise PhysicalQCViolation(ledger_file, int(nat_rows[0]), "epoch_present", "NaT")
    epoch_ns = epochs.astype("int64").to_numpy()
    if len(epoch_ns) > 1:
        non_increasing = np.flatnonzero(epoch_ns[1:] <= epoch_ns[:-1]) + 1
        if len(non_increasing) > 0:
            row = int(non_increasing[0])
            raise PhysicalQCViolation(
                ledger_file, row, "epoch_strictly_increasing", _iso(epochs.iloc[row])
            )

    # --- assertion 1: radius inside the LEO band (finite, in-band) ---
    xyz = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    radius_m = np.linalg.norm(xyz, axis=1)
    radius_ok = np.isfinite(radius_m) & (radius_m >= ORBIT_RADIUS_MIN_M) & (radius_m <= ORBIT_RADIUS_MAX_M)
    bad_radius = np.flatnonzero(~radius_ok)
    if len(bad_radius) > 0:
        row = int(bad_radius[0])
        raise PhysicalQCViolation(ledger_file, row, "radius_in_leo_band", float(radius_m[row]))

    # --- assertion 2: vis-viva semi-major axis inside the LEO band ---
    sma_median_m: float | None = None
    velocity_correction = "no_velocity_columns"
    velocity_rows_checked = 0
    velocity_rows_missing = 0
    if all(column in df.columns for column in ("vx", "vy", "vz")):
        velocity_correction = "omega_cross_r" if rotating_frame else "as_inertial"
        velocity = df[["vx", "vy", "vz"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        has_velocity = np.isfinite(velocity).all(axis=1)
        velocity_rows_missing = int((~has_velocity).sum())
        if rotating_frame:
            # v_inertial = v_ecef + omega x r (shared implementation above).
            velocity = ecef_velocity_to_inertial(velocity, xyz)
        speed_sq = np.sum(velocity**2, axis=1)
        checked = has_velocity & np.isfinite(speed_sq)
        velocity_rows_checked = int(checked.sum())
        if bool(checked.any()):
            inv_a = 2.0 / radius_m - speed_sq / EARTH_MU_M3_PER_S2
            sma = np.full(len(radius_m), np.nan)
            positive = checked & (inv_a > 0)
            sma[positive] = 1.0 / inv_a[positive]
            sma[checked & ~positive] = np.inf  # escape/parabolic: no bound orbit
            in_band = checked & np.isfinite(sma) & (sma >= ORBIT_RADIUS_MIN_M) & (sma <= ORBIT_RADIUS_MAX_M)
            bad_sma = np.flatnonzero(checked & ~in_band)
            if len(bad_sma) > 0:
                row = int(bad_sma[0])
                raise PhysicalQCViolation(
                    ledger_file, row, "vis_viva_sma_in_leo_band", float(sma[row])
                )
            finite_sma = sma[checked & np.isfinite(sma)]
            if len(finite_sma) > 0:
                sma_median_m = float(np.median(finite_sma))

    # --- optional nominal-a gate: median vis-viva a vs mission nominal ---
    # Catches the ECEF-velocity-as-inertial class at ALL altitudes (above
    # ~814 km the down-shift stays inside the LEO band; see docstring).
    if expected_sma_m is not None:
        if sma_median_m is None:
            # Gate requested but no finite vis-viva a available (missing
            # or non-finite velocity columns): trip loudly, never skip.
            raise PhysicalQCViolation(ledger_file, None, "sma_median_off_nominal", None)
        delta_m = abs(sma_median_m - float(expected_sma_m))
        if delta_m > sma_tolerance:
            raise PhysicalQCViolation(
                ledger_file,
                None,
                "sma_median_off_nominal",
                {"sma_median_m": sma_median_m,
                 "expected_sma_m": float(expected_sma_m),
                 "delta_m": delta_m},
            )

    # --- SP3 clock sentinel diagnostic (counted, never filtered) ---
    clock_sentinel_rows = 0
    if "clock" in df.columns:
        clock = pd.to_numeric(df["clock"], errors="coerce").to_numpy(dtype=float)
        clock_sentinel_rows = int(np.count_nonzero(np.isclose(clock, SP3_CLOCK_SENTINEL)))

    assertions = ["epoch_present", "epoch_strictly_increasing", "radius_in_leo_band"]
    if velocity_correction != "no_velocity_columns":
        assertions.append("vis_viva_sma_in_leo_band")
    if expected_sma_m is not None:
        assertions.append("sma_median_off_nominal")
    return {
        "sat_id": sat_id,
        "source": source,
        "n_records": int(len(df)),
        "radius_min_m": float(radius_m.min()),
        "radius_max_m": float(radius_m.max()),
        "radius_median_m": float(np.median(radius_m)),
        "sma_median_m": sma_median_m,
        "expected_sma_m": None if expected_sma_m is None else float(expected_sma_m),
        "sma_tolerance_m": None if expected_sma_m is None else float(sma_tolerance),
        "velocity_correction": velocity_correction,
        "velocity_rows_checked": velocity_rows_checked,
        "velocity_rows_missing": velocity_rows_missing,
        "epoch_start_utc": _iso(epochs.iloc[0]),
        "epoch_end_utc": _iso(epochs.iloc[-1]),
        "epoch_span_s": float((epoch_ns[-1] - epoch_ns[0]) / 1e9),
        "clock_sentinel_rows": clock_sentinel_rows,
        "assertions": assertions,
    }


# ---------------------------------------------------------------------------
# SLR records
# ---------------------------------------------------------------------------


def assert_slr_records(
    df: pd.DataFrame,
    sat_id: str,
    mission_span: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    source: str | None = None,
    *,
    mode: str = "raise",
) -> dict[str, Any]:
    """Assert L2 physics identities on one normalized SLR frame.

    ``df`` must carry ``epoch``, ``time_of_flight_s`` and ``range_m``.
    ``mission_span`` is an optional ``(start_utc, end_utc)`` pair; when
    given, every epoch must fall inside it.

    ``mode="raise"`` (default, the historical contract): raises
    :class:`PhysicalQCViolation` on the first violation and returns the
    QC ledger row when clean.  ``mode="report"`` never raises: every
    assertion is evaluated and the returned ledger carries two extra
    fields -- ``slr_qc_rejections`` (assertion -> offending row count,
    non-zero entries only) and ``qc_rejected_index`` (sorted 0-based
    positions of every row that violates any assertion) -- exactly the
    inputs a caller needs for a counted row-level rejection (Ruling-17:
    mark the rows ``qc_rejected``, count them, keep them; never drop,
    never abort the pipeline).

    This is the single shared implementation of the Class-A TOF identity
    (REVIEW_FINDINGS 8.4): ``analyzers.event_response.load_slr_records``
    routes its record triage Class A through this function (same
    constant ``SLR_TOF_IDENTITY_TOLERANCE_M``, same exception type, same
    NaN semantics) -- one implementation, zero divergence.
    """
    if mode not in ("raise", "report"):
        raise ValueError(f"slr_records_unknown_mode:{mode}")
    report = mode == "report"
    ledger_file = source if source else str(sat_id)
    if df is None or len(df) == 0:
        ledger = {"sat_id": sat_id, "source": source, "n_records": 0, "assertions": []}
        if report:
            ledger["slr_qc_rejections"] = {}
            ledger["qc_rejected_index"] = []
        return ledger
    missing = [column for column in ("epoch", "time_of_flight_s", "range_m") if column not in df.columns]
    if missing:
        raise ValueError(f"slr_records_missing_columns:{missing}")

    violations: dict[str, np.ndarray] = {}

    tof = pd.to_numeric(df["time_of_flight_s"], errors="coerce").to_numpy(dtype=float)
    rng = pd.to_numeric(df["range_m"], errors="coerce").to_numpy(dtype=float)

    # --- identity: range == c * tof / 2 (SI definition of the metre) ---
    identity_gap_m = np.abs(tof / 2.0 * SPEED_OF_LIGHT - rng)
    bad_identity = ~np.isfinite(identity_gap_m) | (identity_gap_m > SLR_TOF_IDENTITY_TOLERANCE_M)
    if report:
        violations["slr_tof_range_identity"] = bad_identity
    elif bad_identity.any():
        row = int(np.flatnonzero(bad_identity)[0])
        raise PhysicalQCViolation(ledger_file, row, "slr_tof_range_identity", float(identity_gap_m[row]))

    # --- one-way range band ---
    bad_range = (rng < SLR_RANGE_MIN_M) | (rng > SLR_RANGE_MAX_M)
    if report:
        violations["slr_range_in_band"] = bad_range
    elif bad_range.any():
        row = int(np.flatnonzero(bad_range)[0])
        raise PhysicalQCViolation(ledger_file, row, "slr_range_in_band", float(rng[row]))

    # --- epoch present and inside the mission span ---
    assertions = ["slr_tof_range_identity", "slr_range_in_band"]
    epochs = pd.to_datetime(df["epoch"], utc=True, errors="coerce")
    nat_mask = epochs.isna().to_numpy()
    if report:
        violations["slr_epoch_present"] = nat_mask
    elif nat_mask.any():
        raise PhysicalQCViolation(ledger_file, int(np.flatnonzero(nat_mask)[0]), "slr_epoch_present", "NaT")
    if mission_span is not None:
        span_start, span_end = mission_span[0], mission_span[1]
        if span_start is not None or span_end is not None:
            # Open-ended windows (operational satellites carry
            # ``window_end=None``): only the bounds that exist are
            # enforced.  ``pd.to_datetime(None, utc=True)`` returns None
            # (not NaT) under pandas 2.2 and ``epochs > None`` raises
            # TypeError -- hence the explicit None guards.
            outside = pd.Series(False, index=epochs.index)
            if span_start is not None:
                outside = outside | (epochs < pd.to_datetime(span_start, utc=True))
            if span_end is not None:
                outside = outside | (epochs > pd.to_datetime(span_end, utc=True))
            outside_mask = outside.to_numpy()
            if report:
                violations["slr_epoch_in_mission_span"] = outside_mask
            elif outside_mask.any():
                row = int(np.flatnonzero(outside_mask)[0])
                raise PhysicalQCViolation(
                    ledger_file, row, "slr_epoch_in_mission_span", _iso(epochs.iloc[row])
                )
            assertions.append("slr_epoch_in_mission_span")

    ledger: dict[str, Any] = {
        "sat_id": sat_id,
        "source": source,
        "n_records": int(len(df)),
        "range_min_m": float(rng.min()),
        "range_max_m": float(rng.max()),
        "range_median_m": float(np.median(rng)),
        "max_tof_identity_gap_m": float(identity_gap_m.max()),
        "epoch_start_utc": _iso(epochs.iloc[0]),
        "epoch_end_utc": _iso(epochs.iloc[-1]),
        "assertions": assertions,
    }
    if report:
        rejected = np.zeros(len(df), dtype=bool)
        counts: dict[str, int] = {}
        for assertion, mask in violations.items():
            hits = int(np.count_nonzero(mask))
            if hits:
                counts[assertion] = hits
                rejected |= mask
        ledger["slr_qc_rejections"] = counts
        ledger["qc_rejected_index"] = [int(i) for i in np.flatnonzero(rejected)]
    return ledger


# ---------------------------------------------------------------------------
# SLR evidence qc_status column (shipped with the parquets)
# ---------------------------------------------------------------------------

# Per-row QC status shipped as the ``qc_status`` column of the released SLR
# evidence parquets.  Rows are MARKED, never dropped (Ruling-17): the 20
# range/cross-target rows found by the final review stay in the release with
# an explicit status so consumers can filter without re-deriving the physics.
QC_STATUS_OK = "ok"
QC_STATUS_RANGE_IMPLAUSIBLE = "range_implausible"
QC_STATUS_CROSS_TARGET = "cross_target"
QC_STATUS_REJECTED = "qc_rejected"

# Targets observed inside a LEO target's provider file that are known foreign
# objects (MEO GNSS satellites).  Only provably-foreign names are listed;
# permissive name variants of the file's own target (e.g. 'sentinel' inside
# sentinel-3a files) stay ``ok``.
KNOWN_FOREIGN_SLR_TARGETS = frozenset({"compassi6b"})


def slr_qc_status(
    frame: pd.DataFrame,
    qc_rejected: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    """Per-row ``qc_status`` string for a staged SLR evidence frame.

    Status precedence (most specific diagnosis wins):

    1. ``cross_target`` -- ``target_name`` is a known foreign object
       (``KNOWN_FOREIGN_SLR_TARGETS``, e.g. COMPASS-I6B normal points inside
       a sentinel-3a provider file);
    2. ``range_implausible`` -- one-way ``range_m`` outside the shared
       ``[SLR_RANGE_MIN_M, SLR_RANGE_MAX_M]`` band;
    3. ``qc_rejected`` -- row flagged by the record triage
       (``_apply_slr_record_triage`` boolean) for any other Class-A
       assertion hit (TOF identity, epoch presence, mission span);
    4. ``ok``.

    The function is pure and deterministic over the shipped columns, so the
    release staging path and any post-hoc patch of already-written parquets
    compute identical values.  ``qc_rejected`` may be None (patch path:
    only the range/target diagnoses are recomputed, which covers every
    flagged row in the current release).
    """
    rng = pd.to_numeric(frame["range_m"], errors="coerce")
    out_of_band = (rng < SLR_RANGE_MIN_M) | (rng > SLR_RANGE_MAX_M)
    status = pd.Series(
        np.where(out_of_band.to_numpy(dtype=bool, na_value=False), QC_STATUS_RANGE_IMPLAUSIBLE, QC_STATUS_OK),
        index=frame.index,
        dtype=object,
    )
    if "target_name" in frame.columns:
        names = frame["target_name"].astype("string").str.strip().str.lower()
        foreign = names.isin(KNOWN_FOREIGN_SLR_TARGETS).to_numpy(dtype=bool, na_value=False)
        status = status.mask(pd.Series(foreign, index=frame.index), QC_STATUS_CROSS_TARGET)
    if qc_rejected is not None:
        rejected = pd.Series(np.asarray(qc_rejected, dtype=bool), index=frame.index)
        status = status.mask(rejected & status.eq(QC_STATUS_OK), QC_STATUS_REJECTED)
    return status


# ---------------------------------------------------------------------------
# TLE epoch self-consistency
# ---------------------------------------------------------------------------


def _kepler_position_m(sat: Satrec) -> np.ndarray:
    """Closed-form Keplerian position (m, TEME axes) from a Satrec's elements."""
    n_rad_s = float(sat.no) / 60.0  # sat.no: Kozai mean motion, rad/min
    a_m = (EARTH_MU_M3_PER_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
    e = float(sat.ecco)
    mean_anomaly = float(sat.mo)
    eccentric = mean_anomaly
    for _ in range(64):  # Newton on E - e sin E - M (converges in <10 for e<<1)
        eccentric -= (eccentric - e * math.sin(eccentric) - mean_anomaly) / (
            1.0 - e * math.cos(eccentric)
        )
    x_perifocal = a_m * (math.cos(eccentric) - e)
    y_perifocal = a_m * math.sqrt(1.0 - e * e) * math.sin(eccentric)
    cos_o, sin_o = math.cos(float(sat.nodeo)), math.sin(float(sat.nodeo))
    cos_i, sin_i = math.cos(float(sat.inclo)), math.sin(float(sat.inclo))
    cos_w, sin_w = math.cos(float(sat.argpo)), math.sin(float(sat.argpo))
    # r_IJK = Rz(raan) Rx(inclo) Rz(argp) r_PQW (active rotations)
    r1 = np.array(
        [cos_w * x_perifocal - sin_w * y_perifocal, sin_w * x_perifocal + cos_w * y_perifocal, 0.0]
    )
    r2 = np.array([r1[0], cos_i * r1[1] - sin_i * r1[2], sin_i * r1[1] + cos_i * r1[2]])
    return np.array(
        [cos_o * r2[0] - sin_o * r2[1], sin_o * r2[0] + cos_o * r2[1], r2[2]]
    )


def assert_tle_epoch_selfconsistency(
    line1: str, line2: str, source: str | None = None
) -> dict[str, Any]:
    """SGP4 epoch self-check for one TLE line pair.

    Propagating the element set at its own epoch must be error-free,
    land inside the LEO radius band, and agree with the closed-form
    Keplerian position built from the same elements to within
    TLE_SGP4_KEPLER_TOLERANCE_M.  Malformed lines raise ``ValueError``
    from :meth:`sgp4.api.Satrec.twoline2rv` itself.
    """
    sat = Satrec.twoline2rv(line1, line2)
    epoch_jd = sat.jdsatepoch + sat.jdsatepochF
    error, position_km, _ = sat.sgp4(epoch_jd, 0.0)
    if error != 0:
        raise PhysicalQCViolation(source or str(sat.satnum), None, "tle_sgp4_epoch_propagation", int(error))
    position_m = np.asarray(position_km, dtype=float) * 1000.0
    radius_m = float(np.linalg.norm(position_m))
    if not (ORBIT_RADIUS_MIN_M <= radius_m <= ORBIT_RADIUS_MAX_M):
        raise PhysicalQCViolation(
            source or str(sat.satnum), None, "tle_sgp4_radius_in_leo_band", radius_m
        )
    kepler_distance_m = float(np.linalg.norm(_kepler_position_m(sat) - position_m))
    if kepler_distance_m > TLE_SGP4_KEPLER_TOLERANCE_M:
        raise PhysicalQCViolation(
            source or str(sat.satnum), None, "tle_sgp4_vs_kepler_distance", kepler_distance_m
        )
    return {
        "norad_id": str(sat.satnum),
        "source": source,
        "epoch_jd_utc": float(epoch_jd),
        "radius_at_epoch_m": radius_m,
        "sgp4_vs_kepler_distance_m": kepler_distance_m,
        "assertions": [
            "tle_sgp4_epoch_propagation",
            "tle_sgp4_radius_in_leo_band",
            "tle_sgp4_vs_kepler_distance",
        ],
    }


# ---------------------------------------------------------------------------
# D11 time-scale alert (advisory, never raises)
# ---------------------------------------------------------------------------


def _utc_astropy_time(ts: Any) -> Time:
    """Normalize a timestamp (pd.Timestamp/datetime/str) to an astropy UTC Time."""
    if isinstance(ts, pd.Timestamp):
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return Time(ts.to_pydatetime(), scale="utc")
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return Time(ts, scale="utc")


def _teme_to_itrs_first_order(position_m: np.ndarray, ts: Any) -> np.ndarray:
    """Rotate a TEME position (m) into ITRS to first order.

    Error budget vs the full IAU-2006/2000A chain (adequate for the
    20 km advisory threshold, total < 1 km):

    - UT1 taken as UTC: |UT1-UTC| <= 0.9 s by IERS convention ->
      cross-range error <= 0.9 s * omega_E * r ~ 0.47 km;
    - polar motion ignored: <= ~10 m;
    - apparent sidereal time via ERFA gst06a (nutation/equation of the
      equinoxes included; leap seconds from the ERFA/IERS table, the
      same authority as processors.timebase).
    """
    t = _utc_astropy_time(ts)
    theta = erfa.gst06a(t.utc.jd1, t.utc.jd2, t.tt.jd1, t.tt.jd2)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y, z = float(position_m[0]), float(position_m[1]), float(position_m[2])
    return np.array([cos_t * x + sin_t * y, -sin_t * x + cos_t * y, z])


def _sgp4_jd_fr(ts: pd.Timestamp) -> tuple[float, float]:
    dt = ts.tz_convert("UTC").tz_localize(None).to_pydatetime() if ts.tz is not None else ts.to_pydatetime()
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


def timescale_alert(
    tle_records: pd.DataFrame,
    orbit_df: pd.DataFrame,
    sat_id: str,
    *,
    threshold_km: float = TIMESCALE_ALERT_THRESHOLD_KM,
    max_samples: int = TIMESCALE_ALERT_MAX_SAMPLES,
) -> dict[str, Any]:
    """D11 time-scale cross-check: nearest-prior-TLE SGP4 vs an orbit arc.

    ``tle_records`` is a DataFrame with ``line1``/``line2`` columns and
    (optionally) ``epoch`` -- the layout produced by
    ``analyzers.state_estimation.load_tle_line_history``; when ``epoch``
    is absent the Satrec's own epoch is used.  ``orbit_df`` must carry
    ``epoch`` plus ``x``/``y``/``z`` in an Earth-fixed frame (ITRF/ECEF,
    meters) -- the contract of every product for which
    ``event_response.load_orbit_file`` returns ``rotating_frame=True``
    and of the ITRF-family SP3 products.

    The nearest TLE at or before the arc start is propagated with SGP4
    to (at most) ``max_samples`` evenly spaced arc samples, rotated to
    ITRS, and compared with the product positions; the median residual
    in km is reported against ``threshold_km``.

    Returns ``{'alert': bool, 'median_km': ..., 'reason': ...}``.
    **Never raises** -- this is an advisory QC signal (a hit means
    "suspected time-scale mis-tag / stale TLE, adjudicate manually",
    REVIEW_FINDINGS 6.2.4 last row and the 43.6 km calibration note).
    """
    base: dict[str, Any] = {"sat_id": sat_id, "threshold_km": threshold_km, "alert": False, "median_km": None}
    if orbit_df is None or orbit_df.empty or "epoch" not in orbit_df.columns:
        return {**base, "reason": "orbit_df_unusable"}
    orbit_epochs = pd.to_datetime(orbit_df["epoch"], utc=True, errors="coerce")
    usable = orbit_epochs.notna().to_numpy()
    if not bool(usable.any()) or not all(c in orbit_df.columns for c in ("x", "y", "z")):
        return {**base, "reason": "orbit_df_unusable"}

    satrecs: list[tuple[pd.Timestamp, Satrec]] = []
    if tle_records is not None and not tle_records.empty:
        for _, row in tle_records.iterrows():
            try:
                sat = Satrec.twoline2rv(str(row["line1"]), str(row["line2"]))
            except Exception:
                continue
            if "epoch" in tle_records.columns and pd.notna(row.get("epoch")):
                epoch = pd.to_datetime(row["epoch"], utc=True)
            else:
                epoch = Time(sat.jdsatepoch + sat.jdsatepochF, format="jd", scale="utc").to_datetime()
                epoch = pd.Timestamp(epoch, tz="UTC")
            satrecs.append((epoch, sat))
    if not satrecs:
        return {**base, "reason": "no_usable_tle_records"}

    arc_positions = orbit_df.loc[usable, ["x", "y", "z"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    arc_epochs = orbit_epochs[usable].reset_index(drop=True)
    arc_start = arc_epochs.iloc[0]
    arc_end = arc_epochs.iloc[-1]

    prior = [(epoch, sat) for epoch, sat in satrecs if epoch <= arc_start]
    if not prior:
        return {**base, "reason": "no_prior_tle"}
    tle_epoch, sat = max(prior, key=lambda item: item[0])

    n_rows = len(arc_epochs)
    sample_index = np.unique(
        np.linspace(0, n_rows - 1, min(n_rows, max(max_samples, 1))).round().astype(int)
    )
    residuals_km: list[float] = []
    for i in sample_index:
        ts = arc_epochs.iloc[int(i)]
        try:
            jd, fr = _sgp4_jd_fr(ts)
            error, position_km, _ = sat.sgp4(jd, fr)
            if error != 0:
                continue
            predicted_m = _teme_to_itrs_first_order(
                np.asarray(position_km, dtype=float) * 1000.0, ts
            )
        except Exception:
            continue
        residual_km = float(np.linalg.norm(predicted_m - arc_positions[int(i)]) / 1000.0)
        if math.isfinite(residual_km):
            residuals_km.append(residual_km)
    if not residuals_km:
        return {**base, "reason": "sgp4_no_valid_samples"}

    median_km = float(np.median(residuals_km))
    return {
        **base,
        "alert": bool(median_km > threshold_km),
        "median_km": median_km,
        "max_km": float(max(residuals_km)),
        "n_samples": len(residuals_km),
        "n_tle_records": len(satrecs),
        "tle_epoch_utc": _iso(tle_epoch),
        "propagation_span_s": float((arc_end - tle_epoch).total_seconds()),
        "arc_start_utc": _iso(arc_start),
        "arc_end_utc": _iso(arc_end),
        "reason": None,
    }
