"""Single source of truth for technical-validation experiment parameters (P4).

REVIEW_FINDINGS ``P4`` requires every experiment parameter -- analysis-window
lengths, bracket-band width, coverage margins, sampling seeds, propagation
bin edges, benchmark matching tolerances, and the TLE sigma model -- to be
defined in exactly one canonical module and referenced everywhere.  A second
numeric definition of any of these values anywhere else is a defect; when an
experiment deliberately deviates it must declare the deviation (value and
reason) in its own output table and documentation.

Sigma-model theory (the derivation behind ``sigma_km``)
------------------------------------------------------
TLE/SGP4 position error decomposes into nearly orthogonal components with
different time behavior:

1. **Radial and cross-track terms are bounded** by the epoch-state error of
   the fitted TLE (a constant of order a few hundred meters); they do not
   grow systematically with propagation distance on the <= 48 h arcs used
   here.
2. **The along-track term accumulates linearly.**  A semi-major-axis error
   ``da`` maps to a mean-motion error ``dn = -(3/2) (n/a) da`` (linearized
   Kepler's third law), so the phase (along-track) error grows as

       ds(t) ~ -(3 n / 2) * da * t,

   i.e. linearly in propagation time t with slope set by the epoch-state
   error (``g`` in km per 24 h).
3. **The drag term grows as t^2** but is negligible for the altitudes in
   this dataset over arcs <= 48 h.

**Sum form.**  For independent orthogonal components the quadrature sum
``sqrt(sigma_0^2 + (g*dt)^2)`` is the natural combination; the *linear* sum

    sigma(dt) = sigma_0 + g * dt

used throughout is a **conservative upper bound** of that quadrature
composition.  The sigma model is therefore a *consistency bound, not a
calibrated confidence interval* -- all coverage numbers computed from it are
lower-bound-style diagnostics, and the empirical calibration curve
(``sigma_calibration_curve.csv``) reports how conservative it actually is.

**Validity domain.**  The model is fitted and valid for propagation
distances ``dt <= SIGMA_VALIDITY_HOURS`` (48 h); beyond that it is an
extrapolation (the failure evidence behind REVIEW_FINDINGS B17).

**Estimation criterion (unified).**  Both parameters, ``sigma_0`` (base, km)
and ``g`` (growth, km per 24 h), are estimated by one ordinary least-squares
line through the *medians of fixed propagation bins* (``SIGMA_FIT_BIN_EDGES_24H``,
in 24 h units) of the TLE-vs-POD position residuals on the fit targets.
Using medians makes the fit robust to rare extrapolation outliers; using one
criterion for both parameters removes the earlier mixed criterion (lowest-
quartile median for sigma_0, bin-median slope for g).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Window / coverage / sampling parameters
# ---------------------------------------------------------------------------

#: Pre-event arm of the response analysis window around the event epoch.
ANALYSIS_WINDOW_PRE_HOURS = 6.0
#: Post-event arm of the response analysis window around the event epoch.
ANALYSIS_WINDOW_POST_HOURS = 24.0
#: Symmetric band around each window used to select TLE/orbit/SLR samples
#: (the "bracket band"; window duration 30 h + 2 x 12 h sampling arms).
BRACKET_BAND_HOURS = 12.0
#: Margin (days) added on both sides of a window when auditing SLR coverage
#: (sparse passes need a wider audit band than the dense orbit/TLE sources).
SLR_COVERAGE_MARGIN_DAYS = 1.0
#: Buffer (hours) excluding stable-window grid cells that fall within this
#: distance of any event window's coverage audit span (window +/- SLR
#: margin), so negative controls cannot share evidence with adjacent events.
#: Equals the post-event arm: an event's SLR audit span reaches exactly
#: window_end + 24 h, so 24 h fully shields the neighbor grid cell.
STABLE_EXCLUSION_BUFFER_HOURS = 24.0
#: Seed of the coverage-matched stable-window resampling (deterministic).
STABLE_MATCHED_SEED = 42
#: Seed of the bootstrap-resampling CI of headline statistics.
BOOTSTRAP_SEED = 42
#: Seed of the Starlink operational-subset sampling (satellites/shells).
STARLINK_SAMPLE_SEED = 20241126
#: Propagation-distance bin edges (hours) for residual curves -- the single
#: shared definition (REVIEW_FINDINGS C8: no experiment-local bin variants).
PROPAGATION_BIN_EDGES_HOURS = (0.0, 6.0, 12.0, 24.0, 48.0, 96.0, float("inf"))

# ---------------------------------------------------------------------------
# Shared physical constants
# ---------------------------------------------------------------------------

#: Spherical-Earth radius (km) subtracted from the semi-major axis to obtain
#: the SMA altitude -- the single shared definition for every table and
#: figure that reports altitude (REVIEW_FINDINGS P4).  NOTE: this is NOT
#: the WGS72 equatorial radius (6378.135 km) that SGP4's ``Satrec`` uses
#: internally for TLE element recovery (e.g. ``satrec.a`` scaling in
#: generate_kozai_comparison.py); that constant belongs to the SGP4 model
#: and deliberately stays there.
EARTH_RADIUS_KM = 6371.0

# ---------------------------------------------------------------------------
# External-benchmark matching parameters
# ---------------------------------------------------------------------------

#: Primary match tolerance against the Shorten et al. benchmark (days); the
#: benchmark carries a systematic +1 day day-of-year offset (their parsing),
#: verified against raw IDS records.
BENCHMARK_MATCH_TOLERANCE_DAYS = 1.0
#: Tolerance sensitivity sweep (days) reported alongside the primary match.
BENCHMARK_MATCH_TOLERANCE_SWEEP_DAYS = (0.5, 1.0, 2.0)

# ---------------------------------------------------------------------------
# Starlink operational-subset parameters
# ---------------------------------------------------------------------------

#: Maximum TLE age (hours) accepted when pairing Starlink ephemeris states
#: with TLEs (the disclosed upper bound; older TLEs are dropped).
STARLINK_TLE_MAX_AGE_HOURS = 12.0
#: Inclination half-band (deg) used to assign Starlink TLEs to orbital
#: shells/plane families -- the single definition shared by the
#: distribution tables and the TV figures (REVIEW_FINDINGS P4).
SHELL_TOLERANCE_DEG = 0.3

# ---------------------------------------------------------------------------
# Tier-equivalence TOST parameters (REVIEW_FINDINGS 7.1.3)
# ---------------------------------------------------------------------------

#: Two one-sided tests alpha (per test; the family is BH/FDR controlled).
TIER_EQUIVALENCE_ALPHA = 0.05
#: Pre-registered equivalence tolerance for inclination (deg).  Basis:
#: 0.01 deg is below any inclination offset with station-keeping or
#: ground-track relevance for LEO altimetry missions (maintenance deadbands
#: are ~0.02-0.05 deg), i.e. two tiers whose inclinations differ by less
#: than this are equivalent for every engineering use of this dataset.
#: Registered from mission-engineering practice BEFORE inspecting the
#: observed tier differences; it is a boundary of relevance, not a
#: post-hoc function of the measured gap.
TOST_TOLERANCE_INCLINATION_DEG = 0.01
#: Pre-registered equivalence tolerance for SMA altitude (m).  Basis: 50 m
#: is the scale of the smallest semi-major-axis offsets with operational
#: relevance (a 50 m SMA error shifts the ground-track repeat cycle by
#: ~1 km per day); smaller differences are indistinguishable for
#: conjunction screening and ground-track maintenance alike.
TOST_TOLERANCE_ALTITUDE_M = 50.0
#: Pre-registered equivalence tolerance for eccentricity (unitless).
#: Basis: derived from the altitude tolerance so the two bounds express the
#: same physical scale -- an eccentricity offset de of the radial
#: oscillation amplitude a*de must stay within the 50 m SMA tolerance at
#: the constellation's nominal a ~ 7,100 km: 50 m / 7,100 km ~ 7.0e-6.
TOST_TOLERANCE_ECCENTRICITY = 7.0e-6

#: Elements compared across tiers (columns of the equivalence input frame).
TIER_EQUIVALENCE_ELEMENTS = ("inclination_deg", "eccentricity", "sma_altitude_km")

#: Per-element equivalence tolerances in the units of the element columns
#: (inclination deg; eccentricity unitless; SMA altitude km).  Pre-registered
#: from engineering relevance (see the constants above); the ONLY place these
#: numbers exist.
TIER_EQUIVALENCE_TOLERANCES = {
    "inclination_deg": TOST_TOLERANCE_INCLINATION_DEG,
    "sma_altitude_km": TOST_TOLERANCE_ALTITUDE_M / 1000.0,
    "eccentricity": TOST_TOLERANCE_ECCENTRICITY,
}

#: Nominal Gaussian coverage at 1/2/3 sigma, for the calibration curve.
NOMINAL_COVERAGE = {1: 0.6827, 2: 0.9545, 3: 0.9973}

#: Minimum fusion-value windows per target below which the per-target
#: fusion rows are explicitly flagged (not reportable evidence; C7
#: "insufficient output warning" -- the shipped run had 10 windows on a
#: single target).
FUSION_MIN_SAMPLE_WINDOWS = 10

#: Norm-appropriate reference for the calibration curve: the residuals the
#: sigma model bounds are 3D position NORMS.  If the per-component errors
#: were Gaussian with the model sigma, the norm would follow a Maxwell
#: distribution with CDF F(k) = erf(k/sqrt(2)) - sqrt(2/pi) k e^{-k^2/2}
#: (19.87% / 73.85% / 97.07% at 1/2/3 sigma).  The 1D Gaussian values in
#: NOMINAL_COVERAGE are kept for continuity with the published 3-sigma
#: coverage numbers, but the Maxwell column is the apples-to-apples
#: reference for a norm; the calibration table reports both.
NOMINAL_COVERAGE_MAXWELL_NORM = {1: 0.1987, 2: 0.7385, 3: 0.9707}

#: Stable-window noise floors (meters) of the two SMA-shift estimators,
#: from the published stable-window validation tables (REVIEW_FINDINGS
#: 7.1.2): the TLE bracketing SMA shift has a median noise floor of
#: 24.0 m (p75 27.9 m); the precise-orbit period-averaged SMA shift a
#: median floor of 1.1 m (p75 2.08 m).  These anchor the errors-in-
#: variables treatment of the TLE-vs-orbit SMA-shift regression (Deming
#: lambda and the OLS attenuation bound): with a 1D Gaussian reference,
#: OLS of y on x attenuates as beta * Var(x_true) / (Var(x_true) +
#: sigma_x^2), so the x-noise floor of 24 m over the event-scale signal
#: variance bounds the attenuation; lambda = (sigma_y / sigma_x)^2.
SMA_SHIFT_NOISE_FLOOR_M = {
    "tle_median": 24.0,
    "tle_p75": 27.9,
    "orbit_median": 1.1,
    "orbit_p75": 2.08,
}

#: 1-sigma (m) assigned to the SECOND independent DORIS SP3 series (the
#: non-truth series in each overlap) in the fusion-value experiment.
#: EMPIRICAL, not a guess: derived from the archive overlap statistics --
#: the shipped per-window fusion table (fusion_value_per_window.csv) holds
#: the second-series error against the held-out truth series for the 10
#: jason-2 ssa-vs-gsc overlap windows, median 1205 m.  The legacy value
#: (1.0 m, "OGDR-class") was wrong by three orders of magnitude and made
#: the fused estimate mathematically identical to the second series (C7);
#: the actual inter-series difference on these overlaps is km-class, so
#: with this value the inverse-variance fusion is a genuine blend
#: (w_second/w_tle ~ 0.4 at 24 h propagation).  Re-derive after WP11
#: regeneration: the overlaps may tighten once the time-scale and arc fixes
#: land.  TLE weights come from the fitted single-source sigma model
#: (``fit_sigma_model``).  P4 deviation declaration: every fusion output
#: row carries this value in the ``second_series_sigma_m`` column.
FUSION_SECOND_SERIES_SIGMA_M = 1205.0

# ---------------------------------------------------------------------------
# Sigma model
# ---------------------------------------------------------------------------

#: Fit bins for the sigma model, in 24 h units (propagation distance).
SIGMA_FIT_BIN_EDGES_24H = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
#: Minimum samples per bin for the bin median to enter the fit.
SIGMA_FIT_BIN_MIN_SAMPLES = 5
#: Targets used to FIT the sigma model (contemporary DORIS+GNSS missions).
SIGMA_FIT_TARGETS = ("sentinel-3a", "sentinel-3b", "jason-3", "saral")
#: Held-out evaluation groups (internal validation design, not a public
#: data partition).
SIGMA_EVAL_CONTEMPORARY_TARGETS = ("cryosat-2", "sentinel-6a")
SIGMA_EVAL_HISTORICAL_TARGETS = ("jason-1", "jason-2", "topex-poseidon", "hy-2a", "swot")

#: The sigma model is fitted and valid up to this propagation distance (h);
#: beyond it every use is an extrapolation and must say so.
SIGMA_VALIDITY_HOURS = 48.0

#: Stable-window tail screening: a stable (no-event) window whose TLE
#: bracket |delta_sma| response exceeds this threshold carries the
#: ``suspect_unreported_maneuver`` quality flag in the shipped label tables
#: (kept, not dropped, for auditability).  20 m is the headline-magnitude
#: scale of the smallest reported maneuvers (median event response 20.3 m),
#: so a stable window above it is a plausible unreported orbit change
#: (e.g. commissioning raises absent from the IDS histories).
STABLE_TAIL_FLAG_THRESHOLD_M = 20.0

#: Frozen snapshot of the sigma model fitted by :func:`fit_sigma_model` on
#: the released fit-target state estimates (see provenance below).  Consumers
#: that have the estimates frame at hand must FIT and pass the model
#: explicitly; this default exists only for callers that cannot fit (e.g.
#: per-row fusion helpers).  Provenance: fitted with the unified bin-median
#: criterion (both parameters from one least-squares line through the
#: ``SIGMA_FIT_BIN_EDGES_24H`` bin medians) on the released
#: ``maneuver_event_state_estimates.csv`` rows with both TLE and orbit
#: states computed on SIGMA_FIT_TARGETS -> base 0.376 km, growth 0.302 km
#: per 24 h, 266 samples (2026-08-31).  This supersedes the legacy mixed-
#: criterion fit 0.431/0.300 (REVIEW_FINDINGS 7.3 pre-registered change 5)
#: and the intermediate snapshot 0.379/0.300 recorded before the released
#: state-estimate table was finalized; it matches the sigma_model string in
#: the shipped harmonization_sigma_split_validation.csv and
#: sigma_calibration_curve.csv.  Regenerate the snapshot by re-running the
#: tv-hardening experiment.
DEFAULT_SIGMA_MODEL: dict[str, float] = {
    "base_km": 0.376,
    "growth_km_per_24h": 0.302,
    "fit_sample_count": 266.0,
}


def sigma_km(sigma_model: dict[str, float], propagation_seconds: float) -> float:
    """TLE 1-sigma position bound (km) at a propagation distance (seconds).

    Linear conservative bound ``base_km + growth_km_per_24h * dt`` (see the
    module docstring for the derivation and the <= 48 h validity domain).
    """
    dt_24h = abs(float(propagation_seconds)) / 86400.0
    return float(sigma_model["base_km"] + sigma_model["growth_km_per_24h"] * dt_24h)


def fit_sigma_model(
    estimates: pd.DataFrame,
    fit_targets: "tuple[str, ...] | list[str] | None" = None,
) -> dict[str, float]:
    """Fit the unified TLE sigma model on TLE-vs-POD position residuals.

    Both parameters come from ONE least-squares line through the medians of
    the fixed bins :data:`SIGMA_FIT_BIN_EDGES_24H` (24 h units); rows with a
    non-computed TLE or orbit state and rows outside ``fit_targets`` never
    enter the fit.  Returns ``{"base_km", "growth_km_per_24h",
    "fit_sample_count"}``; an empty fit returns NaN parameters with count 0.
    """
    targets = tuple(fit_targets) if fit_targets is not None else SIGMA_FIT_TARGETS
    keys = ("tle_state_status", "orbit_state_status", "sat_id")
    if estimates is None or estimates.empty or not set(keys).issubset(estimates.columns):
        return {"base_km": float("nan"), "growth_km_per_24h": float("nan"), "fit_sample_count": 0}

    usable = estimates[
        (estimates["tle_state_status"] == "computed")
        & (estimates["orbit_state_status"] == "computed")
        & (estimates["sat_id"].isin(targets))
    ]
    residuals_km: list[float] = []
    propagation_24h: list[float] = []
    position_keys = ("itrs_x_m", "itrs_y_m", "itrs_z_m", "orbit_x_m", "orbit_y_m", "orbit_z_m")
    for _, row in usable.iterrows():
        try:
            diff = np.array(
                [row["itrs_x_m"] - row["orbit_x_m"], row["itrs_y_m"] - row["orbit_y_m"], row["itrs_z_m"] - row["orbit_z_m"]],
                dtype=float,
            )
            seconds = float(row["tle_propagation_seconds"])
        except (KeyError, TypeError, ValueError):
            # Rows that reach here passed the status filter but lack the
            # ITRS position columns -- the TEME->ITRS transform failed for
            # that row (see analyzers.state_estimation.teme_to_itrs, whose
            # failures are counted in TEME_ITRS_FAILURES) -- or carry a
            # non-numeric propagation field.  Such rows never enter the
            # fit; fit_sample_count reports how many rows DID enter, so the
            # skip is visible in the output.
            continue
        residuals_km.append(float(np.linalg.norm(diff) / 1000.0))
        propagation_24h.append(abs(seconds) / 86400.0)
    if not residuals_km:
        return {"base_km": float("nan"), "growth_km_per_24h": float("nan"), "fit_sample_count": 0}

    x = np.asarray(propagation_24h)
    y = np.asarray(residuals_km)
    centers: list[float] = []
    medians: list[float] = []
    for lo, hi in zip(SIGMA_FIT_BIN_EDGES_24H[:-1], SIGMA_FIT_BIN_EDGES_24H[1:]):
        sel = y[(x >= lo) & (x < hi)]
        if len(sel) >= SIGMA_FIT_BIN_MIN_SAMPLES:
            centers.append((lo + hi) / 2.0)
            medians.append(float(np.median(sel)))
    if len(centers) >= 2:
        slope, intercept = np.polyfit(np.asarray(centers), np.asarray(medians), 1)
        base = float(intercept)
        growth = float(slope)
    else:
        # Not enough populated bins for a line: fall back to the overall
        # median with zero growth (deterministic, documented degenerate fit).
        base = float(np.median(y))
        growth = 0.0
    # A physical sigma has non-negative components; clamp pathological
    # descending bin-median curves instead of emitting a negative bound.
    return {
        "base_km": max(base, 0.0),
        "growth_km_per_24h": max(growth, 0.0),
        "fit_sample_count": len(y),
    }
