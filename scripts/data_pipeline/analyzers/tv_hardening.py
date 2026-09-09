"""Technical-validation hardening analyses (Task 14).

Implements the strict-reviewer fixes:

- Response floor (TV-5): near-zero-response events vs the TLE resolution
  floor, stratified by target and TLE bracketing gap.
- Uncertainty strata (TV-6): annotation time-uncertainty distribution.
- Target-heldout sigma model (TV-2): fit the TLE propagation sigma on a
  subset of targets, evaluate 3-sigma coverage on disjoint evaluation targets
  (an internal validation design, not a public data partition).
- Coverage-matched stable windows (TV-4): resample stable windows to match
  the event windows' evidence-coverage distribution per target.
- Fusion-value experiment (TV-1): fuse TLE with a degraded orbit product
  (Sentinel-3 MOEORB) and compare against held-out POEORB.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from benchmarking.experiment_params import (
    FUSION_MIN_SAMPLE_WINDOWS,
    NOMINAL_COVERAGE,
    NOMINAL_COVERAGE_MAXWELL_NORM,
    SIGMA_EVAL_CONTEMPORARY_TARGETS,
    SIGMA_EVAL_HISTORICAL_TARGETS,
    SIGMA_FIT_TARGETS,
    SIGMA_VALIDITY_HOURS,
    SMA_SHIFT_NOISE_FLOOR_M,
    STABLE_MATCHED_SEED,
    TIER_EQUIVALENCE_ALPHA,
    TIER_EQUIVALENCE_ELEMENTS,
    TIER_EQUIVALENCE_TOLERANCES,
    fit_sigma_model,
    sigma_km,
)

# Historical import surface (compat re-exports): the sigma-model targets and
# the fitter itself now live in benchmarking.experiment_params (P4 single
# source); these names keep existing importers working.
__all__ = [
    "SIGMA_FIT_TARGETS",
    "SIGMA_EVAL_HISTORICAL_TARGETS",
    "fit_tle_sigma_model",
    "sigma_km",
]


# ---------------------------------------------------------------------------
# TV-5: response floor
# ---------------------------------------------------------------------------


def response_floor_rows(response: pd.DataFrame) -> pd.DataFrame:
    """Near-zero-response events per target with TLE bracketing-gap context."""
    frame = response.dropna(subset=["delta_sma_km"]).copy()
    frame["abs_resp_m"] = frame["delta_sma_km"].abs() * 1000.0
    if {"tle_nearest_before_utc", "tle_nearest_after_utc"}.issubset(frame.columns):
        gap = (
            pd.to_datetime(frame["tle_nearest_after_utc"], utc=True)
            - pd.to_datetime(frame["tle_nearest_before_utc"], utc=True)
        ).dt.total_seconds() / 3600.0
        frame["bracket_gap_hours"] = gap
    rows = []
    for sat_id, group in frame.groupby("sat_id"):
        rows.append(
            {
                "sat_id": sat_id,
                "event_count": len(group),
                "near_zero_count_lt1m": int((group["abs_resp_m"] < 1).sum()),
                "near_zero_fraction": float((group["abs_resp_m"] < 1).mean()),
                "median_resp_m": float(group["abs_resp_m"].median()),
                "median_bracket_gap_hours": float(group["bracket_gap_hours"].median())
                if "bracket_gap_hours" in group.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TV-6: annotation time-uncertainty strata
# ---------------------------------------------------------------------------


def uncertainty_strata_rows(annotations: pd.DataFrame) -> pd.DataFrame:
    """Per-target annotation time-uncertainty distribution."""
    rows = []
    unc = pd.to_numeric(annotations["time_uncertainty_seconds"], errors="coerce")
    frame = annotations.assign(_unc_h=unc / 3600.0)
    for sat_id, group in frame.groupby("sat_id"):
        rows.append(
            {
                "sat_id": sat_id,
                "event_count": len(group),
                "sub_hour_uncertainty_count": int((group["_unc_h"] < 1).sum()),
                "hour_level_count": int(((group["_unc_h"] >= 1) & (group["_unc_h"] < 24)).sum()),
                "day_level_count": int((group["_unc_h"] >= 24).sum()),
                "median_uncertainty_hours": float(group["_unc_h"].median()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TV-2: target-heldout sigma model
# ---------------------------------------------------------------------------


def fit_tle_sigma_model(
    estimates: pd.DataFrame,
    fit_targets: "list[str] | tuple[str, ...] | None" = None,
) -> dict[str, float]:
    """Fit TLE base sigma + propagation growth on the fit targets only.

    Compat wrapper: the canonical implementation (unified criterion -- both
    parameters from one least-squares fit through the fixed bin medians) lives
    in :func:`benchmarking.experiment_params.fit_sigma_model`.
    """
    return fit_sigma_model(estimates, fit_targets=fit_targets)


def _usable_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    return estimates[(estimates["tle_state_status"] == "computed") & (estimates["orbit_state_status"] == "computed")]


def _residual_and_seconds(row: pd.Series) -> tuple[float, float] | None:
    """TLE-vs-POD position residual (km) and |propagation| (s) for one row."""
    try:
        diff = np.array(
            [row["itrs_x_m"] - row["orbit_x_m"], row["itrs_y_m"] - row["orbit_y_m"], row["itrs_z_m"] - row["orbit_z_m"]],
            dtype=float,
        )
        seconds = float(row["tle_propagation_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    return float(np.linalg.norm(diff) / 1000.0), abs(seconds)


def sigma_coverage_by_eval_group(estimates: pd.DataFrame, model: dict[str, float]) -> pd.DataFrame:
    """Evaluate the fitted sigma model's 3-sigma coverage per evaluation group."""
    usable = _usable_estimates(estimates)
    rows = []
    for label, targets in (
        ("fit", SIGMA_FIT_TARGETS),
        ("eval_contemporary", SIGMA_EVAL_CONTEMPORARY_TARGETS),
        ("eval_historical", SIGMA_EVAL_HISTORICAL_TARGETS),
    ):
        group = usable[usable["sat_id"].isin(targets)]
        residuals = []
        coverage = []
        for _, row in group.iterrows():
            pair = _residual_and_seconds(row)
            if pair is None:
                continue
            residual_km, seconds = pair
            residuals.append(residual_km)
            coverage.append(residual_km <= 3 * sigma_km(model, seconds))
        if residuals:
            rows.append(
                {
                    "sigma_eval_group": label,
                    "sample_count": len(residuals),
                    "median_residual_km": float(np.median(residuals)),
                    "within_3sigma_fraction": float(np.mean(coverage)),
                }
            )
    return pd.DataFrame(rows)


def sigma_calibration_rows(
    estimates: pd.DataFrame,
    model: dict[str, float],
    k_values: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Nominal vs empirical coverage of the sigma model at 1/2/3 sigma.

    Replaces the single 3-sigma coverage point with a calibration curve
    (REVIEW_FINDINGS 7.1.4).  Empirical coverage is computed only on samples
    inside the model's validity domain (propagation <= SIGMA_VALIDITY_HOURS);
    samples beyond it are counted in ``beyond_validity_domain_count`` and
    excluded from the curve (the model is an extrapolation there).

    Two nominal references are reported.  ``nominal_coverage`` is the 1D
    Gaussian value (continuity with the published coverage numbers);
    ``nominal_coverage_maxwell_norm`` is the norm-appropriate value: the
    residuals are 3D position NORMS, and if the per-component errors were
    Gaussian at the model sigma the norm would be Maxwell-distributed
    (19.87% / 73.85% / 97.07% at 1/2/3 sigma).  Reading the curve: empirical
    coverage BELOW the reference means the bound UNDER-covers at that k
    (too tight as a consistency bound there); coverage ABOVE the reference
    means it is conservative at that k.  The sigma model is a consistency
    bound (linear conservative sum, see ``experiment_params``), so the
    calibration curve quantifies how tight it actually is at each multiple
    -- it is not a calibrated confidence statement in either direction.
    """
    usable = _usable_estimates(estimates)
    groups = (
        ("all", None),
        ("fit", SIGMA_FIT_TARGETS),
        ("eval_contemporary", SIGMA_EVAL_CONTEMPORARY_TARGETS),
        ("eval_historical", SIGMA_EVAL_HISTORICAL_TARGETS),
    )
    rows = []
    for label, targets in groups:
        group = usable if targets is None else usable[usable["sat_id"].isin(targets)]
        pairs = [p for p in (_residual_and_seconds(row) for _, row in group.iterrows()) if p is not None]
        if not pairs:
            continue
        residuals = np.asarray([p[0] for p in pairs])
        seconds = np.asarray([p[1] for p in pairs])
        in_domain = seconds <= SIGMA_VALIDITY_HOURS * 3600.0
        bounds = np.asarray([sigma_km(model, s) for s in seconds])
        for k in k_values:
            covered = residuals <= k * bounds
            rows.append(
                {
                    "sigma_eval_group": label,
                    "k_sigma": int(k),
                    "nominal_coverage": NOMINAL_COVERAGE[int(k)],
                    "nominal_coverage_maxwell_norm": NOMINAL_COVERAGE_MAXWELL_NORM[int(k)],
                    "empirical_coverage": float(covered[in_domain].mean()) if in_domain.any() else float("nan"),
                    "sample_count": int(in_domain.sum()),
                    "beyond_validity_domain_count": int((~in_domain).sum()),
                    "median_residual_km": float(np.median(residuals[in_domain])) if in_domain.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TV-4: coverage-matched stable windows
# ---------------------------------------------------------------------------


def coverage_matched_stable(stable: pd.DataFrame, alignment: pd.DataFrame, seed: int = STABLE_MATCHED_SEED) -> pd.DataFrame:
    """Resample stable windows per target to match the events' coverage tiers.

    Event tiers come from the alignment table (aligned=True -> A else C/B).
    Stable windows carry their own confidence_tier column. Matching is
    deterministic (fixed seed, stable sort) and never drops events.
    """
    event_tier = alignment.assign(tier=np.where(alignment["aligned"].astype(str) == "True", "A", "C"))
    matched_frames = []
    for sat_id, event_group in event_tier.groupby("sat_id"):
        target_a_frac = float((event_group["tier"] == "A").mean())
        stable_group = stable[stable["sat_id"].astype(str) == sat_id].copy()
        if stable_group.empty:
            continue
        stable_group = stable_group.sort_values("window_start_utc").reset_index(drop=True)
        a_pool = stable_group[stable_group["confidence_tier"] == "A"]
        other_pool = stable_group[stable_group["confidence_tier"] != "A"]
        if a_pool.empty or target_a_frac == 0:
            continue  # cannot match this target's event coverage profile
        # Down-sample BOTH pools so the matched subset's A fraction tracks the
        # events' A fraction (round-to-nearest on both sides).
        n_a_pool = len(a_pool)
        n_total = min(len(stable_group), int(n_a_pool / target_a_frac))
        n_a_target = min(n_a_pool, int(round(target_a_frac * n_total)))
        n_other = min(len(other_pool), n_total - n_a_target)
        take_a = a_pool.sample(n=n_a_target, random_state=seed) if n_a_target else a_pool.iloc[0:0]
        take_other = other_pool.sample(n=n_other, random_state=seed) if n_other else other_pool.iloc[0:0]
        matched_frames.append(pd.concat([take_a, take_other]).sort_values("window_start_utc"))
    if not matched_frames:
        return pd.DataFrame()
    return pd.concat(matched_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# TV-1: fusion-value experiment (TLE + degraded orbit vs held-out precise)
# ---------------------------------------------------------------------------


def fusion_value_rows(fused: pd.DataFrame, min_windows: int = FUSION_MIN_SAMPLE_WINDOWS) -> pd.DataFrame:
    """Compare estimator errors against held-out precise orbit.

    Expects a frame with per-window TLE-only error (`tle_error_km`), second-
    series error (`degraded_error_km`), and fused error (`fused_error_km`)
    computed by the caller against the held-out truth series.  Targets with
    fewer than ``min_windows`` fusion windows are flagged in
    ``below_min_sample`` -- their rows are an explicit insufficiency
    warning, not reportable fusion evidence (C7).
    """
    rows = []
    for sat_id, group in fused.groupby("sat_id"):
        rows.append(
            {
                "sat_id": sat_id,
                "sample_count": len(group),
                "below_min_sample": bool(len(group) < min_windows),
                "tle_only_median_km": float(group["tle_error_km"].median()),
                "degraded_only_median_km": float(group["degraded_error_km"].median()),
                "fused_median_km": float(group["fused_error_km"].median()),
                # Ratio of TLE-only to fused median error; below 1.0 means the
                # fused estimator is worse than TLE-only for this target.
                "tle_over_fused_median_ratio": float(group["tle_error_km"].median() / max(group["fused_error_km"].median(), 1e-9)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier-stratified validation (S1-S4) and gap-taxonomy completeness (S5)
# ---------------------------------------------------------------------------


def load_window_tiers(annotated_root) -> pd.DataFrame:
    """Load per-window confidence tiers from annotated event windows."""
    import glob as _glob

    frames = []
    for path in sorted(_glob.glob(str(annotated_root) + "/*/annotated_event_windows.csv")):
        frame = pd.read_csv(path)
        frames.append(frame[["annotation_id", "confidence_tier"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tier_response_rows(response: pd.DataFrame, tiers: pd.DataFrame) -> pd.DataFrame:
    """S1: response magnitude by tier + KS tests (A-vs-C; H5: A-vs-B, both sources)."""
    from scipy.stats import ks_2samp

    merged = response if "confidence_tier" in response.columns else response.merge(tiers, on="annotation_id", how="left")
    merged = merged.dropna(subset=["delta_sma_km"])
    merged["abs_resp_m"] = merged["delta_sma_km"].abs() * 1000.0
    has_orbit = "orbit_mean_sma_shift_m" in merged.columns
    if has_orbit:
        merged["orbit_abs_resp_m"] = merged["orbit_mean_sma_shift_m"].abs()
    rows = []
    for tier, group in merged.groupby("confidence_tier"):
        orbit = group["orbit_abs_resp_m"].dropna() if has_orbit else pd.Series(dtype=float)
        rows.append(
            {
                "confidence_tier": tier,
                "sample_count": len(group),
                "median_abs_resp_m": float(group["abs_resp_m"].median()),
                "p25_abs_resp_m": float(group["abs_resp_m"].quantile(0.25)),
                "p75_abs_resp_m": float(group["abs_resp_m"].quantile(0.75)),
                "orbit_sample_count": len(orbit),
                "orbit_median_abs_resp_m": float(orbit.median()) if len(orbit) else float("nan"),
                "orbit_p75_abs_resp_m": float(orbit.quantile(0.75)) if len(orbit) else float("nan"),
            }
        )
    result = pd.DataFrame(rows)
    a = merged.loc[merged["confidence_tier"] == "A", "abs_resp_m"]
    c = merged.loc[merged["confidence_tier"] == "C", "abs_resp_m"]
    if len(a) > 10 and len(c) > 10:
        ks = ks_2samp(a, c)
        result.attrs["ks_A_vs_C"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
    b = merged.loc[merged["confidence_tier"] == "B", "abs_resp_m"]
    if len(a) > 10 and len(b) > 10:
        ks_ab = ks_2samp(a, b)
        result.attrs["ks_A_vs_B"] = {"statistic": float(ks_ab.statistic), "pvalue": float(ks_ab.pvalue)}
    if has_orbit:
        orbit_a = merged.loc[merged["confidence_tier"] == "A", "orbit_abs_resp_m"].dropna()
        orbit_b = merged.loc[merged["confidence_tier"] == "B", "orbit_abs_resp_m"].dropna()
        if len(orbit_a) > 10 and len(orbit_b) > 10:
            ks_orbit = ks_2samp(orbit_a, orbit_b)
            result.attrs["orbit_ks_A_vs_B"] = {"statistic": float(ks_orbit.statistic), "pvalue": float(ks_orbit.pvalue)}
    return result


def tier_sign_agreement_rows(response: pd.DataFrame, tiers: pd.DataFrame) -> pd.DataFrame:
    """S2: TLE-vs-POD sign agreement by tier."""
    merged = response if "confidence_tier" in response.columns else response.merge(tiers, on="annotation_id", how="left")
    merged = merged.dropna(subset=["delta_sma_km", "orbit_mean_sma_shift_m"])
    merged["sign_agree"] = np.sign(merged["delta_sma_km"]) == np.sign(merged["orbit_mean_sma_shift_m"])
    rows = []
    for tier, group in merged.groupby("confidence_tier"):
        rows.append(
            {
                "confidence_tier": tier,
                "sample_count": len(group),
                "sign_agreement": float(group["sign_agree"].mean()),
            }
        )
    return pd.DataFrame(rows)


def tier_sigma_coverage_rows(estimates: pd.DataFrame, tiers: pd.DataFrame, model: dict[str, float]) -> pd.DataFrame:
    """S3: leave-one-out 3-sigma coverage by confidence tier."""
    merged = estimates if "confidence_tier" in estimates.columns else estimates.merge(tiers, on="annotation_id", how="left")
    usable = merged[(merged["tle_state_status"] == "computed") & (merged["orbit_state_status"] == "computed")].copy()
    rows = []
    for tier, group in usable.groupby("confidence_tier"):
        coverage = []
        for _, row in group.iterrows():
            pair = _residual_and_seconds(row)
            if pair is None:
                continue
            residual_km, seconds = pair
            coverage.append(residual_km <= 3 * sigma_km(model, seconds))
        if coverage:
            rows.append(
                {
                    "confidence_tier": tier,
                    "sample_count": len(coverage),
                    "within_3sigma_fraction": float(np.mean(coverage)),
                }
            )
    return pd.DataFrame(rows)


def matched_specificity_rows(event_response: pd.DataFrame, stable_response: pd.DataFrame, matched_stable: pd.DataFrame) -> pd.DataFrame:
    """S4: specificity using only coverage-matched stable windows."""
    stable_ids = set(matched_stable["annotation_id"].astype(str))
    stable_sub = stable_response[stable_response["annotation_id"].astype(str).isin(stable_ids)]
    events = event_response["delta_sma_km"].abs().dropna() * 1000.0
    stable = stable_sub["delta_sma_km"].abs().dropna() * 1000.0
    return pd.DataFrame(
        [
            {"group": "events", "count": len(events), "median_abs_resp_m": float(events.median()), "frac_lt_5m": float((events < 5).mean())},
            {"group": "stable_matched", "count": len(stable), "median_abs_resp_m": float(stable.median()), "frac_lt_5m": float((stable < 5).mean())},
        ]
    )


def gap_taxonomy_completeness(alignment: pd.DataFrame) -> dict[str, Any]:
    """S5: every non-aligned window must carry a missing-source reason."""
    not_aligned = alignment[alignment["aligned"].astype(str) != "True"]
    reasons = not_aligned["missing_sources"].fillna("")
    return {
        "not_aligned_count": int(len(not_aligned)),
        "with_reason_count": int((reasons != "").sum()),
        "completeness": float((reasons != "").mean()) if len(not_aligned) else 1.0,
    }


# ---------------------------------------------------------------------------
# Tier equivalence: effect sizes + TOST + BH/FDR (REVIEW_FINDINGS 7.1.3)
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvalues: "list[float] | np.ndarray") -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up FDR control).

    Standard step-up procedure: sort the m p-values ascending, form
    ``p_(i) * m / i``, enforce monotonicity from the largest down, clip to 1.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty(m, dtype=float)
    result[order] = adjusted
    return result


def _tost_pvalue(a: np.ndarray, b: np.ndarray, tolerance: float) -> tuple[float, dict[str, float]]:
    """Two one-sided Welch t-tests of H0: |mean(a) - mean(b)| >= tolerance.

    Returns ``(p_tost, diagnostics)`` where ``p_tost = max(p_lower, p_upper)``
    and the equivalence null is rejected (equivalence concluded) when
    ``p_tost < alpha``.  Welch-Satterthwaite degrees of freedom; equal
    variances are NOT assumed.  Diagnostics also carry ``p_exceeds``: the
    one-sided p-value of the difference EXCEEDING the tolerance in the
    observed direction (used for the per-test verdict -- a difference beyond
    the bound must be demonstrated, not just point-estimated).
    """
    from scipy import stats

    na, nb = len(a), len(b)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se2 = var_a / na + var_b / nb
    diff = mean_a - mean_b
    if se2 <= 0:
        # Degenerate zero-variance samples: the difference is exactly known.
        p_lower = 0.0 if diff > -tolerance else 1.0
        p_upper = 0.0 if diff < tolerance else 1.0
        p_exceeds = 0.0 if abs(diff) > tolerance else 1.0
        return max(p_lower, p_upper), {
            "mean_delta": diff,
            "std_error": 0.0,
            "welch_df": float("inf"),
            "p_exceeds": p_exceeds,
        }
    se = float(np.sqrt(se2))
    df = se2**2 / ((var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1))
    # H0_lower: diff <= -tolerance -> reject when (diff + tol)/se is large.
    p_lower = float(stats.t.sf((diff + tolerance) / se, df))
    # H0_upper: diff >= +tolerance -> reject when (diff - tol)/se is small.
    p_upper = float(stats.t.cdf((diff - tolerance) / se, df))
    if diff > 0:
        # H1: diff > +tolerance -> large positive (diff - tol)/se.
        p_exceeds = float(stats.t.sf((diff - tolerance) / se, df))
    else:
        # H1: diff < -tolerance -> large negative (diff + tol)/se.
        p_exceeds = float(stats.t.cdf((diff + tolerance) / se, df))
    return max(p_lower, p_upper), {
        "mean_delta": diff,
        "std_error": se,
        "welch_df": float(df),
        "p_exceeds": p_exceeds,
    }


def deming_slope(x: np.ndarray, y: np.ndarray, sigma_x: float, sigma_y: float) -> float:
    """Deming (errors-in-both-variables) slope with lambda = (sy/sx)^2.

    Both estimators carry noise (the TLE bracketing SMA shift x has a
    24.0 m noise floor, the orbit period-averaged shift y a 1.1 m floor;
    ``SMA_SHIFT_NOISE_FLOOR_M``), so the OLS slope of y on x is attenuated
    by Var(x_true) / (Var(x_true) + sigma_x^2).  The Deming slope uses
        b = [syy - lam*sxx + sqrt((syy - lam*sxx)^2 + 4*lam*sxy^2)] / (2*sxy)
    with lam = (sigma_y/sigma_x)^2, which reduces to syy/sxy when only x
    errs and sxy/sxx when only y errs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    sxx = float(np.var(x, ddof=1))
    syy = float(np.var(y, ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    if sxy == 0 or sxx == 0:
        return float("nan")
    lam = (float(sigma_y) / float(sigma_x)) ** 2
    a_term = syy - lam * sxx
    discriminant = a_term**2 + 4.0 * lam * sxy**2
    root = np.sqrt(discriminant)
    if a_term >= 0.0 or root == 0.0:
        # Direct form; also exact when only x errs (lam -> 0).
        return float((a_term + root) / (2.0 * sxy))
    # Conjugate form for a_term < 0: the direct numerator cancels
    # catastrophically when lambda >> 1 (only y errs), while
    # (root - a_term)(root + a_term) = 4 lam sxy^2 is well conditioned.
    return float(2.0 * lam * sxy / (root - a_term))


def _bootstrap_slope_ci(
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    estimator: str = "ols",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the OLS or Deming slope (seeded, paired resample)."""
    from benchmarking.experiment_params import BOOTSTRAP_SEED

    rng = np.random.default_rng(BOOTSTRAP_SEED if seed is None else seed)
    n = len(x)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.var(xb) == 0:
            slopes[b] = np.nan
        elif estimator == "ols":
            slopes[b] = np.polyfit(xb, yb, 1)[0]
        else:
            slopes[b] = deming_slope(xb, yb, sigma_x, sigma_y)
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) < n_boot / 2:
        return float("nan"), float("nan")
    lo = float(np.quantile(slopes, alpha / 2.0))
    hi = float(np.quantile(slopes, 1.0 - alpha / 2.0))
    return lo, hi


# Magnitude bins of |x| (m) for the banded refits: the pooled slope is
# variance-weighted by the far tail, so per-band slopes show where the
# two estimators' scale relationship actually lives.
SLOPE_MAGNITUDE_BANDS_M = ((0.0, 20.0), (20.0, 100.0), (100.0, 1000.0), (1000.0, float("inf")))


def slope_comparison_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """OLS vs errors-in-variables slopes for the TLE-vs-orbit SMA-shift relation.

    Input: one row per event with ``x_m`` (TLE bracketing SMA shift, m) and
    ``y_m`` (precise-orbit period-averaged SMA shift, m).  Reported side by
    side (REVIEW_FINDINGS 7.1.2):

    - ``ols_slope``: the published-convention regression (attenuated by the
      x noise);
    - ``ols_attenuation_corrected_slope``: OLS undone with the classical
      measurement-error factor (1 + sigma_x^2 / Var(x));
    - ``deming_slope``: the errors-in-variables estimate with
      lambda = (sigma_y/sigma_x)^2 from the pre-registered noise floors;
    - seeded percentile-bootstrap 95% CIs for the pooled OLS and Deming
      slopes;
    - ``fit_scope='pooled'`` plus one row per |x| magnitude band
      (SLOPE_MAGNITUDE_BANDS_M) with per-band slope, Pearson r, and the
      median y/x ratio -- the attenuation at the pre-registered 24 m x-noise
      floor is negligible for the pooled fit (x variance ~5.3e6 m^2 vs
      sigma_x^2 = 576 m^2), so the banded rows make the magnitude dependence
      explicit instead of attributing the pooled slope to attenuation.
    """
    floors = SMA_SHIFT_NOISE_FLOOR_M
    sigma_x = floors["tle_median"]
    sigma_y = floors["orbit_median"]
    x = pd.to_numeric(pairs.get("x_m"), errors="coerce")
    y = pd.to_numeric(pairs.get("y_m"), errors="coerce")
    keep = x.notna() & y.notna()
    x = x[keep].to_numpy(dtype=float)
    y = y[keep].to_numpy(dtype=float)
    if len(x) < 3:
        return pd.DataFrame()

    def _fit_row(scope: str, xb: np.ndarray, yb: np.ndarray) -> dict[str, float | int | str]:
        ols_slope, ols_intercept = np.polyfit(xb, yb, 1)
        var_x = float(np.var(xb, ddof=1))
        corrected = float(ols_slope) * (1.0 + sigma_x**2 / var_x) if var_x > 0 else float("nan")
        deming = deming_slope(xb, yb, sigma_x, sigma_y)
        deming_intercept = float(np.mean(yb) - deming * np.mean(xb)) if np.isfinite(deming) else float("nan")
        pearson = float(np.corrcoef(xb, yb)[0, 1]) if len(xb) >= 3 else float("nan")
        ratio_mask = np.abs(xb) > 0
        median_ratio = float(np.median(yb[ratio_mask] / xb[ratio_mask])) if ratio_mask.any() else float("nan")
        ci_lo, ci_hi = (float("nan"), float("nan"))
        d_lo, d_hi = (float("nan"), float("nan"))
        if len(xb) >= 10:
            ci_lo, ci_hi = _bootstrap_slope_ci(xb, yb, sigma_x, sigma_y, "ols")
            d_lo, d_hi = _bootstrap_slope_ci(xb, yb, sigma_x, sigma_y, "deming")
        return {
            "fit_scope": scope,
            "sample_count": len(xb),
            "sigma_x_m": sigma_x,
            "sigma_y_m": sigma_y,
            "lambda_sigma_y2_over_sigma_x2": (sigma_y / sigma_x) ** 2,
            "x_variance_m2": round(var_x, 3),
            "ols_slope": round(float(ols_slope), 6),
            "ols_intercept_m": round(float(ols_intercept), 4),
            "ols_attenuation_corrected_slope": round(corrected, 6),
            "ols_slope_ci95_low": round(ci_lo, 6),
            "ols_slope_ci95_high": round(ci_hi, 6),
            "deming_slope": round(deming, 6),
            "deming_intercept_m": round(deming_intercept, 4),
            "deming_slope_ci95_low": round(d_lo, 6),
            "deming_slope_ci95_high": round(d_hi, 6),
            "pearson_r": round(pearson, 4),
            "median_ratio_y_over_x": round(median_ratio, 4),
            "note": (
                "x = TLE bracketing SMA shift, y = orbit period-averaged SMA shift; "
                "sigmas are the pre-registered stable-window noise floors "
                "(REVIEW_FINDINGS 7.1.2); OLS y-on-x attenuates by "
                "Var(x_true)/(Var(x_true)+sigma_x^2); CI95 = seeded percentile "
                "bootstrap (2000 resamples); band rows restrict |x| to the "
                "fit_scope range (m); per-band Deming slopes are ill-conditioned "
                "(the band x-spread approaches sigma_x, so lambda no longer "
                "describes the error ratio) and are reported for completeness only"
            ),
        }

    rows = [_fit_row("pooled", x, y)]
    abs_x = np.abs(x)
    for low, high in SLOPE_MAGNITUDE_BANDS_M:
        mask = (abs_x >= low) & (abs_x < high)
        if mask.sum() >= 3:
            hi_label = "inf" if np.isinf(high) else f"{high:.0f}"
            rows.append(_fit_row(f"band_{low:.0f}_{hi_label}m", x[mask], y[mask]))
    return pd.DataFrame(rows)


def tier_equivalence_tost_rows(
    elements: pd.DataFrame,
    min_group: int = 5,
    alpha: float = TIER_EQUIVALENCE_ALPHA,
) -> pd.DataFrame:
    """Tier equivalence via effect sizes + TOST + BH/FDR (replaces KS misuse).

    Input frame: one row per event with ``sat_id``, ``confidence_tier``, and
    the element columns in ``TIER_EQUIVALENCE_ELEMENTS`` (inclination in deg,
    eccentricity, SMA altitude in km).  For every target x element x tier
    pair with at least ``min_group`` events per tier, reports:

    - effect sizes (medians, median/mean delta, Cohen's d) as the primary
      evidence;
    - a TOST equivalence test against the PRE-REGISTERED tolerance from
      ``benchmarking.experiment_params.TIER_EQUIVALENCE_TOLERANCES``
      (engineering-relevance bounds registered before inspecting the
      observed differences -- REVIEW_FINDINGS section 5 phase-0 decision 7b);
    - Benjamini-Hochberg adjusted p-values over the full family of tests
      (3 elements x tier pairs x targets) controlling the FDR.

    A KS test is a difference test and can never support an equivalence
    claim ("statistically similar" was significance-test misuse, B5); this
    table is the equivalence evidence.
    """
    rows: list[dict[str, Any]] = []
    for sat_id, group in elements.groupby("sat_id", sort=True):
        tiers = sorted(str(t) for t in group["confidence_tier"].dropna().unique())
        pairs = [(left, right) for i, left in enumerate(tiers) for right in tiers[i + 1 :]]
        for element in TIER_EQUIVALENCE_ELEMENTS:
            for left, right in pairs:
                a = pd.to_numeric(group.loc[group["confidence_tier"] == left, element], errors="coerce").dropna().to_numpy()
                b = pd.to_numeric(group.loc[group["confidence_tier"] == right, element], errors="coerce").dropna().to_numpy()
                if len(a) < min_group or len(b) < min_group:
                    continue
                tolerance = TIER_EQUIVALENCE_TOLERANCES[element]
                p_tost, diagnostics = _tost_pvalue(a, b, tolerance)
                na, nb = len(a), len(b)
                var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
                pooled_sd = float(np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)))
                mean_delta = diagnostics["mean_delta"]
                median_delta = float(np.median(a)) - float(np.median(b))
                # Per-test verdict (never aggregated -- an "all equivalent"
                # roll-up is exactly the misuse this table replaces):
                #   equivalent        TOST rejects |delta| >= tolerance
                #   exceeds_tolerance the robust effect estimate (median
                #                     delta, this table's primary effect-size
                #                     statistic) lies beyond the
                #                     pre-registered bound -- a descriptive
                #                     breach; its mean-level significance is
                #                     reported separately as tost_p_exceeds
                #   inconclusive      neither statement is supported at this
                #                     sample size
                if p_tost < alpha:
                    verdict = "equivalent"
                elif abs(median_delta) >= tolerance:
                    verdict = "exceeds_tolerance"
                else:
                    verdict = "inconclusive"
                rows.append(
                    {
                        "sat_id": sat_id,
                        "element": element,
                        "tier_pair": f"{left}_vs_{right}",
                        "n_left": na,
                        "n_right": nb,
                        "median_left": round(float(np.median(a)), 6),
                        "median_right": round(float(np.median(b)), 6),
                        "median_abs_delta": round(abs(float(np.median(a)) - float(np.median(b))), 6),
                        "mean_delta": round(mean_delta, 8),
                        "cohens_d": round(mean_delta / pooled_sd, 4) if pooled_sd > 0 else 0.0,
                        "tost_tolerance": tolerance,
                        "tost_pvalue": round(p_tost, 6),
                        "tost_p_exceeds": round(diagnostics["p_exceeds"], 6),
                        "verdict": verdict,
                        "equivalent": bool(p_tost < alpha),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["tost_pvalue_bh"] = benjamini_hochberg(result["tost_pvalue"].to_numpy()).round(6)
    result["equivalent_bh"] = result["tost_pvalue_bh"] < alpha
    return result
