"""Uncertainty-weighted fusion of event-epoch orbit state estimates (A2/A3).

Combines the per-source t0 estimates from
`analyzers.state_estimation` into a fused orbit-state estimate per event:

- Bias calibration (A2/E4): systematic TLE-vs-POD offset per target per year,
  decomposed into radial/along-track/cross-track components using the POD
  state as the reference frame anchor.
- Fusion (A3): inverse-variance weighting of TLE (propagation-decayed) and
  POD states; POD (dm-class) dominates unless missing.
- Validation experiments: E5 leave-one-out TLE-vs-POD consistency check, E6
  fused-response discrimination between A-tier events and stable windows.

SLR geometry-based absolute validation (station coordinates, range
corrections) is out of scope here; SLR remains sparse audit evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from benchmarking import experiment_params

# Nominal 1-sigma position uncertainty of the precise-orbit (POD) source
# (meters).  Precise DORIS/POE products are dm-class; this is the fusion
# design parameter for the POD side only.  The TLE side no longer carries a
# local constant: it flows through the single-source sigma model in
# ``benchmarking.experiment_params`` (REVIEW_FINDINGS P4 / A7 -- the legacy
# hardcoded 350 m base + 400 m per 24 h values are superseded by the fitted
# model, DEFAULT_SIGMA_MODEL by default or an explicitly fitted model
# threaded in by the caller).
POD_SIGMA_M = 0.1


def _rac_basis(orbit_row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Radial/along-track/cross-track unit vectors from the POD state.

    The shipped orbit velocities are Earth-fixed (rotating-frame) values
    (finite differences of ITRF positions; see dataset/docs/metadata.md), so
    the omega x r term is restored through the single shared correction
    before the angular-momentum direction defines the along/cross axes.
    Without it the basis tilts by ~omega/n (~4 deg for these orbits) and
    mixes along-track into cross-track at the 5-7% level.
    """
    from processors.physical_qc import ecef_velocity_to_inertial

    try:
        r = np.array([orbit_row["orbit_x_m"], orbit_row["orbit_y_m"], orbit_row["orbit_z_m"]], dtype=float)
        v = np.array([orbit_row["orbit_vx_mps"], orbit_row["orbit_vy_mps"], orbit_row["orbit_vz_mps"]], dtype=float)
    except KeyError:
        return None
    v_inertial = ecef_velocity_to_inertial(v.reshape(1, 3), r.reshape(1, 3))[0]
    r_norm = np.linalg.norm(r)
    if r_norm == 0 or np.linalg.norm(v_inertial) == 0:
        return None
    radial = r / r_norm
    cross = np.cross(r, v_inertial)
    cross = cross / np.linalg.norm(cross)
    along = np.cross(cross, radial)
    return radial, along, cross


def tle_sigma_m(propagation_seconds: float, sigma_model: dict[str, float] | None = None) -> float:
    """TLE 1-sigma position uncertainty decayed by propagation distance (m).

    Thin wrapper over the single-source sigma model: ``sigma_model`` should
    be the model fitted by ``benchmarking.experiment_params.fit_sigma_model``
    on the current estimates (callers that have the estimates must fit and
    pass it); ``DEFAULT_SIGMA_MODEL`` is the frozen released fit used only
    when no fit is available.
    """
    model = sigma_model if sigma_model is not None else experiment_params.DEFAULT_SIGMA_MODEL
    return experiment_params.sigma_km(model, propagation_seconds) * 1000.0


def tle_minus_pod_components(row: pd.Series) -> dict[str, float] | None:
    """TLE(ITRS) minus POD position difference in the POD R/A/C frame."""
    basis = _rac_basis(row)
    if basis is None:
        return None
    try:
        diff = np.array(
            [row["itrs_x_m"] - row["orbit_x_m"], row["itrs_y_m"] - row["orbit_y_m"], row["itrs_z_m"] - row["orbit_z_m"]],
            dtype=float,
        )
    except KeyError:
        return None
    radial, along, cross = basis
    return {
        "radial_m": float(diff @ radial),
        "along_track_m": float(diff @ along),
        "cross_track_m": float(diff @ cross),
        "norm_km": float(np.linalg.norm(diff) / 1000.0),
    }


def bias_calibration_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    """E4: per-target per-year systematic TLE-vs-POD bias (R/A/C)."""
    rows = []
    usable = estimates[(estimates["tle_state_status"] == "computed") & (estimates["orbit_state_status"] == "computed")]
    if usable.empty:
        return pd.DataFrame()
    usable = usable.copy()
    # format="ISO8601": t0_utc strings mix whole-second and millisecond
    # precision; pandas' single-format inference rejects the mix.
    usable["year"] = pd.to_datetime(usable["t0_utc"], utc=True, format="ISO8601").dt.year
    for (sat_id, year), group in usable.groupby(["sat_id", "year"]):
        components = [tle_minus_pod_components(row) for _, row in group.iterrows()]
        components = [c for c in components if c is not None]
        if not components:
            continue
        frame = pd.DataFrame(components)
        rows.append(
            {
                "sat_id": sat_id,
                "year": int(year),
                "sample_count": len(frame),
                "radial_bias_m": float(frame["radial_m"].median()),
                "along_track_bias_m": float(frame["along_track_m"].median()),
                "cross_track_bias_m": float(frame["cross_track_m"].median()),
                "norm_median_km": float(frame["norm_km"].median()),
                "norm_p75_km": float(frame["norm_km"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def fuse_states(row: pd.Series, sigma_model: dict[str, float] | None = None) -> dict[str, Any]:
    """Inverse-variance fuse TLE and POD t0 position estimates (meters).

    ``sigma_model``: the fitted TLE sigma model (see :func:`tle_sigma_m`).
    """
    tle_ok = row.get("tle_state_status") == "computed" and "itrs_x_m" in row.index and pd.notna(row.get("itrs_x_m"))
    pod_ok = row.get("orbit_state_status") == "computed" and pd.notna(row.get("orbit_x_m"))
    if not tle_ok and not pod_ok:
        return {"fusion_status": "no_sources"}
    if tle_ok and not pod_ok:
        return {
            "fusion_status": "tle_only",
            "fused_x_m": row["itrs_x_m"],
            "fused_y_m": row["itrs_y_m"],
            "fused_z_m": row["itrs_z_m"],
            "fused_sigma_m": tle_sigma_m(float(row["tle_propagation_seconds"]), sigma_model=sigma_model),
        }
    if pod_ok and not tle_ok:
        return {
            "fusion_status": "pod_only",
            "fused_x_m": row["orbit_x_m"],
            "fused_y_m": row["orbit_y_m"],
            "fused_z_m": row["orbit_z_m"],
            "fused_sigma_m": POD_SIGMA_M,
        }
    sigma_tle = tle_sigma_m(float(row["tle_propagation_seconds"]), sigma_model=sigma_model)
    sigma_pod = POD_SIGMA_M
    w_tle = 1.0 / sigma_tle**2
    w_pod = 1.0 / sigma_pod**2
    fused = {
        "fusion_status": "fused",
        "fused_sigma_m": float(np.sqrt(1.0 / (w_tle + w_pod))),
    }
    for axis in ("x", "y", "z"):
        fused[f"fused_{axis}_m"] = float((w_tle * row[f"itrs_{axis}_m"] + w_pod * row[f"orbit_{axis}_m"]) / (w_tle + w_pod))
    return fused


def leave_one_out_rows(estimates: pd.DataFrame, sigma_model: dict[str, float] | None = None) -> pd.DataFrame:
    """E5: leave-POD-out — TLE-only estimate vs held-out POD, per target.

    The fused estimate is dominated by POD by construction; the honest check
    is whether the TLE-only path (the operationally available source) stays
    within its predicted sigma of the held-out POD state.  ``sigma_model``
    should be fitted from ``estimates`` by the caller (see
    :func:`tle_sigma_m`).
    """
    rows = []
    usable = estimates[(estimates["tle_state_status"] == "computed") & (estimates["orbit_state_status"] == "computed")]
    for sat_id, group in usable.groupby("sat_id"):
        residuals = []
        coverage = []
        coverage_clean = []
        extrapolated = 0
        for _, row in group.iterrows():
            residual = propagation_distance_km(row)
            if residual is None:
                continue
            bound_km = tle_sigma_m(float(row["tle_propagation_seconds"]), sigma_model=sigma_model) / 1000.0
            is_extrapolated = pd.notna(row.get("orbit_extrapolation_seconds")) and float(row.get("orbit_extrapolation_seconds") or 0) > 0
            extrapolated += int(is_extrapolated)
            residuals.append(residual)
            coverage.append(residual <= 3 * bound_km)
            if not is_extrapolated:
                coverage_clean.append(residual <= 3 * bound_km)
        if residuals:
            rows.append(
                {
                    "sat_id": sat_id,
                    "sample_count": len(residuals),
                    "tle_only_residual_median_km": float(np.median(residuals)),
                    "tle_only_residual_p75_km": float(np.quantile(residuals, 0.75)),
                    "within_3sigma_fraction": float(np.mean(coverage)),
                    "extrapolated_window_count": extrapolated,
                    "within_3sigma_fraction_no_extrapolation": float(np.mean(coverage_clean)) if coverage_clean else "",
                }
            )
    return pd.DataFrame(rows)


def propagation_distance_km(row: pd.Series) -> float | None:
    """TLE(ITRS)-vs-POD position residual in km for one estimate row."""
    try:
        diff = np.array(
            [row["itrs_x_m"] - row["orbit_x_m"], row["itrs_y_m"] - row["orbit_y_m"], row["itrs_z_m"] - row["orbit_z_m"]],
            dtype=float,
        )
    except (KeyError, TypeError):
        return None
    return float(np.linalg.norm(diff) / 1000.0)


def confidence_discrimination_rows(estimates: pd.DataFrame, events_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """E6: fused TLE response magnitude, events vs tiers (discrimination)."""
    usable = estimates[estimates["tle_state_status"] == "computed"].copy()
    if usable.empty:
        return pd.DataFrame()
    rows = []
    for tier, group in usable.groupby("confidence_tier") if "confidence_tier" in usable.columns else []:
        responses = group.get("delta_sma_km")
        if responses is None:
            continue
        rows.append(
            {
                "confidence_tier": tier,
                "sample_count": len(group),
                "median_abs_delta_sma_km": float(responses.abs().median()),
            }
        )
    return pd.DataFrame(rows)
