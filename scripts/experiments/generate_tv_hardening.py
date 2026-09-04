"""Generate the TV-hardening tables (Task 14): response floor, uncertainty
strata, target-heldout sigma model, coverage-matched stable windows, the
fusion-value experiment (TLE + second DORIS SP3 series vs held-out primary
DORIS SP3 series), the sigma-model calibration curve (1/2/3 sigma), and the
tier-equivalence TOST/FDR table."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from analyzers.event_response import build_target_context, load_orbit_file
from analyzers.state_estimation import estimate_tle_state, interpolate_orbit_state, load_tle_line_history
from analyzers.tv_hardening import (
    coverage_matched_stable,
    fit_tle_sigma_model,
    fusion_value_rows,
    gap_taxonomy_completeness,
    load_window_tiers,
    matched_specificity_rows,
    response_floor_rows,
    sigma_calibration_rows,
    sigma_coverage_by_eval_group,
    slope_comparison_rows,
    tier_equivalence_tost_rows,
    tier_response_rows,
    tier_sigma_coverage_rows,
    tier_sign_agreement_rows,
    uncertainty_strata_rows,
)
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.experiment_params import (
    BOOTSTRAP_SEED,
    FUSION_MIN_SAMPLE_WINDOWS,
    FUSION_SECOND_SERIES_SIGMA_M,
    PROPAGATION_BIN_EDGES_HOURS,
    TIER_EQUIVALENCE_ELEMENTS,
    sigma_km,
)

TABLES = REPO_ROOT / "results" / "tables"
INTERIM = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"
RAW = REPO_ROOT / "data" / "raw" / "reference_validation"
EVIDENCE_TLE = REPO_ROOT / "dataset" / "mission_reported" / "evidence" / "tle"

# Fusion-value experiment: 1-sigma assigned to the SECOND independent DORIS
# SP3 series (the non-truth series in each overlap) now lives in
# benchmarking.experiment_params.FUSION_SECOND_SERIES_SIGMA_M (P4 single
# source; the derivation notes moved with the constant).  Alias kept so the
# local call sites below read unchanged.
SECOND_SERIES_SIGMA_M = FUSION_SECOND_SERIES_SIGMA_M


def parse_event_times(values) -> pd.Series:
    """UTC-parse event timestamps, tolerating mixed ISO precision.

    The interim annotation tables mix whole-second (``...T17:51:45Z``) and
    millisecond (``...T17:51:45.523Z``) strings; pandas' single-format
    inference rejects the mix, so every event-time parse in this module
    goes through this helper with an explicit ISO8601 format hint.
    """
    return pd.to_datetime(pd.Series(list(values)), utc=True, format="ISO8601")


def build_tier_equivalence_elements(
    annotations: pd.DataFrame,
    evidence_tle_root: Path = EVIDENCE_TLE,
) -> pd.DataFrame:
    """Per-event TLE elements joined to confidence tiers BY KEY.

    Elements come from the released evidence snapshots at the nearest
    catalog epoch (same loader as the distribution-validation experiment).
    Tiers attach via ``annotation_id`` merge -- never positional truncation:
    if the element loader drops an event, its tier row is dropped with it.
    """
    from generate_distribution_validation import _event_elements

    frames = []
    for sat_id, group in annotations.groupby("sat_id", sort=True):
        tle_path = Path(evidence_tle_root) / f"{sat_id}.parquet"
        if not tle_path.exists():
            continue
        elements = _event_elements(pd.read_parquet(tle_path), parse_event_times(group["event_time_utc"]))
        if elements.empty:
            continue
        # _event_elements column names -> the canonical element names of
        # TIER_EQUIVALENCE_ELEMENTS (inclination deg, eccentricity, SMA km).
        elements = elements.rename(
            columns={"inclination": "inclination_deg", "sma_altitude": "sma_altitude_km"}
        )
        elements["sat_id"] = sat_id
        # The element rows are positionally aligned to the group rows by the
        # loader; carrying annotation_id makes every later join key-based.
        elements["annotation_id"] = group["annotation_id"].to_numpy()[: len(elements)]
        frames.append(elements)
    if not frames:
        return pd.DataFrame()
    tiers = annotations[["annotation_id", "confidence_tier"]].drop_duplicates("annotation_id")
    return pd.concat(frames, ignore_index=True).merge(tiers, on="annotation_id", how="inner", validate="one_to_one")


def _fusion_value_for_target(
    sat_id: str,
    windows: pd.DataFrame,
    sigma_model: dict[str, float],
    max_windows: int = 40,
) -> pd.DataFrame:
    """Fusion-value experiment actually run, per target.

    Where two independent DORIS SP3 series (e.g. SSALTO ssa* vs GSFC gsc*)
    cover the same event epoch, the higher-priority series is the held-out
    TRUTH and the other is the SECOND evidence source; the TLE is the third.
    This is NOT a "TLE + operational degraded product" experiment: both
    series are precise SP3 products, and on the available overlaps the
    second series differs from the truth series at the km level (median
    1205 m; see SECOND_SERIES_SIGMA_M).  Reported per window: TLE-only
    error, second-series-only error, and the inverse-variance fused error
    against the held-out truth, plus the never-worse-than-best-source
    check.  With the empirical km-class second-series sigma the fusion is a
    genuine blend rather than fused == second source.
    """
    context = build_target_context(RAW, sat_id)
    orbit_index = context.get("orbit_index") or []
    precise = [e for e in orbit_index if e.get("source_type") == "precise_orbit"]
    by_series: dict[str, list] = {}
    for entry in precise:
        by_series.setdefault(Path(entry["path"]).name[:3].lower(), []).append(entry)
    if len(by_series) < 2:
        return pd.DataFrame()
    series_order = sorted(by_series, key=lambda s: {"ssa": 0, "gsc": 1, "grg": 2, "lca": 3}.get(s, 9))
    truth_entries = by_series[series_order[0]]
    degraded = by_series[series_order[1]]
    tle_lines = load_tle_line_history(RAW / sat_id / "tle")
    rows = []
    sample = windows.sort_values("event_time_utc")
    for _, window in sample.iterrows():
        if len(rows) >= max_windows:
            break
        t0 = parse_event_times([window["event_time_utc"]]).iloc[0]

        def spline_at(entries):
            for entry in entries:
                es, ee = pd.to_datetime(entry["start_utc"], utc=True), pd.to_datetime(entry["end_utc"], utc=True)
                if es - pd.Timedelta(hours=1) > t0 or ee + pd.Timedelta(hours=1) < t0:
                    continue
                try:
                    frame, _ = load_orbit_file(entry["path"])
                except Exception:
                    continue
                state = interpolate_orbit_state(frame, t0)
                if state is not None:
                    return state
            return None

        truth = spline_at(truth_entries)
        degraded_state = spline_at(degraded)
        tle_state = estimate_tle_state(tle_lines, t0)
        if truth is None or degraded_state is None or tle_state.get("tle_state_status") != "computed" or "itrs_x_m" not in tle_state:
            continue

        def err(state, x_key, y_key, z_key):
            return float(
                np.sqrt(
                    (state[x_key] - truth["orbit_x_m"]) ** 2
                    + (state[y_key] - truth["orbit_y_m"]) ** 2
                    + (state[z_key] - truth["orbit_z_m"]) ** 2
                )
                / 1000.0
            )

        tle_error = err(tle_state, "itrs_x_m", "itrs_y_m", "itrs_z_m")
        degraded_error = err(degraded_state, "orbit_x_m", "orbit_y_m", "orbit_z_m")
        sigma_tle = sigma_km(sigma_model, float(tle_state["tle_propagation_seconds"]))
        sigma_degraded = SECOND_SERIES_SIGMA_M / 1000.0
        w_tle, w_deg = 1.0 / sigma_tle**2, 1.0 / sigma_degraded**2
        fused_position = {
            axis: (w_tle * tle_state[f"itrs_{axis}_m"] + w_deg * degraded_state[f"orbit_{axis}_m"]) / (w_tle + w_deg)
            for axis in ("x", "y", "z")
        }
        fused_error = float(
            np.sqrt(sum((fused_position[axis] - truth[f"orbit_{axis}_m"]) ** 2 for axis in ("x", "y", "z"))) / 1000.0
        )
        rows.append(
            {
                "sat_id": sat_id,
                "annotation_id": window["annotation_id"],
                "truth_series": series_order[0],
                "second_series": series_order[1],
                # P4 deviation declaration: the assigned second-series sigma
                # is carried on every output row.
                "second_series_sigma_m": SECOND_SERIES_SIGMA_M,
                "tle_error_km": tle_error,
                "degraded_error_km": degraded_error,
                "fused_error_km": fused_error,
                "fused_not_worse_than_best": fused_error <= min(tle_error, degraded_error) * 1.05,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(TABLES))
    args = parser.parse_args()
    output = ensure_directory(resolve_repo_path(args.output_dir))

    response = pd.read_csv(resolve_repo_path(str(TABLES / "maneuver_event_response_validation.csv")))
    response_floor_rows(response).to_csv(output / "annotation_response_floor.csv", index=False)
    print("wrote annotation_response_floor.csv", flush=True)

    annotation_frames = []
    for path in sorted(glob.glob(str(INTERIM / "*" / "maneuver_annotations.csv"))):
        annotation_frames.append(pd.read_csv(path))
    annotations = pd.concat(annotation_frames, ignore_index=True)
    uncertainty_strata_rows(annotations).to_csv(output / "annotation_uncertainty_strata.csv", index=False)
    print("wrote annotation_uncertainty_strata.csv", flush=True)

    estimates = pd.read_csv(resolve_repo_path(str(TABLES / "maneuver_event_state_estimates.csv")))
    model = fit_tle_sigma_model(estimates)
    coverage = sigma_coverage_by_eval_group(estimates, model)
    coverage.insert(1, "sigma_model", f"base={model['base_km']:.3f}km growth={model['growth_km_per_24h']:.3f}km/24h")
    coverage.to_csv(output / "harmonization_sigma_split_validation.csv", index=False)
    print("wrote harmonization_sigma_split_validation.csv", model, flush=True)

    # Sigma-model calibration curve: nominal vs empirical coverage at
    # 1/2/3 sigma (REVIEW_FINDINGS 7.1.4), respecting the <= 48 h validity
    # domain declaration carried in the table columns.
    calibration = sigma_calibration_rows(estimates, model)
    calibration.insert(1, "sigma_model", f"base={model['base_km']:.3f}km growth={model['growth_km_per_24h']:.3f}km/24h")
    calibration.to_csv(output / "sigma_calibration_curve.csv", index=False)
    print("wrote sigma_calibration_curve.csv", model, flush=True)

    # Errors-in-variables treatment of the TLE-vs-orbit SMA-shift slope
    # (REVIEW_FINDINGS 7.1.2): OLS vs attenuation-corrected OLS vs Deming,
    # with lambda from the pre-registered stable-window noise floors.
    slope_pairs = response.dropna(subset=["delta_sma_km", "orbit_mean_sma_shift_m"])
    slope_table = slope_comparison_rows(
        pd.DataFrame({"x_m": slope_pairs["delta_sma_km"] * 1000.0, "y_m": slope_pairs["orbit_mean_sma_shift_m"]})
    )
    if not slope_table.empty:
        slope_table.to_csv(output / "slope_estimation_comparison.csv", index=False)
        print(f"wrote slope_estimation_comparison.csv n={int(slope_table['sample_count'].iloc[0])}", flush=True)

    stable = pd.read_csv(resolve_repo_path(str(TABLES / "stable_windows.csv")))
    alignment = pd.read_csv(resolve_repo_path(str(TABLES / "reference_all_window_alignment_status.csv")))
    matched = coverage_matched_stable(stable, alignment)
    matched.to_csv(output / "stable_windows_matched.csv", index=False)
    # per-target match report (event A-fraction vs matched A-fraction)
    event_frac = (alignment["aligned"].astype(str) == "True").groupby(alignment["sat_id"]).mean()
    matched_frac = matched.groupby("sat_id")["confidence_tier"].apply(lambda s: (s == "A").mean())
    report = pd.DataFrame(
        {
            "sat_id": event_frac.index,
            "event_tier_a_fraction": event_frac.to_numpy(),
            "matched_tier_a_fraction": [matched_frac.get(s, np.nan) for s in event_frac.index],
            "matched_window_count": [int((matched["sat_id"] == s).sum()) for s in event_frac.index],
        }
    )
    report["matched_tier_a_fraction"] = report["matched_tier_a_fraction"].round(3)
    report.to_csv(output / "stable_windows_match_report.csv", index=False)
    print(f"wrote stable_windows_matched.csv rows={len(matched)} + match report", flush=True)

    fusion_frames = []
    for sat_id in ("jason-1", "jason-2", "cryosat-2", "hy-2a", "saral", "topex-poseidon"):
        windows = pd.read_csv(INTERIM / sat_id / "maneuver_annotations.csv")
        result = _fusion_value_for_target(sat_id, windows, sigma_model=model)
        if not result.empty:
            fusion_frames.append(result)
            print(f"{sat_id}: {len(result)} fusion-value windows", flush=True)
        # C7 explicit insufficiency warning: below the pre-registered
        # minimum the per-target fusion rows are a warning, not evidence.
        if len(result) < FUSION_MIN_SAMPLE_WINDOWS:
            print(
                f"WARNING: {sat_id}: only {len(result)} fusion-value windows "
                f"(< {FUSION_MIN_SAMPLE_WINDOWS}) -- per-target fusion rows are "
                "flagged below_min_sample and are not reportable evidence",
                flush=True,
            )

    # S1-S5: tier-stratified validation.
    tiers = load_window_tiers(INTERIM)
    s1 = tier_response_rows(response, tiers)
    ks = s1.attrs.get("ks_A_vs_C", {})
    s1["ks_A_vs_C_statistic"] = ks.get("statistic", "")
    s1["ks_A_vs_C_pvalue"] = ks.get("pvalue", "")
    ks_ab = s1.attrs.get("ks_A_vs_B", {})
    s1["ks_A_vs_B_statistic"] = ks_ab.get("statistic", "")
    s1["ks_A_vs_B_pvalue"] = ks_ab.get("pvalue", "")
    ks_orbit_ab = s1.attrs.get("orbit_ks_A_vs_B", {})
    s1["orbit_ks_A_vs_B_statistic"] = ks_orbit_ab.get("statistic", "")
    s1["orbit_ks_A_vs_B_pvalue"] = ks_orbit_ab.get("pvalue", "")
    s1.to_csv(output / "tier_response_distribution.csv", index=False)
    tier_sign_agreement_rows(response, tiers).to_csv(output / "tier_sign_agreement.csv", index=False)
    estimates_with_tiers = estimates.merge(
        response[["annotation_id", "confidence_tier"]].drop_duplicates("annotation_id"),
        on="annotation_id", how="left",
    )
    tier_sigma_coverage_rows(estimates_with_tiers, tiers, model).to_csv(output / "tier_sigma_coverage.csv", index=False)
    stable_response_path = resolve_repo_path(str(TABLES / "stable_window_response_validation.csv"))
    stable_response = pd.read_csv(stable_response_path)
    matched_specificity_rows(response, stable_response, matched).to_csv(output / "matched_specificity.csv", index=False)
    import json as _json
    (output / "gap_taxonomy_completeness.json").write_text(_json.dumps(gap_taxonomy_completeness(alignment), indent=2))
    print("wrote S1-S5 tier-stratified tables", flush=True)

    # Tier equivalence via effect sizes + TOST + BH/FDR (REVIEW_FINDINGS
    # 7.1.3).  Per-event TLE elements come from the released evidence
    # snapshots (dataset/mission_reported/evidence/tle) at the nearest
    # catalog epoch, using
    # the same loader as the distribution-validation experiment; tolerances
    # are pre-registered in benchmarking.experiment_params.
    from generate_distribution_validation import _load_annotations

    annotations = _load_annotations(INTERIM)
    if annotations.empty:
        print(
            "skip: tier_equivalence_tost needs the pipeline workspace "
            "(run the data pipeline first)",
            flush=True,
        )
    else:
        elements = build_tier_equivalence_elements(annotations)
        if not elements.empty:
            tost = tier_equivalence_tost_rows(elements)
            tost.to_csv(output / "tier_equivalence_tost.csv", index=False)
            verdicts = tost["verdict"].value_counts().to_dict() if "verdict" in tost else {}
            print(
                f"wrote tier_equivalence_tost.csv rows={len(tost)} "
                f"elements={list(TIER_EQUIVALENCE_ELEMENTS)} verdicts={verdicts}",
                flush=True,
            )


    # m7: bootstrap 95% CI for headline medians.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    def boot_median_ci(values, n_boot=2000):
        values = np.asarray(values)
        meds = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))
    headlines = []
    ev_resp = response["delta_sma_km"].abs().dropna().to_numpy() * 1000.0
    lo, hi = boot_median_ci(ev_resp)
    headlines.append({"statistic": "event |delta_sma| median (m)", "value": float(np.median(ev_resp)), "ci95_lo": lo, "ci95_hi": hi, "n": len(ev_resp)})
    stable_resp = stable_response["delta_sma_km"].abs().dropna().to_numpy() * 1000.0
    lo, hi = boot_median_ci(stable_resp)
    headlines.append({"statistic": "stable |delta_sma| median (m)", "value": float(np.median(stable_resp)), "ci95_lo": lo, "ci95_hi": hi, "n": len(stable_resp)})
    pd.DataFrame(headlines).to_csv(output / "headline_statistics.csv", index=False)
    print("wrote headline_statistics.csv", flush=True)

    # m1: sign agreement stratified by response magnitude.
    both = response.dropna(subset=["delta_sma_km", "orbit_mean_sma_shift_m"]).copy()
    if not both.empty:
        both["abs_resp_m"] = both["delta_sma_km"].abs() * 1000.0
        both["sign_agree"] = np.sign(both["delta_sma_km"]) == np.sign(both["orbit_mean_sma_shift_m"])
        sign_tab = both.groupby(
            pd.cut(both["abs_resp_m"], [0, 5, 20, 100, 1e9], labels=["<5m", "5-20m", "20-100m", ">100m"]),
            observed=True,
        )["sign_agree"].agg(["count", "mean"]).reset_index().rename(columns={"abs_resp_m": "response_band"})
        sign_tab.to_csv(output / "sign_agreement_by_magnitude.csv", index=False)
        print("wrote sign_agreement_by_magnitude.csv", flush=True)

    # m2: E2 propagation curve without edge-extrapolated orbit estimates.
    estimates_full = estimates[(estimates["tle_state_status"] == "computed") & (estimates["orbit_state_status"] == "computed")].copy()
    if not estimates_full.empty:
        residual_km = np.sqrt(
            (estimates_full["itrs_x_m"] - estimates_full["orbit_x_m"]) ** 2
            + (estimates_full["itrs_y_m"] - estimates_full["orbit_y_m"]) ** 2
            + (estimates_full["itrs_z_m"] - estimates_full["orbit_z_m"]) ** 2
        ) / 1000.0
        estimates_full["residual_km"] = residual_km
        estimates_full["propagation_hours"] = estimates_full["tle_propagation_seconds"] / 3600.0
        for subset_name, subset in (("all", estimates_full), ("no_extrapolation", estimates_full[estimates_full["orbit_extrapolation_seconds"].isna()])):
            binned = subset.groupby(pd.cut(subset["propagation_hours"], list(PROPAGATION_BIN_EDGES_HOURS)), observed=True)["residual_km"].agg(["count", "median", lambda s: s.quantile(0.75)])
            binned.columns = ["count", "median", "p75"]
            binned = binned.reset_index().rename(columns={"propagation_hours": "propagation_bin_hours"})
            binned["subset"] = subset_name
            binned.to_csv(output / f"sgp4_propagation_residuals_{'clean' if subset_name == 'no_extrapolation' else 'all'}.csv", index=False)
        print("wrote sgp4 propagation residual tables", flush=True)

    if fusion_frames:
        fusion = pd.concat(fusion_frames, ignore_index=True)
        fusion.to_csv(output / "fusion_value_per_window.csv", index=False)
        summary = fusion_value_rows(fusion, min_windows=FUSION_MIN_SAMPLE_WINDOWS)
        summary.to_csv(output / "fusion_value_summary.csv", index=False)
        flagged = summary[summary["below_min_sample"]]["sat_id"].tolist() if "below_min_sample" in summary else []
        print(f"wrote fusion_value_summary.csv (below_min_sample targets: {flagged or 'none'})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
