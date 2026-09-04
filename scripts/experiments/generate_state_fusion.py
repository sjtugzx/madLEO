"""Generate harmonization fusion tables (E4 bias calibration, E5/E6 validation)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyzers.state_fusion import (
    bias_calibration_rows,
    confidence_discrimination_rows,
    fuse_states,
    leave_one_out_rows,
)
from analyzers.tv_hardening import fit_tle_sigma_model
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path

DEFAULT_STATE_TABLE = REPO_ROOT / "results" / "tables" / "maneuver_event_state_estimates.csv"
DEFAULT_RESPONSE_TABLE = REPO_ROOT / "results" / "tables" / "maneuver_event_response_validation.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "tables"
INTERIM = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-table", default=str(DEFAULT_STATE_TABLE))
    parser.add_argument("--response-table", default=str(DEFAULT_RESPONSE_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = ensure_directory(resolve_repo_path(args.output_dir))
    estimates = pd.read_csv(resolve_repo_path(args.state_table))

    # Single-source sigma model: fitted from the same estimates table (P4 --
    # no experiment-local sigma constants anywhere).
    sigma_model = fit_tle_sigma_model(estimates)
    print(f"sigma model: base={sigma_model['base_km']:.3f}km growth={sigma_model['growth_km_per_24h']:.3f}km/24h", flush=True)

    # Per-event fused states.
    fused_rows = []
    for _, row in estimates.iterrows():
        fused = fuse_states(row, sigma_model=sigma_model)
        fused_rows.append({"annotation_id": row["annotation_id"], "sat_id": row["sat_id"], "t0_utc": row["t0_utc"], **fused})
    fused = pd.DataFrame(fused_rows)
    fused.to_csv(output_dir / "maneuver_event_fused_states.csv", index=False)

    # E4: TLE-vs-POD bias calibration per target per year.
    bias = bias_calibration_rows(estimates)
    bias.to_csv(output_dir / "source_bias_calibration.csv", index=False)

    # E5: leave-POD-out consistency of the TLE-only path.
    loo = leave_one_out_rows(estimates, sigma_model=sigma_model)
    loo.to_csv(output_dir / "harmonization_validation_summary.csv", index=False)

    # E6: response-magnitude discrimination by confidence tier (per-window
    # tiers from the annotated event windows, TLE response deltas from the
    # response table).
    from analyzers.tv_hardening import load_window_tiers

    tiers = load_window_tiers(INTERIM)
    if {"annotation_id", "confidence_tier"}.issubset(tiers.columns) and not tiers.empty:
        merged = estimates.merge(tiers[["annotation_id", "confidence_tier"]], on="annotation_id", how="left")
    else:
        # C10: with no pipeline workspace (interim missing) load_window_tiers
        # returns a column-less empty frame; the positional KeyError used to
        # abort the whole state-fusion run here. Skip E6 explicitly instead --
        # the fused/bias/leave-one-out tables above stay valid without it.
        print(
            f"skip: harmonization_confidence_discrimination needs per-window tiers "
            f"from {INTERIM} (run the data pipeline first)",
            flush=True,
        )
        merged = estimates
    response_path = resolve_repo_path(args.response_table)
    if response_path.exists():
        response = pd.read_csv(response_path)
        delta_col = "delta_sma_km" if "delta_sma_km" in response.columns else None
        if delta_col:
            merged = merged.merge(response[["annotation_id", delta_col]], on="annotation_id", how="left")
    if "confidence_tier" in merged.columns:
        discrimination = confidence_discrimination_rows(merged)
        discrimination.to_csv(output_dir / "harmonization_confidence_discrimination.csv", index=False)
    else:
        discrimination = pd.DataFrame()

    print(
        f"fused={len(fused)} bias_rows={len(bias)} loo_rows={len(loo)} discrimination_rows={len(discrimination)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
