"""Technical-validation experiment runner for MAD-LEO.

Dispatches to the experiment generators in this directory. Run from the
repository root, e.g.
`python scripts/experiments/run_experiments.py starlink-distributions`.

Most generators consume the workspace produced by the data pipeline
(`data/`, populated by scripts/data_pipeline) and write tables to
`results/tables/`. The `starlink-distributions`, `distribution-validation`,
`window-sensitivity` and `attenuation-demo` experiments read the shipped
`dataset/` snapshots (plus the tracked `experiments/validation/` tables)
directly and can run without any prior pipeline execution (the
TLE-vs-ephemeris consistency part additionally needs the raw TLE archive
from the acquisition stage).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Bootstrap: make the flat data_pipeline packages and this directory importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data_pipeline"))

COMMANDS = {
    "event-response": "generate_maneuver_event_response",
    "stable-windows": "generate_stable_windows",
    "state-estimates": "generate_maneuver_state_estimates",
    "state-fusion": "generate_state_fusion",
    "tv-hardening": "generate_tv_hardening",
    "distribution-validation": "generate_distribution_validation",
    "starlink-distributions": "generate_starlink_distributions",
    "external-crossvalidation": "generate_external_crossvalidation",
    "kozai-comparison": "generate_kozai_comparison",
    "core-label-tables": "generate_core_label_tables",
    "provenance-map": "generate_provenance_map",
    "results-figures": "generate_results_figures",
    "slr-crossformat-check": "slr_crossformat_check",
    # SLR geometric O-C residual audit (replaces the NP-precision-shift SLR
    # validation with the standard observed-minus-computed method; reads the
    # shipped evidence snapshots and needs no credentials).
    "slr-oc-audit": "generate_slr_oc_audit",
    # Six supplementary analysis figures for the paper (SLR O-C residuals,
    # sigma calibration, TOST equivalence, Kozai comparison, benchmark
    # tolerance sweep, match offsets); reads the published
    # experiments/validation tables
    # snapshots with a results/tables fallback.
    "new-analysis-figures": "generate_new_analysis_figures",
    # Paper-ready technical-validation figures (tv_dataset 2x3,
    # tv_external_validation 1x3, tv_event_response 2x3, tv_consistency 2x3,
    # plus the standalone tv_anatomy_a/b) drawn from the shipped validation
    # tables, with the anatomy figures embedding the regenerated component PDFs.
    "paper-tv-figures": "make_tv_figures",
    # Analysis-window sensitivity of the TLE SMA-shift response (release
    # -6 h/+24 h vs symmetric +/-24 h window); reads the shipped release
    # tables only and verifies the recomputation against the published
    # response table.
    "window-sensitivity": "generate_window_sensitivity",
    # Synthetic two-estimator attenuation demo (period-averaged orbit vs
    # TLE-scale estimator over an injected-da x bracket-spacing grid on a
    # quiet shipped sentinel-3a orbit arc); shipped evidence only.
    "attenuation-demo": "generate_attenuation_demo",
    # Catalog-indicated orbit-change candidates of the operational slice
    # (consecutive-TLE semi-major-axis steps on the released TLE table;
    # candidate audit artifacts, never labels).
    "starlink-step-candidates": "generate_starlink_step_candidates",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS), help="experiment to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to the experiment")
    args = parser.parse_args()

    module = importlib.import_module(COMMANDS[args.command])
    sys.argv = [args.command, *args.args]
    result = module.main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
