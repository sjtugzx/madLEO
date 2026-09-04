"""Generate quantitative maneuver event-response validation tables.

Reads the all-window source-alignment audit table, computes per-window
TLE / precise-orbit / SLR response magnitudes from local raw products,
and writes:

- ``maneuver_event_response_validation.csv`` (per event window)
- ``maneuver_event_response_summary.csv`` (per target and confidence tier)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyzers.event_response import (
    PER_WINDOW_COLUMNS,
    SUMMARY_COLUMNS,
    compute_target_responses,
    summarize_responses,
)
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path


DEFAULT_ALIGNMENT_TABLE = "results/tables/reference_all_window_alignment_status.csv"
DEFAULT_RAW_ROOT = "data/raw/reference_validation"
DEFAULT_OUTPUT_DIR = "results/tables"
PER_WINDOW_FILENAME = "maneuver_event_response_validation.csv"
SUMMARY_FILENAME = "maneuver_event_response_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quantitative maneuver event-response validation tables")
    parser.add_argument("--alignment-table", default=DEFAULT_ALIGNMENT_TABLE, help="All-window source-alignment audit CSV")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="Root of raw per-target source directories")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for the response CSV outputs")
    parser.add_argument("--include", nargs="+", default=None, help="Restrict to these sat_ids")
    parser.add_argument("--margin-hours", type=float, default=12.0, help="Pre/post window sampling margin in hours")
    return parser.parse_args()


def generate_maneuver_event_response(
    alignment_table: str,
    raw_root: str,
    output_dir: str,
    include: list[str] | None = None,
    margin_hours: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Compute per-window responses and summaries; write both CSV outputs."""
    alignment = pd.read_csv(resolve_repo_path(alignment_table))
    if include:
        alignment = alignment[alignment["sat_id"].isin(include)].copy()

    frames = []
    for sat_id, windows in alignment.groupby("sat_id", sort=True):
        print(f"[{sat_id}] {len(windows)} windows")
        frames.append(compute_target_responses(windows, raw_root=resolve_repo_path(raw_root), sat_id=str(sat_id), margin_hours=margin_hours))
    per_window = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=PER_WINDOW_COLUMNS)
    )
    per_window = per_window.sort_values(["sat_id", "window_start_utc", "annotation_id"]).reset_index(drop=True)
    summary = summarize_responses(per_window)

    destination = ensure_directory(resolve_repo_path(output_dir))
    per_window_path = destination / PER_WINDOW_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    per_window.to_csv(per_window_path, index=False)
    summary.to_csv(summary_path, index=False)
    return per_window, summary, {"per_window": per_window_path, "summary": summary_path}


def main() -> None:
    args = parse_args()
    per_window, summary, paths = generate_maneuver_event_response(
        alignment_table=args.alignment_table,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        include=args.include,
        margin_hours=args.margin_hours,
    )
    status_counts = {
        source: per_window[f"{source}_status"].value_counts().to_dict() if not per_window.empty else {}
        for source in ("tle", "orbit", "slr")
    }
    print(
        json.dumps(
            {
                "per_window_output": str(paths["per_window"]),
                "summary_output": str(paths["summary"]),
                "window_count": int(len(per_window)),
                "summary_rows": int(len(summary)),
                "status_counts": status_counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
