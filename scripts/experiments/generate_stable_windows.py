"""Generate stable (no-event) windows for reference validation targets.

Stable windows are the negative counterpart to the mission-reported maneuver
event windows: same 30h duration, placed on a deterministic non-overlapping
grid over each target's TLE-derived mission interval, excluding any window
overlapping an event window plus a 24h safety buffer, with per-calendar-year
quotas matched to the local event density. See
``benchmarking.stable_windows`` for the full design rationale.

This module is also the (single, registered) writer of the per-window
source-response table for the stable windows (REVIEW_FINDINGS C3: the
shipped ``stable_window_response_validation.csv`` previously had no writer
anywhere, so a fresh reproduction crashed at tv-hardening).  Responses
reuse ``analyzers.event_response.compute_target_responses`` -- the same
per-window file selection used for event windows (per the T11 runbook
ruling, responses must be computed per-window file path, never from
whole-arc frames).

Outputs:
- ``<output-dir>/<sat_id>/stable_windows.csv`` per target
- ``results/tables/stable_windows.csv`` (all targets combined)
- ``results/tables/stable_windows_summary.csv`` (per-target decision table)
- ``results/tables/stable_window_response_validation.csv``
  (per-window TLE/orbit/SLR responses; shipped schema)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from alignment.audit_reference_window_alignment import (
    infer_orbit_product_spans,
    infer_slr_file_spans,
    infer_tle_epochs,
)
from analyzers.event_response import PER_WINDOW_COLUMNS, compute_target_responses
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.stable_windows import (
    SUMMARY_COLUMNS,
    STABLE_WINDOW_COLUMNS,
    mine_stable_windows,
    target_pod_dir,
)
from downloaders.download_batch_tle_history import DEFAULT_CONFIG, load_annotation_config

DEFAULT_EVENT_WINDOW_ROOT = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"
DEFAULT_ALIGNMENT_TABLE = REPO_ROOT / "results" / "tables" / "reference_all_window_alignment_status.csv"
DEFAULT_COMBINED_OUTPUT = REPO_ROOT / "results" / "tables" / "stable_windows.csv"
DEFAULT_SUMMARY_OUTPUT = REPO_ROOT / "results" / "tables" / "stable_windows_summary.csv"
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "reference_validation"
DEFAULT_RESPONSE_OUTPUT = REPO_ROOT / "results" / "tables" / "stable_window_response_validation.csv"

STABLE_RESPONSE_FILENAME = "stable_window_response_validation.csv"
# Shipped schema (compare experiments/validation/stable_window_response_validation.csv,
# 36 columns): the event per-window schema WITHOUT the orbit_mean_sma trio and
# the L4 parse-failure columns -- the shipped stable table has never carried
# them.  Schema changes here are pre-registered release-scope decisions, not
# incidental drift.
STABLE_RESPONSE_EXCLUDED_COLUMNS = (
    "orbit_mean_sma_before_m",
    "orbit_mean_sma_after_m",
    "orbit_mean_sma_shift_m",
    "orbit_parse_failure_count",
    "orbit_parse_failure_detail",
)
STABLE_RESPONSE_COLUMNS = [c for c in PER_WINDOW_COLUMNS if c not in STABLE_RESPONSE_EXCLUDED_COLUMNS]


def _resolve(path: str | Path) -> Path:
    resolved = resolve_repo_path(str(path))
    if resolved is None:
        raise ValueError(f"cannot resolve path: {path}")
    return resolved


def load_event_windows_for_target(
    sat_id: str,
    event_window_root: Path,
    alignment_table: pd.DataFrame | None,
) -> pd.DataFrame:
    """Load a target's mission-reported event windows.

    Primary source is the per-target normalized ``event_windows.csv``; when
    it is absent, fall back to the authoritative all-window alignment audit
    table (which carries the same window bounds per annotation).
    """
    path = event_window_root / sat_id / "event_windows.csv"
    if path.exists():
        frame = pd.read_csv(path)
        return frame[frame["sat_id"].astype(str) == sat_id].reset_index(drop=True)
    if alignment_table is not None and not alignment_table.empty:
        frame = alignment_table[alignment_table["sat_id"].astype(str) == sat_id]
        return frame[["sat_id", "window_start_utc", "window_end_utc"]].reset_index(drop=True)
    return pd.DataFrame(columns=["sat_id", "window_start_utc", "window_end_utc"])


def compute_stable_window_responses(stable_windows: pd.DataFrame, raw_root: str | Path) -> pd.DataFrame:
    """Compute per-window TLE/orbit/SLR responses for the mined stable windows.

    Same path as the event-response experiment: each target's windows go
    through ``analyzers.event_response.compute_target_responses``, which
    selects evidence files PER WINDOW (TLE bracket, orbit files overlapping
    the window band, SLR passes near the window) -- never a whole-arc frame
    (T11 runbook ruling: whole-arc median-diff shifts orbit_mean_sma by
    meters).  Output carries the shipped stable-response schema.
    """
    if stable_windows is None or stable_windows.empty:
        return pd.DataFrame(columns=STABLE_RESPONSE_COLUMNS)
    frames = []
    for sat_id, windows in stable_windows.groupby("sat_id", sort=True):
        print(f"[{sat_id}] {len(windows)} stable windows: computing responses", flush=True)
        frames.append(
            compute_target_responses(windows, raw_root=Path(raw_root), sat_id=str(sat_id))
        )
    responses = pd.concat(frames, ignore_index=True)
    responses = responses.sort_values(["sat_id", "window_start_utc", "annotation_id"]).reset_index(drop=True)
    return responses[STABLE_RESPONSE_COLUMNS]


def generate_stable_windows(
    config: dict,
    event_window_root: str | Path = DEFAULT_EVENT_WINDOW_ROOT,
    alignment_table_path: str | Path = DEFAULT_ALIGNMENT_TABLE,
    output_dir: str | Path = DEFAULT_EVENT_WINDOW_ROOT,
    combined_output: str | Path = DEFAULT_COMBINED_OUTPUT,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    response_output: str | Path = DEFAULT_RESPONSE_OUTPUT,
    compute_responses: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Mine stable windows for every configured target with local data."""
    event_root = _resolve(event_window_root)
    alignment_path = _resolve(alignment_table_path)
    alignment_table = pd.read_csv(alignment_path) if alignment_path.exists() else None
    out_root = _resolve(output_dir)

    combined_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    written: dict[str, Path] = {}

    for target in config.get("targets", []):
        sat_id = target["sat_id"]
        event_windows = load_event_windows_for_target(sat_id, event_root, alignment_table)
        if event_windows.empty:
            continue  # not one of the 11 reference targets with local windows
        tle_epochs = infer_tle_epochs(target.get("tle_raw_path"))
        slr_spans = infer_slr_file_spans(target.get("slr_raw_path"))
        orbit_spans = infer_orbit_product_spans([target_pod_dir(target), target.get("precise_orbit_raw_path")])
        windows, summary = mine_stable_windows(
            sat_id=sat_id,
            batch=target.get("batch", ""),
            event_windows=event_windows,
            tle_epochs=tle_epochs,
            slr_spans=slr_spans,
            orbit_spans=orbit_spans,
        )
        summary_rows.append(summary)
        if windows.empty:
            continue
        target_dir = ensure_directory(out_root / sat_id)
        output_path = target_dir / "stable_windows.csv"
        windows.to_csv(output_path, index=False)
        written[sat_id] = output_path
        combined_frames.append(windows)

    combined = (
        pd.concat(combined_frames, ignore_index=True)
        if combined_frames
        else pd.DataFrame(columns=STABLE_WINDOW_COLUMNS)
    )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    combined_path = _resolve(combined_output)
    ensure_directory(combined_path.parent)
    combined.to_csv(combined_path, index=False)
    summary_path = _resolve(summary_output)
    ensure_directory(summary_path.parent)
    summary.to_csv(summary_path, index=False)
    written["_combined"] = combined_path
    written["_summary"] = summary_path

    # C3 writer: per-window responses for the mined stable windows (skipped
    # loudly when the raw archive backing the computation is absent).
    resolved_raw_root = Path(raw_root) if Path(str(raw_root)).is_absolute() else _resolve(raw_root)
    if compute_responses and not combined.empty:
        if resolved_raw_root is not None and resolved_raw_root.is_dir():
            responses = compute_stable_window_responses(combined, resolved_raw_root)
            response_path = _resolve(response_output)
            ensure_directory(response_path.parent)
            responses.to_csv(response_path, index=False)
            written["_response"] = response_path
        else:
            print(
                f"skip: stable-window responses need the raw archive at {resolved_raw_root} "
                f"(run the data pipeline acquisition stage first); downstream consumers "
                f"of {STABLE_RESPONSE_FILENAME} will not find a fresh table",
                flush=True,
            )
    return combined, summary, written


def refresh_summary_from_windows(
    windows_path: str | Path,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
) -> pd.DataFrame:
    """Recompute the coverage/tier count columns of the per-target summary
    from a final, gate-semantics stable-window table (typically the shipped
    ``dataset/mission_reported/annotations/stable_windows.csv`` after the
    release selfcheck
    recompute), preserving the mission/event metadata columns of the existing
    summary.

    Mining-time coverage reflects raw-archive availability (used to PLACE the
    windows); the released table's tiers are assigned on the shipped evidence
    by the release gate. This refresh keeps the released summary consistent
    with the released windows without re-mining.
    """
    windows_path = _resolve(windows_path)
    summary_path = _resolve(summary_output)
    windows = pd.read_csv(windows_path)
    summary = pd.read_csv(summary_path)
    for col in ("tle_status", "slr_status", "orbit_status", "confidence_tier"):
        if col not in windows.columns:
            raise ValueError(f"{windows_path} lacks required column {col!r}")

    grouped = windows.groupby("sat_id", sort=False)
    counts = pd.DataFrame({
        "stable_window_count": grouped.size(),
        "aligned_window_count": grouped.apply(lambda d: int((d["confidence_tier"] == "A").sum()), include_groups=False),
        "tle_covered_count": grouped.apply(lambda d: int((d["tle_status"] == "covered").sum()), include_groups=False),
        "slr_covered_count": grouped.apply(lambda d: int((d["slr_status"] == "covered").sum()), include_groups=False),
        "orbit_covered_count": grouped.apply(lambda d: int((d["orbit_status"] == "covered").sum()), include_groups=False),
        "confidence_A_count": grouped.apply(lambda d: int((d["confidence_tier"] == "A").sum()), include_groups=False),
        "confidence_B_count": grouped.apply(lambda d: int((d["confidence_tier"] == "B").sum()), include_groups=False),
        "confidence_C_count": grouped.apply(lambda d: int((d["confidence_tier"] == "C").sum()), include_groups=False),
    })
    count_cols = list(counts.columns)
    merged = summary.drop(columns=[c for c in count_cols if c in summary.columns]).merge(
        counts.reset_index(), on="sat_id", how="left", validate="one_to_one"
    )
    if merged[count_cols].isna().any().any():
        missing = merged[merged["stable_window_count"].isna()]["sat_id"].tolist()
        raise ValueError(f"no windows found for summary targets: {missing}")
    merged = merged[SUMMARY_COLUMNS]
    merged.to_csv(summary_path, index=False)
    print(f"refreshed {summary_path} from {windows_path} ({len(merged)} targets)", flush=True)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stable (no-event) windows for reference targets")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--refresh-summary-from",
        default=None,
        metavar="WINDOWS_CSV",
        help="Only recompute the summary table's coverage/tier counts from the "
             "given final (gate-semantics) stable-window table, then exit; "
             "no mining, no response computation",
    )
    parser.add_argument(
        "--event-window-root",
        default=str(DEFAULT_EVENT_WINDOW_ROOT),
        help="Root holding per-target event_windows.csv inputs",
    )
    parser.add_argument(
        "--alignment-table",
        default=str(DEFAULT_ALIGNMENT_TABLE),
        help="All-window alignment audit table (fallback event-window source)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EVENT_WINDOW_ROOT),
        help="Root for per-target stable_windows.csv outputs",
    )
    parser.add_argument("--combined-output", default=str(DEFAULT_COMBINED_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument(
        "--raw-root",
        default=str(DEFAULT_RAW_ROOT),
        help="Root of raw per-target source directories for response computation",
    )
    parser.add_argument(
        "--response-output",
        default=str(DEFAULT_RESPONSE_OUTPUT),
        help="Per-window stable-window response table (shipped schema)",
    )
    parser.add_argument(
        "--skip-responses",
        action="store_true",
        help="Skip computing stable_window_response_validation.csv (mine windows only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_summary_from:
        refresh_summary_from_windows(args.refresh_summary_from, args.summary_output)
        return
    config = load_annotation_config(args.config)
    combined, summary, written = generate_stable_windows(
        config,
        event_window_root=args.event_window_root,
        alignment_table_path=args.alignment_table,
        output_dir=args.output_dir,
        combined_output=args.combined_output,
        summary_output=args.summary_output,
        raw_root=args.raw_root,
        response_output=args.response_output,
        compute_responses=not args.skip_responses,
    )
    print(
        json.dumps(
            {
                "target_count": int(summary["sat_id"].nunique()) if not summary.empty else 0,
                "stable_window_count": int(len(combined)),
                "per_target_outputs": {key: str(value) for key, value in sorted(written.items())},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
