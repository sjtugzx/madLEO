"""Generate full reference annotation alignment tables from local evidence audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from alignment.audit_reference_window_alignment import audit_reference_window_alignment, write_alignment_audit
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from downloaders.download_batch_tle_history import DEFAULT_CONFIG, load_annotation_config


ANNOTATION_LABEL_SOURCE = "ids_doris_mission_reported_maneuver"
ANNOTATED_EVENT_WINDOW_COLUMNS = [
    "annotation_id",
    "sat_id",
    "batch",
    "window_start_utc",
    "window_end_utc",
    "scope",
    "event_label",
    "confidence_tier",
    "annotation_label_source",
    "tle_status",
    "slr_status",
    "orbit_status",
    "aligned",
    "missing_sources",
    "quality_flags",
]
SUMMARY_COLUMNS = [
    "sat_id",
    "batch",
    "target_count",
    "event_window_count",
    "aligned_window_count",
    "tle_covered_count",
    "slr_covered_count",
    "orbit_covered_count",
    "confidence_A_count",
    "confidence_B_count",
    "confidence_C_count",
    "first_window_start_utc",
    "last_window_end_utc",
]


def _missing_sources(value: object) -> list[str]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _coverage_confidence_tier(row: pd.Series) -> str:
    has_tle = row.get("tle_status") == "covered"
    has_slr = row.get("slr_status") == "covered"
    has_orbit = row.get("orbit_status") == "covered"
    if has_tle and has_slr and has_orbit:
        return "A"
    if has_tle and has_orbit:
        return "B"
    return "C"


def build_annotated_event_windows(alignment_audit: pd.DataFrame) -> pd.DataFrame:
    """Attach mission-event labels and coverage-confidence tiers to audit rows."""
    if alignment_audit.empty:
        return pd.DataFrame(columns=ANNOTATED_EVENT_WINDOW_COLUMNS)
    annotated = alignment_audit.copy()
    annotated["event_label"] = "event"
    annotated["annotation_label_source"] = ANNOTATION_LABEL_SOURCE
    annotated["confidence_tier"] = annotated.apply(_coverage_confidence_tier, axis=1)
    annotated["quality_flags"] = annotated["missing_sources"].apply(
        lambda value: ",".join(f"missing_{source}" for source in _missing_sources(value))
    )
    for column in ANNOTATED_EVENT_WINDOW_COLUMNS:
        if column not in annotated.columns:
            annotated[column] = ""
    return (
        annotated[ANNOTATED_EVENT_WINDOW_COLUMNS]
        .sort_values(["sat_id", "window_start_utc", "annotation_id"])
        .reset_index(drop=True)
    )


def _summary_row(group: pd.DataFrame, sat_id: str, batch: str, target_count: int) -> dict:
    return {
        "sat_id": sat_id,
        "batch": batch,
        "target_count": target_count,
        "event_window_count": int(len(group)),
        "aligned_window_count": int(group["aligned"].astype(bool).sum()),
        "tle_covered_count": int((group["tle_status"] == "covered").sum()),
        "slr_covered_count": int((group["slr_status"] == "covered").sum()),
        "orbit_covered_count": int((group["orbit_status"] == "covered").sum()),
        "confidence_A_count": int((group["confidence_tier"] == "A").sum()),
        "confidence_B_count": int((group["confidence_tier"] == "B").sum()),
        "confidence_C_count": int((group["confidence_tier"] == "C").sum()),
        "first_window_start_utc": str(group["window_start_utc"].min()) if len(group) else "",
        "last_window_end_utc": str(group["window_end_utc"].max()) if len(group) else "",
    }


def summarize_annotation_alignment(annotated: pd.DataFrame) -> pd.DataFrame:
    """Summarize annotated maneuver windows per target plus an ALL row."""
    if annotated.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    for sat_id, group in annotated.groupby("sat_id", sort=True):
        batch = ",".join(sorted(set(group["batch"].astype(str))))
        rows.append(_summary_row(group, sat_id=sat_id, batch=batch, target_count=1))
    rows.append(_summary_row(annotated, sat_id="ALL", batch=",".join(sorted(set(annotated["batch"].astype(str)))), target_count=annotated["sat_id"].nunique()))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def write_annotated_event_windows(annotated: pd.DataFrame, output_root: str | Path) -> dict[str, Path]:
    """Write per-target annotated event-window tables under the normalized source root."""
    root = Path(output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    written: dict[str, Path] = {}
    for sat_id, group in annotated.groupby("sat_id", sort=True):
        target_dir = ensure_directory(root / str(sat_id))
        output_path = target_dir / "annotated_event_windows.csv"
        group.to_csv(output_path, index=False)
        written[str(sat_id)] = output_path
    return written


def write_summary(summary: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = REPO_ROOT / destination
    ensure_directory(destination.parent)
    summary.to_csv(destination, index=False)
    return destination


def generate_reference_annotation_alignment(
    config: dict,
    batches: list[str],
    scope: str = "all",
    days: int = 365,
    slr_margin_days: int = 1,
    as_of_utc: str | None = None,
    local_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return audit, annotated windows, and summary for configured reference targets."""
    audit = audit_reference_window_alignment(
        config,
        batches=batches,
        scope=scope,
        days=days,
        slr_margin_days=slr_margin_days,
        as_of_utc=as_of_utc,
        local_only=local_only,
    )
    annotated = build_annotated_event_windows(audit)
    summary = summarize_annotation_alignment(annotated)
    return audit, annotated, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reference annotation alignment tables")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--batch", action="append", default=None, help="Planning batch to include (repeatable); default covers both A and B")
    parser.add_argument("--scope", choices=["all", "recent"], default="all")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--slr-margin-days", type=int, default=1)
    parser.add_argument("--as-of-utc", default=None)
    parser.add_argument("--include-configured-missing", action="store_true")
    parser.add_argument(
        "--normalized-output-root",
        default="data/interim/normalized_sources/reference_validation",
        help="Root for per-target annotated_event_windows.csv outputs",
    )
    parser.add_argument(
        "--alignment-output",
        default="results/tables/reference_all_window_alignment_status.csv",
        help="Tracked all-window source-alignment decision table",
    )
    parser.add_argument(
        "--summary-output",
        default="results/tables/reference_annotation_alignment_summary.csv",
        help="Tracked annotation-alignment summary table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_annotation_config(args.config)
    audit, annotated, summary = generate_reference_annotation_alignment(
        config,
        batches=args.batch or ["A", "B"],
        scope=args.scope,
        days=args.days,
        slr_margin_days=args.slr_margin_days,
        as_of_utc=args.as_of_utc,
        local_only=not args.include_configured_missing,
    )
    alignment_path = write_alignment_audit(audit, args.alignment_output)
    annotated_paths = write_annotated_event_windows(annotated, resolve_repo_path(args.normalized_output_root))
    summary_path = write_summary(summary, args.summary_output)
    print(
        json.dumps(
            {
                "alignment_output": str(alignment_path),
                "annotated_target_count": len(annotated_paths),
                "annotated_window_count": int(len(annotated)),
                "summary_output": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
