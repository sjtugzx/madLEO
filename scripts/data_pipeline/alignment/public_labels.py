"""Export release-facing public label tables from the annotation protocol.

Builds `events.csv` (mission-reported maneuver windows with confidence
tiers) and `stable_windows.csv` (strict no-event windows) per target, clipped
to the configured release window. Audit tables keep full mission coverage;
only release-facing exports are clipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path

DEFAULT_ANNOTATED_ROOT = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"
DEFAULT_STABLE_TABLE = REPO_ROOT / "results" / "tables" / "stable_windows.csv"
# Interim staging tree for the per-target public label view. The shipped public
# label surface of the MAD-LEO release is the combined annotations/ tables
# written by release_export; this export must never target a release tree
# (the retired leo-orbital-dynamics tree was silently double-written before
# this default moved to interim).
DEFAULT_RELEASE_ROOT = REPO_ROOT / "data" / "interim" / "public_labels"
# Public label tables keep full mission history. The lower bound only rules
# out pre-mission garbage; the upper end is deliberately UNBOUNDED (C21): a
# hard-coded cutoff (the retired 2026-04-15) silently dropped mission records
# that post-dated it (shipped data runs to 2026-07). Clip the end explicitly
# only when a caller passes its own release window.
DEFAULT_RELEASE_WINDOW: tuple[str, str | None] = ("1900-01-01T00:00:00Z", None)

PUBLIC_COLUMNS = [
    "annotation_id",
    "sat_id",
    "event_label",
    "confidence_tier",
    "event_time_utc",
    "window_start_utc",
    "window_end_utc",
    "quality_flags",
]


def parse_release_window(
    window: tuple[str, str | None] | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """Return the (start, end) release window as UTC timestamps.

    ``end`` may be ``None`` (unbounded) -- the default keeps full mission
    history instead of clipping at a stale hard-coded cutoff (C21).
    """
    start_raw, end_raw = window if window is not None else DEFAULT_RELEASE_WINDOW
    start = pd.to_datetime(start_raw, utc=True)
    end = pd.to_datetime(end_raw, utc=True) if end_raw is not None else None
    return start, end


def clip_to_release_window(
    frame: pd.DataFrame,
    release_start: pd.Timestamp,
    release_end: pd.Timestamp | None,
) -> tuple[pd.DataFrame, int]:
    """Keep rows fully inside the release window; return (clipped, dropped).

    A ``None`` end leaves the upper bound open (full mission history).
    """
    if frame.empty:
        return frame, 0
    starts = pd.to_datetime(frame["window_start_utc"], utc=True)
    ends = pd.to_datetime(frame["window_end_utc"], utc=True)
    keep = starts >= release_start
    if release_end is not None:
        keep &= ends <= release_end
    return frame[keep].reset_index(drop=True), int((~keep).sum())


def _required_columns(rows: pd.DataFrame) -> list[str]:
    """Fail loudly if the upstream schema dropped a public column."""
    missing = [c for c in PUBLIC_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"public label export missing required columns: {missing}")
    return PUBLIC_COLUMNS


def _public_event_rows(annotated: pd.DataFrame) -> pd.DataFrame:
    rows = annotated.copy()
    rows["event_label"] = "event"
    if "quality_flags" not in rows.columns:
        rows["quality_flags"] = ""
    if "event_time_utc" not in rows.columns:
        rows["event_time_utc"] = ""
    return rows[_required_columns(rows)]


def _public_stable_rows(stable: pd.DataFrame) -> pd.DataFrame:
    rows = stable.copy()
    rows["event_label"] = "no_event"
    if "event_time_utc" not in rows.columns:
        rows["event_time_utc"] = ""
    if "quality_flags" not in rows.columns:
        rows["quality_flags"] = ""
    return rows[_required_columns(rows)]


def export_public_label_tables(
    sat_ids: list[str],
    annotated_root: str | Path = DEFAULT_ANNOTATED_ROOT,
    stable_table: str | Path = DEFAULT_STABLE_TABLE,
    release_root: str | Path = DEFAULT_RELEASE_ROOT,
    release_window: tuple[str, str | None] | None = None,
) -> dict[str, Any]:
    """Write clipped per-target public label tables into the release subset."""
    release_start, release_end = parse_release_window(release_window)
    annotated_root = resolve_repo_path(str(annotated_root))
    release_root = resolve_repo_path(str(release_root))
    stable_path = resolve_repo_path(str(stable_table))
    stable = pd.read_csv(stable_path) if stable_path.exists() else pd.DataFrame()

    metadata: dict[str, Any] = {
        "release_window": [str(release_start), str(release_end) if release_end is not None else "unbounded"],
        "targets": {},
    }
    for sat_id in sat_ids:
        annotated_path = annotated_root / sat_id / "annotated_event_windows.csv"
        events = pd.read_csv(annotated_path) if annotated_path.exists() else pd.DataFrame()
        events_missing_event_time = 0
        if not events.empty:
            # The window table places the annotation; the mission-reported epoch
            # lives in maneuver_annotations.csv keyed by the same annotation_id.
            # Backfill when the column is absent or shipped unfilled (the
            # retired release tree carried an empty event_time_utc column).
            annotations_path = annotated_root / sat_id / "maneuver_annotations.csv"
            if annotations_path.exists():
                if "event_time_utc" in events.columns:
                    filled = events["event_time_utc"].notna() & (events["event_time_utc"].astype(str).str.strip() != "")
                    if not bool(filled.any()):
                        events = events.drop(columns=["event_time_utc"])
                if "event_time_utc" not in events.columns:
                    annotations = pd.read_csv(annotations_path, usecols=["annotation_id", "event_time_utc"])
                    events = events.merge(annotations, on="annotation_id", how="left")
                    events_missing_event_time = int(events["event_time_utc"].isna().sum())
                    events["event_time_utc"] = events["event_time_utc"].fillna("")
            events = _public_event_rows(events)
        events, events_dropped = clip_to_release_window(events, release_start, release_end)

        sat_stable = stable[stable["sat_id"].astype(str) == sat_id].copy() if not stable.empty else pd.DataFrame()
        sat_stable = _public_stable_rows(sat_stable) if not sat_stable.empty else sat_stable
        sat_stable, stable_dropped = clip_to_release_window(sat_stable, release_start, release_end)

        target_dir = ensure_directory(release_root / sat_id)
        events.to_csv(target_dir / "events.csv", index=False)
        sat_stable.to_csv(target_dir / "stable_windows.csv", index=False)
        metadata["targets"][sat_id] = {
            "events": len(events),
            "events_clipped_out_of_release_window": events_dropped,
            "events_missing_event_time": events_missing_event_time,
            "stable_windows": len(sat_stable),
            "stable_windows_clipped_out_of_release_window": stable_dropped,
        }
    (release_root / "public_labels_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    """CLI entry: export clipped public label tables for the 11 targets."""
    import argparse

    parser = argparse.ArgumentParser(description="Export release-clipped public label tables")
    parser.add_argument("--annotated-root", default=str(DEFAULT_ANNOTATED_ROOT))
    parser.add_argument("--stable-table", default=str(DEFAULT_STABLE_TABLE))
    parser.add_argument("--release-root", default=str(DEFAULT_RELEASE_ROOT))
    parser.add_argument("--include", action="append", default=None, help="sat_id filter (repeatable)")
    args = parser.parse_args()

    default_targets = [
        "sentinel-3a", "sentinel-3b", "jason-3", "sentinel-6a", "cryosat-2", "saral",
        "jason-1", "jason-2", "topex-poseidon", "hy-2a", "swot",
    ]
    metadata = export_public_label_tables(
        args.include or default_targets,
        annotated_root=args.annotated_root,
        stable_table=args.stable_table,
        release_root=args.release_root,
    )
    summary = {k: (v["events"], v["stable_windows"]) for k, v in metadata["targets"].items()}
    clipped = sum(v["events_clipped_out_of_release_window"] + v["stable_windows_clipped_out_of_release_window"] for v in metadata["targets"].values())
    print(json.dumps({"targets": summary, "clipped_total": clipped}, sort_keys=True))
    return 0
