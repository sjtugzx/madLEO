"""Generate core label decision tables for the validated core targets.

Registered as the ``core-label-tables`` experiment; the single writer of the
three shipped decision tables ``maneuver_annotation_summary.csv``,
``maneuver_evidence_alignment_summary.csv`` and
``maneuver_confidence_tier_summary.csv`` (REVIEW_FINDINGS C4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from benchmarking.config import REPO_ROOT, resolve_repo_path
from benchmarking.label_policy import summarize_gate_attrition
from benchmarking.schema import EVENT_LABELS


CORE_TARGETS = ("sentinel-3a", "sentinel-3b", "jason-3")

LABEL_SUMMARY_COLUMNS = [
    "sat_id",
    "track",
    "event_count",
    "no_event_count",
    "ignore_count",
    "total_public_windows",
    "needs_review_count",
    "auto_accepted_count",
    "excluded_count",
    "quality_flags",
]

SOURCE_AGREEMENT_COLUMNS = [
    "sat_id",
    "track",
    "pod_positive_count",
    "tle_positive_count",
    "slr_positive_count",
    "pod_tle_slr_agree_count",
    "tle_conflict_count",
    "slr_insufficient_count",
    "missing_pod_count",
    "missing_tle_count",
    "missing_slr_count",
    "quality_flags",
]

GATE_ATTRITION_COLUMNS = [
    "sat_id",
    "track",
    "pod_candidate_count",
    "tle_consistent_count",
    "tle_conflict_count",
    "slr_supportive_count",
    "slr_insufficient_count",
    "quality_pass_count",
    "event_count",
    "no_event_count",
    "ignore_count",
    "quality_flags",
]

MANEUVER_LABEL_SOURCE = "ids_doris_mission_reported_maneuver"
MANEUVER_CONFIDENCE_TIERS = ["A", "B", "C", "audit_only"]
MANEUVER_TIER_DEFINITIONS = {
    "A": "mission_reported_event_with_local_tle_slr_and_orbit_overlap",
    "B": "mission_reported_event_with_local_tle_and_orbit_overlap_slr_missing_or_insufficient",
    "C": "mission_reported_event_with_incomplete_local_source_coverage",
    "audit_only": "candidate_or_source_anomaly_without_mission_reported_event",
}

MANEUVER_ANNOTATION_SUMMARY_COLUMNS = [
    "sat_id",
    "batch",
    "target_count",
    "mission_reported_event_count",
    "public_event_count",
    "aligned_window_count",
    "first_window_start_utc",
    "last_window_end_utc",
    "annotation_label_source",
    "quality_flags",
]

MANEUVER_EVIDENCE_ALIGNMENT_SUMMARY_COLUMNS = [
    "sat_id",
    "batch",
    "mission_reported_event_count",
    "tle_covered_count",
    "slr_covered_count",
    "orbit_covered_count",
    "all_source_aligned_count",
    "missing_tle_count",
    "missing_slr_count",
    "missing_orbit_count",
    "local_slr_gap_count",
    "source_conflict_count",
    "evidence_scope",
    "quality_flags",
]

MANEUVER_CONFIDENCE_TIER_SUMMARY_COLUMNS = [
    "sat_id",
    "batch",
    "confidence_tier",
    "event_count",
    "event_fraction",
    "tier_definition",
    "annotation_label_source",
    "quality_flags",
]

COUNT_COLUMNS = {
    "label_summary": [
        "event_count",
        "no_event_count",
        "ignore_count",
        "total_public_windows",
        "needs_review_count",
        "auto_accepted_count",
        "excluded_count",
    ],
    "source_agreement_summary": [
        "pod_positive_count",
        "tle_positive_count",
        "slr_positive_count",
        "pod_tle_slr_agree_count",
        "tle_conflict_count",
        "slr_insufficient_count",
        "missing_pod_count",
        "missing_tle_count",
        "missing_slr_count",
    ],
    "gate_attrition_summary": [
        "pod_candidate_count",
        "tle_consistent_count",
        "tle_conflict_count",
        "slr_supportive_count",
        "slr_insufficient_count",
        "quality_pass_count",
        "event_count",
        "no_event_count",
        "ignore_count",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate core label decision summary tables")
    parser.add_argument(
        "--release-root",
        default="data/release/mad-leo/reference_validation_subset",
        help="Benchmark release root containing per-target label artifacts",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory for generated label decision tables",
    )
    parser.add_argument(
        "--reference-alignment-table",
        default="results/tables/reference_all_window_alignment_status.csv",
        help="Reference mission-reported event source-alignment table for maneuver summaries",
    )
    return parser.parse_args()


def quality_flags(*flags: str | None) -> str:
    values = [flag for flag in flags if flag]
    return "|".join(values) if values else "ok"


def safe_json_loads(value: Any) -> Dict[str, Any]:
    if value in (None, "") or pd.isna(value):
        return {}
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _missing_sources(value: Any) -> set[str]:
    if value in (None, "") or pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "") or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _coverage_tier(row: pd.Series) -> str:
    has_tle = row.get("tle_status") == "covered"
    has_slr = row.get("slr_status") == "covered"
    has_orbit = row.get("orbit_status") == "covered"
    if has_tle and has_slr and has_orbit:
        return "A"
    if has_tle and has_orbit:
        return "B"
    return "C"


def _batch_value(group: pd.DataFrame) -> str:
    if group.empty or "batch" not in group.columns:
        return ""
    return ",".join(sorted(set(group["batch"].dropna().astype(str))))


def _quality_for_group(group: pd.DataFrame) -> str:
    return quality_flags("coverage_gaps_present" if not group["aligned"].map(_as_bool).all() else None)


def _track(labels: pd.DataFrame) -> str:
    if "track" in labels.columns and not labels.empty:
        tracks = labels["track"].dropna().astype(str)
        if len(tracks) > 0:
            return tracks.iloc[0]
    return "validated"


def summarize_labels(labels: pd.DataFrame, sat_id: str) -> Dict[str, Any]:
    flags = []
    if labels.empty:
        flags.append("label_audit_empty")
    elif not set(labels.get("event_label", pd.Series(dtype=str)).dropna()).issubset(EVENT_LABELS):
        flags.append("unsupported_public_label")

    label_series = labels.get("event_label", pd.Series(dtype=str))
    review_series = labels.get("review_status", pd.Series(dtype=str))
    supported = labels[label_series.isin(EVENT_LABELS)] if not labels.empty else labels
    return {
        "sat_id": sat_id,
        "track": _track(labels),
        "event_count": int((supported.get("event_label", pd.Series(dtype=str)) == "event").sum()),
        "no_event_count": int((supported.get("event_label", pd.Series(dtype=str)) == "no_event").sum()),
        "ignore_count": int((supported.get("event_label", pd.Series(dtype=str)) == "ignore").sum()),
        "total_public_windows": int(len(supported)),
        "needs_review_count": int((review_series == "needs_review").sum()),
        "auto_accepted_count": int((review_series == "auto_accepted").sum()),
        "excluded_count": int((review_series == "excluded").sum()),
        "quality_flags": quality_flags(*flags),
    }


def summarize_source_agreement(labels: pd.DataFrame, sat_id: str) -> Dict[str, Any]:
    flags = []
    if labels.empty:
        flags.append("label_audit_empty")

    signal_labels = labels[labels.get("event_label", pd.Series(dtype=str)) != "no_event"] if not labels.empty else labels
    sources = signal_labels.get(
        "evidence_sources",
        pd.Series([""] * len(signal_labels), dtype=str),
    ).fillna("").astype(str)
    statuses = labels.get("gate_status_json", pd.Series(["{}"] * len(labels), dtype=str)).apply(safe_json_loads)
    source_sets = sources.apply(lambda value: {item for item in value.split(",") if item})

    return {
        "sat_id": sat_id,
        "track": _track(labels),
        "pod_positive_count": int(sum("pod" in items for items in source_sets)),
        "tle_positive_count": int(sum("tle" in items for items in source_sets)),
        "slr_positive_count": int(sum("slr" in items for items in source_sets)),
        "pod_tle_slr_agree_count": int(sum({"pod", "tle", "slr"}.issubset(items) for items in source_sets)),
        "tle_conflict_count": int(sum(status.get("tle") == "conflict" for status in statuses)),
        "slr_insufficient_count": int(sum(status.get("slr") == "insufficient" for status in statuses)),
        "missing_pod_count": int(sum(status.get("pod") == "missing" for status in statuses)),
        "missing_tle_count": int(sum(status.get("tle") == "missing" for status in statuses)),
        "missing_slr_count": int(sum(status.get("slr") == "missing" for status in statuses)),
        "quality_flags": quality_flags(*flags),
    }


def summarize_attrition(labels: pd.DataFrame, attrition: pd.DataFrame, sat_id: str) -> Dict[str, Any]:
    flags = []
    if labels.empty:
        flags.append("label_audit_empty")
    if attrition.empty and not labels.empty:
        attrition = summarize_gate_attrition(labels, sat_id=sat_id, track=_track(labels))
        flags.append("gate_attrition_recomputed")

    if attrition.empty:
        row = {column: 0 for column in GATE_ATTRITION_COLUMNS if column.endswith("_count")}
        row.update({"sat_id": sat_id, "track": _track(labels)})
    else:
        first = attrition.iloc[0].to_dict()
        row = {
            "sat_id": sat_id,
            "track": str(first.get("track", _track(labels))),
        }
        for column in GATE_ATTRITION_COLUMNS:
            if column.endswith("_count"):
                row[column] = int(first.get(column, 0) or 0)

    row["quality_flags"] = quality_flags(*flags)
    return row


def add_aggregate_row(df: pd.DataFrame, table_key: str, columns: List[str]) -> pd.DataFrame:
    count_columns = COUNT_COLUMNS[table_key]
    target_rows = df[df["sat_id"] != "ALL"]
    aggregate = {
        "sat_id": "ALL",
        "track": "validated",
        "quality_flags": quality_flags(
            "target_quality_flags_present" if (target_rows["quality_flags"] != "ok").any() else None
        ),
    }
    for column in count_columns:
        aggregate[column] = int(target_rows[column].fillna(0).astype(int).sum())
    return pd.concat([df, pd.DataFrame([aggregate])], ignore_index=True)[columns]


def build_label_tables(release_root: Path, targets: Iterable[str]) -> Dict[str, pd.DataFrame]:
    label_rows: List[Dict[str, Any]] = []
    agreement_rows: List[Dict[str, Any]] = []
    attrition_rows: List[Dict[str, Any]] = []

    for sat_id in targets:
        target_dir = release_root / sat_id
        labels = read_csv_or_empty(target_dir / "label_audit.csv")
        attrition = read_csv_or_empty(target_dir / "gate_attrition.csv")
        label_rows.append(summarize_labels(labels, sat_id))
        agreement_rows.append(summarize_source_agreement(labels, sat_id))
        attrition_rows.append(summarize_attrition(labels, attrition, sat_id))

    label_summary = pd.DataFrame(label_rows, columns=LABEL_SUMMARY_COLUMNS)
    source_agreement = pd.DataFrame(agreement_rows, columns=SOURCE_AGREEMENT_COLUMNS)
    gate_attrition = pd.DataFrame(attrition_rows, columns=GATE_ATTRITION_COLUMNS)

    return {
        "label_summary": add_aggregate_row(label_summary, "label_summary", LABEL_SUMMARY_COLUMNS),
        "source_agreement_summary": add_aggregate_row(
            source_agreement,
            "source_agreement_summary",
            SOURCE_AGREEMENT_COLUMNS,
        ),
        "gate_attrition_summary": add_aggregate_row(
            gate_attrition,
            "gate_attrition_summary",
            GATE_ATTRITION_COLUMNS,
        ),
    }


def _maneuver_annotation_summary_row(group: pd.DataFrame, sat_id: str, batch: str, target_count: int) -> Dict[str, Any]:
    return {
        "sat_id": sat_id,
        "batch": batch,
        "target_count": target_count,
        "mission_reported_event_count": int(len(group)),
        "public_event_count": int(len(group)),
        "aligned_window_count": int(group["aligned"].map(_as_bool).sum()) if not group.empty else 0,
        "first_window_start_utc": str(group["window_start_utc"].min()) if not group.empty else "",
        "last_window_end_utc": str(group["window_end_utc"].max()) if not group.empty else "",
        "annotation_label_source": MANEUVER_LABEL_SOURCE,
        "quality_flags": _quality_for_group(group) if not group.empty else "empty_annotation_alignment",
    }


def _maneuver_evidence_summary_row(group: pd.DataFrame, sat_id: str, batch: str) -> Dict[str, Any]:
    missing_sets = group.get("missing_sources", pd.Series([""] * len(group))).apply(_missing_sources)
    local_slr_gap = (
        (group.get("tle_status", pd.Series(dtype=str)) == "covered")
        & (group.get("orbit_status", pd.Series(dtype=str)) == "covered")
        & (group.get("slr_status", pd.Series(dtype=str)) != "covered")
    )
    return {
        "sat_id": sat_id,
        "batch": batch,
        "mission_reported_event_count": int(len(group)),
        "tle_covered_count": int((group.get("tle_status", pd.Series(dtype=str)) == "covered").sum()),
        "slr_covered_count": int((group.get("slr_status", pd.Series(dtype=str)) == "covered").sum()),
        "orbit_covered_count": int((group.get("orbit_status", pd.Series(dtype=str)) == "covered").sum()),
        "all_source_aligned_count": int(group["aligned"].map(_as_bool).sum()) if not group.empty else 0,
        "missing_tle_count": int(sum("tle" in values for values in missing_sets)),
        "missing_slr_count": int(sum("slr" in values for values in missing_sets)),
        "missing_orbit_count": int(sum("orbit" in values for values in missing_sets)),
        "local_slr_gap_count": int(local_slr_gap.sum()) if not group.empty else 0,
        "source_conflict_count": 0,
        "evidence_scope": "local_event_window_overlap",
        "quality_flags": _quality_for_group(group) if not group.empty else "empty_annotation_alignment",
    }


def _tier_rows(group: pd.DataFrame, sat_id: str, batch: str) -> list[Dict[str, Any]]:
    tiers = group["confidence_tier"].value_counts() if "confidence_tier" in group.columns else pd.Series(dtype=int)
    total = int(len(group))
    rows = []
    for tier in MANEUVER_CONFIDENCE_TIERS:
        count = int(tiers.get(tier, 0))
        rows.append(
            {
                "sat_id": sat_id,
                "batch": batch,
                "confidence_tier": tier,
                "event_count": count,
                "event_fraction": round(count / total, 6) if total else 0.0,
                "tier_definition": MANEUVER_TIER_DEFINITIONS[tier],
                "annotation_label_source": MANEUVER_LABEL_SOURCE,
                "quality_flags": "ok",
            }
        )
    return rows


def build_maneuver_label_tables(alignment_audit: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build mission-reported maneuver label and evidence decision tables."""
    if alignment_audit.empty:
        return {
            "maneuver_annotation_summary": pd.DataFrame(columns=MANEUVER_ANNOTATION_SUMMARY_COLUMNS),
            "maneuver_evidence_alignment_summary": pd.DataFrame(columns=MANEUVER_EVIDENCE_ALIGNMENT_SUMMARY_COLUMNS),
            "maneuver_confidence_tier_summary": pd.DataFrame(columns=MANEUVER_CONFIDENCE_TIER_SUMMARY_COLUMNS),
        }

    annotated = alignment_audit.copy()
    annotated["confidence_tier"] = annotated.apply(_coverage_tier, axis=1)
    annotation_rows: list[Dict[str, Any]] = []
    evidence_rows: list[Dict[str, Any]] = []
    tier_rows: list[Dict[str, Any]] = []

    for sat_id, group in annotated.groupby("sat_id", sort=True):
        batch = _batch_value(group)
        annotation_rows.append(_maneuver_annotation_summary_row(group, sat_id=sat_id, batch=batch, target_count=1))
        evidence_rows.append(_maneuver_evidence_summary_row(group, sat_id=sat_id, batch=batch))
        tier_rows.extend(_tier_rows(group, sat_id=sat_id, batch=batch))

    all_batch = _batch_value(annotated)
    annotation_rows.append(
        _maneuver_annotation_summary_row(
            annotated,
            sat_id="ALL",
            batch=all_batch,
            target_count=int(annotated["sat_id"].nunique()),
        )
    )
    evidence_rows.append(_maneuver_evidence_summary_row(annotated, sat_id="ALL", batch=all_batch))
    tier_rows.extend(_tier_rows(annotated, sat_id="ALL", batch=all_batch))

    return {
        "maneuver_annotation_summary": pd.DataFrame(annotation_rows, columns=MANEUVER_ANNOTATION_SUMMARY_COLUMNS),
        "maneuver_evidence_alignment_summary": pd.DataFrame(
            evidence_rows,
            columns=MANEUVER_EVIDENCE_ALIGNMENT_SUMMARY_COLUMNS,
        ),
        "maneuver_confidence_tier_summary": pd.DataFrame(tier_rows, columns=MANEUVER_CONFIDENCE_TIER_SUMMARY_COLUMNS),
    }


def write_maneuver_label_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["maneuver_annotation_summary"].to_csv(output_dir / "maneuver_annotation_summary.csv", index=False)
    tables["maneuver_evidence_alignment_summary"].to_csv(
        output_dir / "maneuver_evidence_alignment_summary.csv",
        index=False,
    )
    tables["maneuver_confidence_tier_summary"].to_csv(
        output_dir / "maneuver_confidence_tier_summary.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(resolve_repo_path(args.output_dir) or (REPO_ROOT / args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_table = Path(
        resolve_repo_path(args.reference_alignment_table) or (REPO_ROOT / args.reference_alignment_table)
    )
    if not alignment_table.exists():
        raise FileNotFoundError(
            f"Reference mission-reported alignment table is required for Task 9 maneuver summaries: {alignment_table}"
        )
    release_root = Path(resolve_repo_path(args.release_root) or (REPO_ROOT / args.release_root))
    if release_root.exists():
        tables = build_label_tables(release_root, CORE_TARGETS)
        tables["label_summary"].to_csv(output_dir / "core_label_summary.csv", index=False)
        tables["source_agreement_summary"].to_csv(output_dir / "core_source_agreement_summary.csv", index=False)
        tables["gate_attrition_summary"].to_csv(output_dir / "core_gate_attrition_summary.csv", index=False)
    write_maneuver_label_tables(build_maneuver_label_tables(read_csv_or_empty(alignment_table)), output_dir)


if __name__ == "__main__":
    main()
