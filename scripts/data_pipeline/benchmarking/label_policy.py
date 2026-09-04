"""Track-specific annotation rules for event, no-event, and ignore windows."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import pandas as pd

from benchmarking.schema import event_row
from benchmarking.window_quality import evaluate_window_quality


MANEUVER_TO_FAMILY = {
    "orbit_raise": ("orbit_raise", "semi_major_axis"),
    "orbit_lower": ("orbit_lower", "semi_major_axis"),
    "plane_change": ("plane_change", "inclination"),
    "phasing": ("unknown_maneuver", "mean_longitude"),
    "station_keeping": ("station_keeping", "multi_axis"),
}


def _source_summary(evidence_slice: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for row in evidence_slice.itertuples(index=False):
        source_info = summary.setdefault(row.source, {"positive_score": 0.0, "negative_score": 0.0, "count": 0})
        if row.supports_event:
            source_info["positive_score"] = max(source_info["positive_score"], float(row.score))
        else:
            source_info["negative_score"] = max(source_info["negative_score"], float(row.score))
        source_info["count"] += 1
    return summary


def _infer_semantics(evidence_slice: pd.DataFrame) -> tuple[str, str]:
    for row in evidence_slice.itertuples(index=False):
        metrics = json.loads(row.metrics_json)
        maneuver_type = metrics.get("maneuver_type")
        if maneuver_type in MANEUVER_TO_FAMILY:
            return MANEUVER_TO_FAMILY[maneuver_type]
    return "unknown_maneuver", "multi_axis"


MAX_LABEL_EVIDENCE_IDS = 20


def _evidence_ids(evidence_slice: pd.DataFrame, limit: int = MAX_LABEL_EVIDENCE_IDS) -> List[str]:
    return [str(row.evidence_id) for row in evidence_slice.head(limit).itertuples(index=False)]


def _first_missing_required(required_sources: Iterable[str], positive_sources: set[str]) -> str:
    for source in ["pod", "tle", "slr"]:
        if source in required_sources and source not in positive_sources:
            return f"missing_{source}"
    for source in sorted(set(required_sources) - positive_sources):
        return f"missing_{source}"
    return ""


def _has_source_conflict(evidence_slice: pd.DataFrame, source: str) -> bool:
    source_rows = evidence_slice[evidence_slice["source"] == source]
    return bool((source_rows["supports_event"] == False).any())


def annotate_candidates(
    candidates_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    track: str,
    policy: Dict,
    source_tables: Dict[str, pd.DataFrame] | None = None,
    release_start: object | None = None,
    release_end: object | None = None,
    max_gap_hours_by_source: Dict[str, float] | None = None,
    boundary_margin_hours: float = 0,
) -> pd.DataFrame:
    """Annotate fused candidates into event or ignore labels."""
    if candidates_df is None or len(candidates_df) == 0:
        return pd.DataFrame()

    rows: List[Dict] = []
    track_policy = policy.get(track, {})
    required_sources = set(track_policy.get("required_sources", []))
    support_sources = set(track_policy.get("support_sources", []))
    min_combined_score = float(track_policy.get("min_combined_score", 0.8))
    if evidence_df is None or evidence_df.empty or "candidate_id" not in evidence_df.columns:
        evidence_groups: Dict[str, pd.DataFrame] = {}
    else:
        evidence_groups = {
            str(candidate_id): group
            for candidate_id, group in evidence_df.groupby("candidate_id", sort=False)
        }

    for candidate in candidates_df.itertuples(index=False):
        evidence_slice = evidence_groups.get(str(candidate.candidate_id), pd.DataFrame(columns=evidence_df.columns))
        source_scores = _source_summary(evidence_slice)
        positive_sources = {source for source, stats in source_scores.items() if stats["positive_score"] > 0}
        combined_score = float(sum(stats["positive_score"] for stats in source_scores.values()))
        event_family, orbital_effect = _infer_semantics(evidence_slice)
        ignore_reason = ""
        quality_flags: List[str] = []
        gate_status = {
            "pod": "pass" if "pod" in positive_sources else "missing",
            "tle": "pass" if "tle" in positive_sources else "missing",
            "slr": "pass" if "slr" in positive_sources else "missing",
        }
        quality = None
        if source_tables is not None or release_start is not None or release_end is not None:
            quality = evaluate_window_quality(
                source_tables=source_tables,
                start=candidate.t_start_lo,
                end=candidate.t_end_hi,
                required_sources=required_sources,
                release_start=release_start,
                release_end=release_end,
                max_gap_hours_by_source=max_gap_hours_by_source,
                boundary_margin_hours=boundary_margin_hours,
            )
            quality_flags = quality["flags"]
            for source, status in quality["source_status"].items():
                if status != "pass":
                    gate_status[source] = status
        if _has_source_conflict(evidence_slice, "tle"):
            ignore_reason = "tle_conflict"
            gate_status["tle"] = "conflict"
        if not ignore_reason and _has_source_conflict(evidence_slice, "slr"):
            slr_rows = evidence_slice[evidence_slice["source"] == "slr"]
            if (slr_rows["details"] == "insufficient_slr_coverage").any():
                ignore_reason = "slr_insufficient_coverage"
                gate_status["slr"] = "insufficient"
            else:
                ignore_reason = "slr_conflict"
                gate_status["slr"] = "conflict"

        if ignore_reason:
            label = "ignore"
        elif quality is not None and not quality["ok"]:
            label = "ignore"
            ignore_reason = quality["reason"]
        elif required_sources.issubset(positive_sources):
            if support_sources:
                support_ok = bool(support_sources.intersection(positive_sources))
            else:
                support_ok = True
            if support_ok and combined_score >= min_combined_score:
                label = "event"
            else:
                label = "ignore"
                ignore_reason = "combined_score_below_threshold"
        else:
            label = "ignore"
            ignore_reason = _first_missing_required(required_sources, positive_sources)

        review_status = "needs_review" if label == "event" else "excluded"
        notes = f"combined_score={combined_score:.3f}"
        truncated_count = max(0, len(evidence_slice) - MAX_LABEL_EVIDENCE_IDS)
        if truncated_count:
            notes = f"{notes};evidence_ids_truncated={truncated_count}"
        rows.append(
            event_row(
                sat_id=candidate.sat_id,
                track=track,
                event_label=label,
                t_start_lo=candidate.t_start_lo,
                t_start_hi=candidate.t_start_hi,
                t_end_lo=candidate.t_end_lo,
                t_end_hi=candidate.t_end_hi,
                event_family=event_family if label == "event" else "none",
                orbital_effect=orbital_effect if label == "event" else "none",
                evidence_sources=positive_sources,
                review_status=review_status,
                notes=notes,
                candidate_id=candidate.candidate_id,
                evidence_ids=_evidence_ids(evidence_slice),
                quality_flags=quality_flags,
                ignore_reason=ignore_reason,
                gate_status=gate_status,
            )
        )

    return pd.DataFrame(rows)


def summarize_gate_attrition(labels_df: pd.DataFrame, sat_id: str, track: str) -> pd.DataFrame:
    """Summarize label gate outcomes for one target."""
    if labels_df is None or labels_df.empty:
        return pd.DataFrame(
            [
                {
                    "sat_id": sat_id,
                    "track": track,
                    "pod_candidate_count": 0,
                    "tle_consistent_count": 0,
                    "tle_conflict_count": 0,
                    "slr_supportive_count": 0,
                    "slr_insufficient_count": 0,
                    "quality_pass_count": 0,
                    "event_count": 0,
                    "no_event_count": 0,
                    "ignore_count": 0,
                }
            ]
        )

    raw_statuses = labels_df.get("gate_status_json", pd.Series(["{}"] * len(labels_df)))
    statuses = raw_statuses.fillna("{}").apply(json.loads)
    return pd.DataFrame(
        [
            {
                "sat_id": sat_id,
                "track": track,
                "pod_candidate_count": int(len(labels_df)),
                "tle_consistent_count": int(sum(status.get("tle") == "pass" for status in statuses)),
                "tle_conflict_count": int(sum(status.get("tle") == "conflict" for status in statuses)),
                "slr_supportive_count": int(sum(status.get("slr") == "pass" for status in statuses)),
                "slr_insufficient_count": int(sum(status.get("slr") == "insufficient" for status in statuses)),
                "quality_pass_count": int(sum(labels_df["event_label"].isin(["event", "no_event"]))),
                "event_count": int((labels_df["event_label"] == "event").sum()),
                "no_event_count": int((labels_df["event_label"] == "no_event").sum()),
                "ignore_count": int((labels_df["event_label"] == "ignore").sum()),
            }
        ]
    )


# REVIEW_FINDINGS C20: ``mine_no_event_windows`` (a second, 24-h
# stable-window mining scheme) and its private helpers were removed --
# the released no-event windows come from
# ``benchmarking.stable_windows.mine_stable_windows`` only.
