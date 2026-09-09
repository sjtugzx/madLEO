"""Mine stable (no-event) windows for reference validation targets.

Stable windows are the negative counterpart to the mission-reported maneuver
event windows. Design decisions (kept simple and defensible for the paper
artifact, not ML training data):

- Mission coverage per target is derived from its local TLE archive: the
  first to the last TLE epoch bounds the interval in which the target is
  observable at all.
- Candidate stable windows use the same 30h duration as the event windows
  (event time -6h .. +24h semantics). Candidates are laid down on a
  non-overlapping grid with stride equal to the window duration, anchored at
  the first TLE epoch, so generation is deterministic without any RNG.
- A candidate is rejected if it overlaps any mission-reported event window
  expanded by a safety buffer (default 24h) on both sides. Event windows
  themselves already carry the -6h/+24h asymmetry, so the buffer is applied
  symmetrically on top of the recorded window bounds.
- Selection quota: per calendar year, the number of stable windows equals the
  number of event windows starting in that year (minimum 1 per year that has
  any valid candidate). This keeps each target's stable-window total roughly
  comparable to its event count while spreading windows across the mission.
  Within a year, windows are picked evenly spaced from the valid candidate
  list (deterministic index arithmetic, no randomness).
- Local-coverage status (TLE/SLR/orbit) and the resulting confidence tier
  mirror the event-window audit exactly: TLE is "covered" when the archive
  has an epoch at or before the window start and at or after the window end;
  SLR is "covered" when a local SLR file span overlaps the window plus a
  1-day margin; orbit is "covered" when a local POD/precise-orbit product
  span overlaps the window. Tier A = all three covered, B = TLE + orbit,
  C = otherwise (same rule as
  ``generate_reference_annotation_alignment._coverage_confidence_tier``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from benchmarking.experiment_params import (
    ANALYSIS_WINDOW_POST_HOURS,
    ANALYSIS_WINDOW_PRE_HOURS,
    SLR_COVERAGE_MARGIN_DAYS,
    STABLE_EXCLUSION_BUFFER_HOURS,
)

# P4 single source: the stable-window geometry derives from the shared
# analysis-window parameters (event at start + 6 h, 30 h total), the SLR
# coverage margin, and the canonical exclusion buffer; nothing here
# re-defines a numeric policy value.
WINDOW_HOURS = ANALYSIS_WINDOW_PRE_HOURS + ANALYSIS_WINDOW_POST_HOURS
EXCLUSION_BUFFER_HOURS = STABLE_EXCLUSION_BUFFER_HOURS
SLR_MARGIN_DAYS = int(SLR_COVERAGE_MARGIN_DAYS)
ANNOTATION_LABEL_SOURCE = "tle_archive_mission_coverage_mining"

STABLE_WINDOW_COLUMNS = [
    "annotation_id",
    "sat_id",
    "batch",
    "window_start_utc",
    "window_end_utc",
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
    "mission_start_utc",
    "mission_end_utc",
    "mission_years",
    "event_window_count",
    "requested_stable_window_count",
    "stable_window_count",
    "aligned_window_count",
    "tle_covered_count",
    "slr_covered_count",
    "orbit_covered_count",
    "confidence_A_count",
    "confidence_B_count",
    "confidence_C_count",
    "constrained",
    "notes",
]


def _parse_utc(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, format="ISO8601")


def iso_utc(value: object) -> str:
    """Format a timestamp as ISO-8601 UTC ending in Z (release schema)."""
    if value is None or pd.isna(value):
        return ""
    ts = pd.to_datetime(value, utc=True, format="ISO8601")
    if ts.microsecond:
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def mission_coverage_from_tle_epochs(epochs: Iterable[pd.Timestamp]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (first, last) TLE epoch bounding the observable mission interval."""
    ordered = sorted(pd.to_datetime(list(epochs), utc=True, format="ISO8601"))
    if not ordered:
        return None
    return ordered[0], ordered[-1]


def merge_intervals(intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge overlapping or touching (start, end) intervals."""
    merged: list[list[pd.Timestamp]] = []
    for start, end in sorted(intervals, key=lambda pair: pair[0]):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_exclusion_intervals(
    event_windows: pd.DataFrame,
    buffer_hours: float = EXCLUSION_BUFFER_HOURS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Expand every event window by the safety buffer on both sides and merge."""
    if event_windows is None or event_windows.empty:
        return []
    buffer = pd.Timedelta(hours=buffer_hours)
    intervals = [
        (_parse_utc(row.window_start_utc) - buffer, _parse_utc(row.window_end_utc) + buffer)
        for row in event_windows.itertuples(index=False)
    ]
    return merge_intervals(intervals)


def candidate_windows(
    mission_start: pd.Timestamp,
    mission_end: pd.Timestamp,
    exclusions: list[tuple[pd.Timestamp, pd.Timestamp]],
    window_hours: float = WINDOW_HOURS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Lay down non-overlapping candidate windows over the mission interval.

    The grid is anchored at ``mission_start`` with stride equal to the window
    duration; candidates overlapping a merged exclusion interval are dropped.
    """
    window = pd.Timedelta(hours=window_hours)
    candidates: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = mission_start
    while cursor + window <= mission_end:
        start = cursor
        end = cursor + window
        overlaps = any(start < excl_end and end > excl_start for excl_start, excl_end in exclusions)
        if not overlaps:
            candidates.append((start, end))
        cursor += window
    return candidates


def _yearly_event_counts(event_windows: pd.DataFrame) -> dict[int, int]:
    if event_windows is None or event_windows.empty:
        return {}
    years = pd.to_datetime(event_windows["window_start_utc"], utc=True, format="ISO8601").dt.year
    return years.value_counts().to_dict()


def select_windows_per_year(
    candidates: list[tuple[pd.Timestamp, pd.Timestamp]],
    yearly_quota: dict[int, int],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Pick evenly spaced windows per calendar year, deterministically.

    Each year's quota comes from ``yearly_quota``; years present among the
    candidates but absent from the quota dict still get one window (minimum
    coverage). Within a year the picks are evenly spaced indices into the
    year's candidate list, so the result is a pure function of the inputs.
    """
    by_year: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for start, end in candidates:
        by_year.setdefault(start.year, []).append((start, end))

    selected: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for year in sorted(by_year):
        year_candidates = by_year[year]
        quota = max(1, int(yearly_quota.get(year, 0)))
        take = min(quota, len(year_candidates))
        if take == len(year_candidates):
            selected.extend(year_candidates)
            continue
        # Evenly spaced deterministic indices, deduplicated, order preserved.
        last = len(year_candidates) - 1
        indices = sorted({round(i * last / (take - 1)) for i in range(take)}) if take > 1 else [last // 2]
        selected.extend(year_candidates[index] for index in indices)
    return sorted(selected, key=lambda pair: pair[0])


def coverage_confidence_tier(tle_status: str, slr_status: str, orbit_status: str) -> str:
    """Coverage-based tier, identical to the event-window annotation rule."""
    has_tle = tle_status == "covered"
    has_slr = slr_status == "covered"
    has_orbit = orbit_status == "covered"
    if has_tle and has_slr and has_orbit:
        return "A"
    if has_tle and has_orbit:
        return "B"
    return "C"


def _span_overlaps(span: dict, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    span_start = _parse_utc(span["start_utc"])
    span_end = _parse_utc(span["end_utc"])
    return bool(span_start < end and span_end > start)


def assess_window_coverage(
    start: pd.Timestamp,
    end: pd.Timestamp,
    tle_epochs: list[pd.Timestamp],
    slr_spans: list[dict],
    orbit_spans: list[dict],
    slr_margin_days: int = SLR_MARGIN_DAYS,
) -> dict:
    """Assess local TLE/SLR/orbit coverage for one window.

    Semantics mirror ``audit_reference_window_alignment``: TLE needs epochs
    bracketing the window; SLR file spans are checked against the window plus
    a margin; orbit product spans must overlap the window itself.
    """
    tle_covered = any(epoch <= start for epoch in tle_epochs) and any(epoch >= end for epoch in tle_epochs)
    margin = pd.Timedelta(days=slr_margin_days)
    slr_covered = any(_span_overlaps(span, start - margin, end + margin) for span in slr_spans)
    orbit_covered = any(_span_overlaps(span, start, end) for span in orbit_spans)
    statuses = {
        "tle_status": "covered" if tle_covered else "missing",
        "slr_status": "covered" if slr_covered else "missing",
        "orbit_status": "covered" if orbit_covered else "missing",
    }
    missing = [source for source in ("tle", "slr", "orbit") if statuses[f"{source}_status"] != "covered"]
    statuses["missing_sources"] = ",".join(missing)
    statuses["aligned"] = not missing
    return statuses


def stable_annotation_id(sat_id: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Deterministic stable-window identifier in the ann_<hash> style."""
    digest = hashlib.md5(f"{sat_id}|{iso_utc(start)}|{iso_utc(end)}".encode("utf-8")).hexdigest()
    return f"stab_{digest[:12]}"


def mine_stable_windows(
    sat_id: str,
    batch: str,
    event_windows: pd.DataFrame,
    tle_epochs: list[pd.Timestamp],
    slr_spans: list[dict],
    orbit_spans: list[dict],
    window_hours: float = WINDOW_HOURS,
    buffer_hours: float = EXCLUSION_BUFFER_HOURS,
    slr_margin_days: int = SLR_MARGIN_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """Mine stable windows for one target; return (windows, summary_row)."""
    coverage = mission_coverage_from_tle_epochs(tle_epochs)
    empty = pd.DataFrame(columns=STABLE_WINDOW_COLUMNS)
    if coverage is None:
        return empty, _summary_row(sat_id, batch, None, None, event_windows, empty, "no_tle_epochs")

    mission_start, mission_end = coverage
    exclusions = build_exclusion_intervals(event_windows, buffer_hours=buffer_hours)
    candidates = candidate_windows(mission_start, mission_end, exclusions, window_hours=window_hours)
    selected = select_windows_per_year(candidates, _yearly_event_counts(event_windows))

    rows = []
    for start, end in selected:
        statuses = assess_window_coverage(
            start, end, tle_epochs, slr_spans, orbit_spans, slr_margin_days=slr_margin_days
        )
        tier = coverage_confidence_tier(
            statuses["tle_status"], statuses["slr_status"], statuses["orbit_status"]
        )
        rows.append(
            {
                "annotation_id": stable_annotation_id(sat_id, start, end),
                "sat_id": sat_id,
                "batch": batch,
                "window_start_utc": iso_utc(start),
                "window_end_utc": iso_utc(end),
                "event_label": "no_event",
                "confidence_tier": tier,
                "annotation_label_source": ANNOTATION_LABEL_SOURCE,
                "tle_status": statuses["tle_status"],
                "slr_status": statuses["slr_status"],
                "orbit_status": statuses["orbit_status"],
                "aligned": statuses["aligned"],
                "missing_sources": statuses["missing_sources"],
                "quality_flags": ",".join(
                    f"missing_{source}" for source in statuses["missing_sources"].split(",") if source
                ),
            }
        )
    windows = pd.DataFrame(rows, columns=STABLE_WINDOW_COLUMNS)
    summary = _summary_row(sat_id, batch, mission_start, mission_end, event_windows, windows, "")
    return windows, summary


def _summary_row(
    sat_id: str,
    batch: str,
    mission_start: pd.Timestamp | None,
    mission_end: pd.Timestamp | None,
    event_windows: pd.DataFrame,
    stable_windows: pd.DataFrame,
    notes: str,
) -> dict:
    event_count = 0 if event_windows is None else int(len(event_windows))
    yearly_quota = _yearly_event_counts(event_windows)
    mission_years = 0
    requested = 0
    if mission_start is not None and mission_end is not None:
        years = list(range(mission_start.year, mission_end.year + 1))
        mission_years = len(years)
        requested = sum(max(1, int(yearly_quota.get(year, 0))) for year in years)
    stable_count = int(len(stable_windows))
    constrained = stable_count < requested
    if constrained and not notes:
        notes = "stable_window_quota_limited_by_event_exclusions"
    tiers = (
        stable_windows["confidence_tier"].value_counts().to_dict() if stable_count else {}
    )
    return {
        "sat_id": sat_id,
        "batch": batch,
        "mission_start_utc": iso_utc(mission_start),
        "mission_end_utc": iso_utc(mission_end),
        "mission_years": mission_years,
        "event_window_count": event_count,
        "requested_stable_window_count": requested,
        "stable_window_count": stable_count,
        "aligned_window_count": int(stable_windows["aligned"].astype(bool).sum()) if stable_count else 0,
        "tle_covered_count": int((stable_windows["tle_status"] == "covered").sum()) if stable_count else 0,
        "slr_covered_count": int((stable_windows["slr_status"] == "covered").sum()) if stable_count else 0,
        "orbit_covered_count": int((stable_windows["orbit_status"] == "covered").sum()) if stable_count else 0,
        "confidence_A_count": int(tiers.get("A", 0)),
        "confidence_B_count": int(tiers.get("B", 0)),
        "confidence_C_count": int(tiers.get("C", 0)),
        "constrained": bool(constrained),
        "notes": notes,
    }


def target_pod_dir(target: dict) -> Path | None:
    """Resolve the POD directory for a config target (audit parity)."""
    explicit = target.get("pod_raw_path")
    if explicit:
        return Path(str(explicit))
    for key in ("tle_raw_path", "precise_orbit_raw_path", "slr_raw_path"):
        value = target.get(key)
        if value:
            return Path(str(value)).parent / "pod"
    return None
