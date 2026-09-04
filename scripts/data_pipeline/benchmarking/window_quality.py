"""Per-window quality checks for benchmark label gates."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


DEFAULT_MAX_GAP_HOURS_BY_SOURCE = {
    "pod": 6.0,
    "tle": 48.0,
    "slr": 72.0,
}


def evaluate_window_quality(
    source_tables: Dict[str, pd.DataFrame] | None,
    start: object,
    end: object,
    required_sources: Iterable[str],
    release_start: object | None = None,
    release_end: object | None = None,
    max_gap_hours_by_source: Dict[str, float] | None = None,
    boundary_margin_hours: float = 0,
) -> Dict:
    """Evaluate major source gaps and release-boundary artifacts for one window."""
    flags: list[str] = []
    reasons: list[str] = []
    source_status: Dict[str, str] = {}
    window_start = pd.to_datetime(start, utc=True)
    window_end = pd.to_datetime(end, utc=True)

    boundary_margin = pd.Timedelta(hours=float(boundary_margin_hours))
    if boundary_margin > pd.Timedelta(0):
        if release_start is not None and window_start < pd.to_datetime(release_start, utc=True) + boundary_margin:
            flags.append("boundary_flag")
            reasons.append("boundary_artifact")
        if release_end is not None and window_end > pd.to_datetime(release_end, utc=True) - boundary_margin:
            flags.append("boundary_flag")
            reasons.append("boundary_artifact")

    gap_thresholds = dict(DEFAULT_MAX_GAP_HOURS_BY_SOURCE)
    gap_thresholds.update(max_gap_hours_by_source or {})
    if source_tables:
        for source in required_sources:
            df = source_tables.get(source)
            threshold_hours = gap_thresholds.get(source)
            if threshold_hours is None:
                continue
            has_gap = _window_crosses_major_gap(df, window_start, window_end, float(threshold_hours))
            source_status[source] = "gap" if has_gap else "pass"
            if has_gap:
                flags.append("gap_flag")
                reasons.append("major_gap")

    flags = sorted(set(flags))
    reasons = sorted(set(reasons), key=reasons.index)
    return {
        "ok": not flags,
        "flags": flags,
        "reason": reasons[0] if reasons else "",
        "source_status": source_status,
    }


def _window_crosses_major_gap(
    df: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    threshold_hours: float,
) -> bool:
    if df is None or len(df) == 0 or "epoch" not in df.columns:
        return False

    cache_key = f"_major_gaps_{float(threshold_hours)}"
    if cache_key not in df.attrs:
        epochs = pd.to_datetime(df["epoch"], errors="coerce", utc=True).dropna().sort_values().drop_duplicates()
        threshold = pd.Timedelta(hours=threshold_hours)
        gaps = []
        previous = None
        for current in epochs:
            if previous is not None and current - previous > threshold:
                gaps.append((previous, current))
            previous = current
        df.attrs[cache_key] = gaps

    return any(current > start and previous < end for previous, current in df.attrs[cache_key])
