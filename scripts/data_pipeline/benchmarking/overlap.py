"""Overlap discovery helpers for validated benchmark sources."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


def summarize_overlap(
    target_sources: Dict[str, pd.DataFrame],
    required_sources: Iterable[str] | None = None,
    min_overlap_seconds: float = 0.0,
) -> Dict:
    """Summarize source coverage and shared overlap for one target."""
    source_ranges = {}
    overlap_start = None
    overlap_end = None
    required_sources = list(required_sources or target_sources.keys())

    for source_name, df in target_sources.items():
        if df is None or len(df) == 0 or "epoch" not in df.columns:
            source_ranges[source_name] = {
                "row_count": 0,
                "time_start": None,
                "time_end": None,
            }
            continue

        epochs = pd.to_datetime(df["epoch"], errors="coerce").dropna()
        if len(epochs) == 0:
            source_ranges[source_name] = {
                "row_count": int(len(df)),
                "time_start": None,
                "time_end": None,
            }
            continue

        source_start = epochs.min()
        source_end = epochs.max()
        source_ranges[source_name] = {
            "row_count": int(len(df)),
            "time_start": source_start,
            "time_end": source_end,
        }

        overlap_start = source_start if overlap_start is None else max(overlap_start, source_start)
        overlap_end = source_end if overlap_end is None else min(overlap_end, source_end)

    overlap_seconds = 0.0
    if overlap_start is not None and overlap_end is not None and overlap_end >= overlap_start:
        overlap_seconds = float((overlap_end - overlap_start).total_seconds())

    required_sources_present = True
    for source_name in required_sources:
        summary = source_ranges.get(source_name, {})
        if summary.get("row_count", 0) <= 0 or summary.get("time_start") is None or summary.get("time_end") is None:
            required_sources_present = False
            break

    return {
        "sources": source_ranges,
        "required_sources": required_sources,
        "min_overlap_seconds": float(min_overlap_seconds),
        "required_sources_present": required_sources_present,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "overlap_seconds": overlap_seconds,
        "validated_ready": required_sources_present and overlap_seconds >= float(min_overlap_seconds),
    }
