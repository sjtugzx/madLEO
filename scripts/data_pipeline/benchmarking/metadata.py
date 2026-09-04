"""Helpers for persisting download and build metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory
from benchmarking.overlap import summarize_overlap


DEFAULT_METADATA_ROOT = REPO_ROOT / "data" / "metadata" / "benchmark"
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "interim" / "cache" / "benchmark"


def json_safe(value: Any) -> Any:
    """Convert common Python and pandas objects into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value


def metadata_root(path: str | None = None) -> Path:
    """Return the root directory for benchmark metadata."""
    if path is None:
        return DEFAULT_METADATA_ROOT
    root = Path(path)
    if root.is_absolute():
        return root
    return REPO_ROOT / root


def target_metadata_dir(root: Path, sat_id: str, source_name: str | None = None) -> Path:
    """Return a metadata directory for a target or source."""
    path = ensure_directory(root / sat_id)
    if source_name is not None:
        path = ensure_directory(path / source_name)
    return path


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload with UTF-8 encoding."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    """Write line-delimited JSON records."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False))
            handle.write("\n")


def local_file_records(paths: Iterable[Path]) -> List[Dict]:
    """Return metadata for downloaded files."""
    records: List[Dict] = []
    for path in paths:
        if not path.exists():
            continue
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            }
        )
    return records


def summarize_listing(items: Iterable[Dict]) -> Dict[str, Any]:
    """Summarize a provider listing response."""
    rows = list(items)
    summary: Dict[str, Any] = {"count": len(rows)}
    if not rows:
        return summary

    first = rows[0]
    keys = sorted({key for row in rows for key in row.keys()})
    summary["keys"] = keys
    for start_key, end_key in [("validity_start", "validity_stop"), ("start_time", "end_time")]:
        starts = [row.get(start_key) for row in rows if row.get(start_key) is not None]
        ends = [row.get(end_key) for row in rows if row.get(end_key) is not None]
        if starts:
            summary["listing_start"] = min(starts)
        if ends:
            summary["listing_end"] = max(ends)
    if "filename" in first:
        summary["sample_filenames"] = [row["filename"] for row in rows[:5] if row.get("filename")]
    elif "name" in first:
        summary["sample_filenames"] = [row["name"] for row in rows[:5] if row.get("name")]
    return summary


def summarize_dataframe(df: pd.DataFrame, epoch_col: str = "epoch") -> Dict[str, Any]:
    """Summarize a normalized source table."""
    if df is None or len(df) == 0:
        return {"row_count": 0}
    summary: Dict[str, Any] = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
    }
    if epoch_col in df.columns:
        epochs = pd.to_datetime(df[epoch_col], errors="coerce").dropna()
        if len(epochs) > 0:
            summary["time_start"] = epochs.min()
            summary["time_end"] = epochs.max()
    if "sat_id" in df.columns:
        summary["sat_ids"] = sorted(df["sat_id"].dropna().astype(str).unique().tolist())[:20]
    return summary


def _parse_summary_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, "", "NaT"):
        return pd.NaT
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert(None)


def _ordered_union(left: Iterable[Any], right: Iterable[Any]) -> List[str]:
    values: List[str] = []
    seen = set()
    for value in list(left or []) + list(right or []):
        text = str(value)
        if text in seen:
            continue
        values.append(text)
        seen.add(text)
    return values


def summarize_parquet_cache_metadata(
    cache_root: Path | str | None,
    source_name: str,
    sat_id: str,
    epoch_col: str = "epoch",
) -> Dict[str, Any] | None:
    """Summarize cached normalized parquet files without loading full tables."""
    if cache_root is None:
        return None
    cache_dir = Path(cache_root) / source_name / sat_id
    if not cache_dir.exists():
        return None

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    row_count = 0
    columns: List[str] = []
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    for path in sorted(cache_dir.glob("*.parquet")):
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception:
            continue
        row_count += int(parquet_file.metadata.num_rows)
        columns = _ordered_union(columns, parquet_file.schema_arrow.names)
        if epoch_col not in parquet_file.schema_arrow.names:
            continue
        epoch_index = parquet_file.schema_arrow.names.index(epoch_col)
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            statistics = parquet_file.metadata.row_group(row_group_index).column(epoch_index).statistics
            if statistics is None or not statistics.has_min_max:
                continue
            start = _parse_summary_timestamp(statistics.min)
            end = _parse_summary_timestamp(statistics.max)
            if not pd.isna(start):
                starts.append(start)
            if not pd.isna(end):
                ends.append(end)

    if row_count <= 0 or not starts or not ends:
        return None
    return {
        "row_count": row_count,
        "columns": columns,
        "time_start": min(starts),
        "time_end": max(ends),
    }


def _cache_summary_is_fresher(cache_summary: Dict[str, Any], existing_summary: Dict[str, Any]) -> bool:
    existing_start = _parse_summary_timestamp(existing_summary.get("time_start"))
    existing_end = _parse_summary_timestamp(existing_summary.get("time_end"))
    cache_start = _parse_summary_timestamp(cache_summary.get("time_start"))
    cache_end = _parse_summary_timestamp(cache_summary.get("time_end"))
    if pd.isna(cache_start) or pd.isna(cache_end):
        return False
    if pd.isna(existing_start) or pd.isna(existing_end):
        return True

    existing_rows = int(existing_summary.get("row_count") or 0)
    cache_rows = int(cache_summary.get("row_count") or 0)
    existing_span_seconds = max(float((existing_end - existing_start).total_seconds()), 1.0)
    cache_end_extension_seconds = float((cache_end - existing_end).total_seconds())
    expands_rows = cache_rows > existing_rows
    extends_materially = cache_end_extension_seconds > max(7 * 86400.0, existing_span_seconds * 0.1)
    return expands_rows and cache_start <= existing_start and extends_materially


def refresh_source_summaries_from_cache(
    source_summaries: Dict[str, Any],
    sat_id: str,
    cache_root: Path | str | None = DEFAULT_CACHE_ROOT,
) -> tuple[Dict[str, Any], List[str]]:
    """Refresh stale source summaries from local parquet cache metadata when available."""
    refreshed = dict(source_summaries)
    refreshed_sources: List[str] = []
    for source_name, existing_summary in source_summaries.items():
        cache_summary = summarize_parquet_cache_metadata(cache_root, source_name, sat_id)
        if not cache_summary or not _cache_summary_is_fresher(cache_summary, existing_summary):
            continue
        updated = dict(existing_summary)
        updated.update(cache_summary)
        updated["columns"] = _ordered_union(existing_summary.get("columns", []), cache_summary.get("columns", []))
        updated["summary_refresh_source"] = "parquet_cache_metadata"
        refreshed[source_name] = updated
        refreshed_sources.append(source_name)
    return refreshed, refreshed_sources


def summarize_overlap_from_source_summaries(
    source_summaries: Dict[str, Any],
    required_sources: Iterable[str] | None = None,
    min_overlap_seconds: float = 0.0,
) -> Dict:
    """Summarize overlap from lightweight source-summary metadata."""
    target_sources = {}
    for source_name, summary in source_summaries.items():
        start = _parse_summary_timestamp(summary.get("time_start"))
        end = _parse_summary_timestamp(summary.get("time_end"))
        if pd.isna(start) or pd.isna(end):
            target_sources[source_name] = pd.DataFrame()
            continue
        target_sources[source_name] = pd.DataFrame(
            {"epoch": [start, end] if start != end else [start]}
        )
        target_sources[source_name].attrs["summary_row_count"] = int(summary.get("row_count") or 0)
    overlap = summarize_overlap(
        target_sources,
        required_sources=required_sources,
        min_overlap_seconds=min_overlap_seconds,
    )
    for source_name, summary in source_summaries.items():
        if source_name in overlap["sources"]:
            overlap["sources"][source_name]["row_count"] = int(summary.get("row_count") or 0)
    return overlap


def build_source_download_metadata(
    sat_id: str,
    source_name: str,
    provider: str,
    date_range: Dict[str, str],
    source_config: Dict,
    listing: Iterable[Dict],
    downloaded_files: Iterable[Path],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a structured metadata payload for a downloaded source."""
    downloaded_files = list(downloaded_files)
    payload: Dict[str, Any] = {
        "sat_id": sat_id,
        "source": source_name,
        "provider": provider,
        "date_range": date_range,
        "source_config": source_config,
        "listing_summary": summarize_listing(listing),
        "downloaded_files": local_file_records(downloaded_files),
        "download_count": len(downloaded_files),
    }
    if extra:
        payload["extra"] = extra
    return payload
