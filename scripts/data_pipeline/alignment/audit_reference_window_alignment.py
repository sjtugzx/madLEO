"""Audit event-window alignment across local TLE, SLR, and orbit products."""

from __future__ import annotations

import argparse
import calendar
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.normalization import parse_tle_line_pairs
from downloaders.download_batch_tle_history import DEFAULT_CONFIG, load_annotation_config


OUTPUT_COLUMNS = [
    "sat_id",
    "batch",
    "annotation_id",
    "window_start_utc",
    "window_end_utc",
    "scope",
    "as_of_utc",
    "mission_tail_cutoff_utc",
    "tle_status",
    "tle_epoch_count",
    "tle_nearest_before_utc",
    "tle_nearest_after_utc",
    "slr_status",
    "slr_file_count",
    "slr_files",
    "orbit_status",
    "orbit_file_count",
    "orbit_source_types",
    "orbit_files",
    "aligned",
    "missing_sources",
]


def _parse_utc(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, format="ISO8601")


def _iso(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    ts = pd.to_datetime(value, utc=True, format="ISO8601")
    if ts.microsecond:
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    return resolve_repo_path(str(value))


def _target_pod_path(target: dict) -> Path:
    explicit = target.get("pod_raw_path")
    if explicit:
        return resolve_repo_path(str(explicit))
    for key in ("tle_raw_path", "precise_orbit_raw_path", "slr_raw_path"):
        path = _resolve_path(target.get(key))
        if path is not None:
            return path.parent / "pod"
    return REPO_ROOT / "data" / "raw" / "reference_validation" / target["sat_id"] / "pod"


def _event_windows_path(target: dict) -> Path:
    normalized = _resolve_path(target.get("normalized_annotation_path"))
    if normalized is not None:
        return normalized.parent / "event_windows.csv"
    return REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation" / target["sat_id"] / "event_windows.csv"


def _has_local_event_windows(target: dict) -> bool:
    return _event_windows_path(target).exists()


def _is_complete_local_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0 and not path.name.startswith(".") and ".part" not in path.name


def select_mission_tail_windows(
    event_windows: pd.DataFrame,
    days: int = 365,
    as_of_utc: str | None = None,
) -> pd.DataFrame:
    """Select windows in the final per-target mission/event year."""
    if event_windows.empty:
        return event_windows.copy()
    result = event_windows.copy()
    result["window_end_utc"] = pd.to_datetime(result["window_end_utc"], utc=True, format="ISO8601")
    if as_of_utc:
        result = result[result["window_end_utc"] <= _parse_utc(as_of_utc)].copy()
        if result.empty:
            return result
    last_end = result["window_end_utc"].max()
    cutoff = last_end - pd.Timedelta(days=days)
    result = result[result["window_end_utc"] >= cutoff].copy()
    result["mission_tail_cutoff_utc"] = _iso(cutoff)
    result["window_end_utc"] = result["window_end_utc"].map(_iso)
    result["window_start_utc"] = pd.to_datetime(result["window_start_utc"], utc=True, format="ISO8601").map(_iso)
    return result.reset_index(drop=True)


def _iter_nonempty_files(paths: Iterable[Path | None]) -> Iterable[Path]:
    for path in paths:
        if path is None or not path.exists():
            continue
        if _is_complete_local_file(path):
            yield path
            continue
        if path.is_dir():
            for item in path.iterdir():
                if _is_complete_local_file(item):
                    yield item


def infer_tle_epochs(tle_dir: str | Path | None) -> list[pd.Timestamp]:
    """Infer local TLE epochs from raw TLE line-pair records.

    Delegates to the canonical parser
    ``benchmarking.normalization.parse_tle_line_pairs`` (checksum + NORAD
    cross-check, counted rejections; REVIEW_FINDINGS 6.3.1): a corrupt or
    unpaired line can no longer contribute a phantom epoch.  Returned
    timestamps are tz-aware UTC, matching the audit path's comparisons.
    """
    path = _resolve_path(tle_dir)
    epochs: list[pd.Timestamp] = []
    for file_path in _iter_nonempty_files([path]):
        if file_path.suffix.lower() not in {".tle", ".txt"}:
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        frame = parse_tle_line_pairs(lines)
        if frame.empty:
            continue
        epochs.extend(pd.to_datetime(frame["epoch"], utc=True).tolist())
    return sorted(set(epochs))


def _parse_compact_datetime(date_text: str, time_text: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S"), tz="UTC")
    except ValueError:
        return None


def _parse_year_doy(year_doy: str) -> pd.Timestamp | None:
    try:
        year = int(year_doy[:4])
        doy = int(year_doy[4:7])
        hour = int(year_doy[7:9] or "0")
        minute = int(year_doy[9:11] or "0")
    except ValueError:
        return None
    return pd.Timestamp(datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, hours=hour, minutes=minute))


def _parse_sp3_year_doy(two_digit_year: str, doy: str) -> pd.Timestamp | None:
    try:
        year_fragment = int(two_digit_year)
        day = int(doy)
    except ValueError:
        return None
    year = 2000 + year_fragment if year_fragment < 57 else 1900 + year_fragment
    return pd.Timestamp(datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1))


def _orbit_span_from_name(path: Path, source_type: str) -> dict | None:
    name = path.name

    sentinel = re.search(r"_V(\d{8})T(\d{6})_(\d{8})T(\d{6})_", name)
    if sentinel:
        start = _parse_compact_datetime(sentinel.group(1), sentinel.group(2))
        end = _parse_compact_datetime(sentinel.group(3), sentinel.group(4))
        if start is not None and end is not None:
            return _span_row(path, source_type, start, end)

    timed_pairs = re.findall(r"_(\d{8})_(\d{6})", name)
    if len(timed_pairs) >= 2:
        start_pair, end_pair = timed_pairs[-2], timed_pairs[-1]
        start = _parse_compact_datetime(*start_pair)
        end = _parse_compact_datetime(*end_pair)
        if start is not None and end is not None:
            return _span_row(path, source_type, start, end)

    rinex = re.search(r"_(\d{11})_(\d{2})D_", name)
    if rinex:
        start = _parse_year_doy(rinex.group(1))
        if start is not None:
            end = start + pd.Timedelta(days=int(rinex.group(2)))
            return _span_row(path, "gnss_pod_rinex_daily", start, end)

    sp3 = re.search(r"\.b(\d{2})(\d{3})\.e(\d{2})(\d{3})\.", name)
    if sp3:
        start = _parse_sp3_year_doy(sp3.group(1), sp3.group(2))
        end = _parse_sp3_year_doy(sp3.group(3), sp3.group(4))
        if start is not None and end is not None:
            return _span_row(path, source_type, start, end + pd.Timedelta(days=1))

    return None


def _span_row(path: Path, source_type: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    return {
        "path": str(path),
        "source_type": source_type,
        "start_utc": _iso(start),
        "end_utc": _iso(end),
    }


def infer_orbit_product_spans(paths: Iterable[str | Path | None]) -> list[dict]:
    """Infer local POD/precise-orbit product spans from supported filenames."""
    rows: list[dict] = []
    for base_path in paths:
        path = _resolve_path(base_path)
        if path is None:
            continue
        source_type = "pod" if path.name == "pod" else "precise_orbit"
        for file_path in _iter_nonempty_files([path]):
            span = _orbit_span_from_name(file_path, source_type=source_type)
            if span is not None:
                rows.append(span)
    return sorted(rows, key=lambda row: (row["start_utc"], row["path"]))


def _slr_file_span(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    daily = re.search(r"[_.](\d{8})(?:\D|$)", path.name)
    if daily:
        try:
            start = pd.Timestamp(datetime.strptime(daily.group(1), "%Y%m%d"), tz="UTC")
        except ValueError:
            return None
        return start, start + pd.Timedelta(days=1)

    monthly = re.search(r"[_.](\d{6})(?:\D|$)", path.name)
    if monthly:
        try:
            year = int(monthly.group(1)[:4])
            month = int(monthly.group(1)[4:6])
        except ValueError:
            return None
        days = calendar.monthrange(year, month)[1]
        start = pd.Timestamp(datetime(year, month, 1, tzinfo=timezone.utc))
        return start, start + pd.Timedelta(days=days)
    return None


def infer_slr_file_spans(slr_dir: str | Path | None) -> list[dict]:
    """Infer local SLR observation file spans from dated ILRS filenames."""
    path = _resolve_path(slr_dir)
    rows: list[dict] = []
    for file_path in _iter_nonempty_files([path]):
        span = _slr_file_span(file_path)
        if span is None:
            continue
        rows.append({"path": str(file_path), "start_utc": _iso(span[0]), "end_utc": _iso(span[1])})
    return sorted(rows, key=lambda row: (row["start_utc"], row["path"]))


def _overlaps(start: pd.Timestamp, end: pd.Timestamp, row: dict) -> bool:
    row_start = _parse_utc(row["start_utc"])
    row_end = _parse_utc(row["end_utc"])
    return bool(row_start < end and row_end > start)


def _tle_coverage(epochs: list[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    before = [epoch for epoch in epochs if epoch <= start]
    after = [epoch for epoch in epochs if epoch >= end]
    return {
        "status": "covered" if before and after else "missing",
        "epoch_count": len(epochs),
        "nearest_before_utc": _iso(before[-1]) if before else "",
        "nearest_after_utc": _iso(after[0]) if after else "",
    }


def _slr_coverage(spans: list[dict], start: pd.Timestamp, end: pd.Timestamp, margin_days: int) -> dict:
    audit_start = start - pd.Timedelta(days=margin_days)
    audit_end = end + pd.Timedelta(days=margin_days)
    matches = [row for row in spans if _overlaps(audit_start, audit_end, row)]
    return {
        "status": "covered" if matches else "missing",
        "file_count": len(matches),
        "files": ";".join(Path(row["path"]).name for row in matches),
    }


def _orbit_coverage(spans: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    matches = [row for row in spans if _overlaps(start, end, row)]
    return {
        "status": "covered" if matches else "missing",
        "file_count": len(matches),
        "source_types": ",".join(sorted({row["source_type"] for row in matches})),
        "files": ";".join(Path(row["path"]).name for row in matches),
    }


def _target_event_windows(target: dict, scope: str, days: int, as_of_utc: str | None = None) -> pd.DataFrame:
    path = _event_windows_path(target)
    if not path.exists():
        return pd.DataFrame()
    windows = pd.read_csv(path)
    if scope == "recent":
        return select_mission_tail_windows(windows, days=days, as_of_utc=as_of_utc)
    windows = windows.copy()
    if as_of_utc:
        windows = windows[pd.to_datetime(windows["window_end_utc"], utc=True, format="ISO8601") <= _parse_utc(as_of_utc)].copy()
    windows["mission_tail_cutoff_utc"] = ""
    windows["window_start_utc"] = pd.to_datetime(windows["window_start_utc"], utc=True, format="ISO8601").map(_iso)
    windows["window_end_utc"] = pd.to_datetime(windows["window_end_utc"], utc=True, format="ISO8601").map(_iso)
    return windows.reset_index(drop=True)


def audit_reference_window_alignment(
    config: dict,
    batches: list[str],
    scope: str = "recent",
    days: int = 365,
    slr_margin_days: int = 1,
    as_of_utc: str | None = None,
    local_only: bool = False,
) -> pd.DataFrame:
    """Audit local source coverage for reference maneuver event windows."""
    if scope not in {"recent", "all"}:
        raise ValueError("scope must be 'recent' or 'all'")

    rows = []
    batch_set = set(batches)
    for target in config.get("targets", []):
        if target.get("batch") not in batch_set:
            continue
        if local_only and not _has_local_event_windows(target):
            continue
        windows = _target_event_windows(target, scope=scope, days=days, as_of_utc=as_of_utc)
        tle_epochs = infer_tle_epochs(target.get("tle_raw_path"))
        slr_spans = infer_slr_file_spans(target.get("slr_raw_path"))
        orbit_spans = infer_orbit_product_spans([_target_pod_path(target), target.get("precise_orbit_raw_path")])
        for event in windows.itertuples(index=False):
            start = _parse_utc(event.window_start_utc)
            end = _parse_utc(event.window_end_utc)
            tle = _tle_coverage(tle_epochs, start, end)
            slr = _slr_coverage(slr_spans, start, end, margin_days=slr_margin_days)
            orbit = _orbit_coverage(orbit_spans, start, end)
            missing = [
                source
                for source, status in [
                    ("tle", tle["status"]),
                    ("slr", slr["status"]),
                    ("orbit", orbit["status"]),
                ]
                if status != "covered"
            ]
            rows.append(
                {
                    "sat_id": target["sat_id"],
                    "batch": target.get("batch", ""),
                    "annotation_id": getattr(event, "annotation_id", ""),
                    "window_start_utc": event.window_start_utc,
                    "window_end_utc": event.window_end_utc,
                    "scope": scope,
                    "as_of_utc": _iso(as_of_utc) if as_of_utc else "",
                    "mission_tail_cutoff_utc": getattr(event, "mission_tail_cutoff_utc", ""),
                    "tle_status": tle["status"],
                    "tle_epoch_count": tle["epoch_count"],
                    "tle_nearest_before_utc": tle["nearest_before_utc"],
                    "tle_nearest_after_utc": tle["nearest_after_utc"],
                    "slr_status": slr["status"],
                    "slr_file_count": slr["file_count"],
                    "slr_files": slr["files"],
                    "orbit_status": orbit["status"],
                    "orbit_file_count": orbit["file_count"],
                    "orbit_source_types": orbit["source_types"],
                    "orbit_files": orbit["files"],
                    "aligned": not missing,
                    "missing_sources": ",".join(missing),
                }
            )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if "aligned" in result.columns:
        result["aligned"] = result["aligned"].astype(object)
    return result


def write_alignment_audit(audit: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = REPO_ROOT / destination
    ensure_directory(destination.parent)
    audit.to_csv(destination, index=False)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local reference maneuver window source alignment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--batch", action="append", default=None, help="Planning batch to include (repeatable); default audits both A and B")
    parser.add_argument("--scope", choices=["recent", "all"], default="recent")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--slr-margin-days", type=int, default=1)
    parser.add_argument("--as-of-utc", default=None, help="Exclude event windows ending after this UTC timestamp")
    parser.add_argument("--local-only", action="store_true", help="Skip configured targets without local event windows")
    parser.add_argument("--output", default="results/tables/reference_recent_window_alignment_status.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_annotation_config(args.config)
    audit = audit_reference_window_alignment(
        config,
        batches=args.batch or ["A", "B"],
        scope=args.scope,
        days=args.days,
        slr_margin_days=args.slr_margin_days,
        as_of_utc=args.as_of_utc,
        local_only=args.local_only,
    )
    output = write_alignment_audit(audit, args.output)
    aligned_count = int(audit["aligned"].sum()) if not audit.empty else 0
    print(json.dumps({"output": str(output), "rows": len(audit), "aligned": aligned_count}, sort_keys=True))


if __name__ == "__main__":
    main()
