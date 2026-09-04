"""Audit local Batch A evidence-source acquisition status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from downloaders.download_batch_tle_history import (
    DEFAULT_CONFIG,
    DEFAULT_METADATA_ROOT,
    _resolve_metadata_root,
    iter_batch_a_targets,
    load_annotation_config,
)


OUTPUT_COLUMNS = [
    "sat_id",
    "batch",
    "norad",
    "ids_annotation_status",
    "ids_annotation_file_count",
    "normalized_annotation_status",
    "normalized_event_window_status",
    "tle_status",
    "tle_file_count",
    "tle_metadata_status",
    "pod_status",
    "pod_file_count",
    "precise_orbit_status",
    "precise_orbit_file_count",
    "slr_status",
    "slr_file_count",
    "source_readiness",
    "limitation",
]


def _path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return resolve_repo_path(str(value))


def _file_exists(path: Path | None) -> bool:
    return bool(path and path.exists() and path.is_file() and path.stat().st_size > 0)


def _count_files(path: Path | None) -> int:
    if path is None or not path.exists() or not path.is_dir():
        return 0
    return sum(
        1
        for item in path.iterdir()
        if item.is_file()
        and item.stat().st_size > 0
        and not item.name.startswith(".")
        and ".part" not in item.name
    )


def _target_pod_path(target: dict) -> Path | None:
    if target.get("pod_raw_path"):
        return _path(target.get("pod_raw_path"))
    for source_key in ("tle_raw_path", "precise_orbit_raw_path", "slr_raw_path"):
        source_path = _path(target.get(source_key))
        if source_path is not None:
            return source_path.parent / "pod"
    return _path(f"data/raw/reference_validation/{target['sat_id']}/pod")


def _metadata_status(metadata_root: str | Path, sat_id: str, source: str) -> str:
    metadata_path = _resolve_metadata_root(metadata_root) / sat_id / source / "download_metadata.json"
    if not metadata_path.exists():
        return "missing_metadata"
    try:
        payload: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid_metadata"
    return str(payload.get("status", "unknown"))


def _status_from_count(count: int) -> str:
    return "available" if count > 0 else "missing"


def build_source_status(target: dict, metadata_root: str | Path = DEFAULT_METADATA_ROOT) -> dict:
    """Return one stable audit row for a Batch A target."""
    sat_id = target["sat_id"]
    annotation_path = _path(target.get("raw_annotation_path"))
    normalized_annotation_path = _path(target.get("normalized_annotation_path"))
    event_window_path = normalized_annotation_path.parent / "event_windows.csv" if normalized_annotation_path else None

    tle_count = _count_files(_path(target.get("tle_raw_path")))
    pod_count = _count_files(_target_pod_path(target))
    precise_orbit_count = _count_files(_path(target.get("precise_orbit_raw_path")))
    slr_count = _count_files(_path(target.get("slr_raw_path")))
    tle_metadata_status = _metadata_status(metadata_root, sat_id, "tle")
    tle_status = "downloaded" if tle_metadata_status == "downloaded" and tle_count > 0 else _status_from_count(tle_count)

    missing = []
    if not _file_exists(annotation_path):
        missing.append("ids_annotations")
    if not _file_exists(normalized_annotation_path):
        missing.append("normalized_annotations")
    if not _file_exists(event_window_path):
        missing.append("event_windows")
    if tle_count == 0:
        missing.append("tle")
    if pod_count == 0:
        missing.append("pod")
    if precise_orbit_count == 0:
        missing.append("precise_orbit")
    if slr_count == 0:
        missing.append("slr")

    return {
        "sat_id": sat_id,
        "batch": target.get("batch", ""),
        "norad": str(target.get("norad", "")),
        "ids_annotation_status": "available" if _file_exists(annotation_path) else "missing",
        "ids_annotation_file_count": int(_file_exists(annotation_path)),
        "normalized_annotation_status": "available" if _file_exists(normalized_annotation_path) else "missing",
        "normalized_event_window_status": "available" if _file_exists(event_window_path) else "missing",
        "tle_status": tle_status,
        "tle_file_count": tle_count,
        "tle_metadata_status": tle_metadata_status,
        "pod_status": _status_from_count(pod_count),
        "pod_file_count": pod_count,
        "precise_orbit_status": _status_from_count(precise_orbit_count),
        "precise_orbit_file_count": precise_orbit_count,
        "slr_status": _status_from_count(slr_count),
        "slr_file_count": slr_count,
        "source_readiness": "complete" if not missing else "partial",
        "limitation": ";".join(missing),
    }


def audit_batch_a_sources(config: dict, metadata_root: str | Path = DEFAULT_METADATA_ROOT) -> pd.DataFrame:
    """Build a Batch A evidence-source audit table."""
    return audit_sources_by_batch(config, batches=["A"], metadata_root=metadata_root)


def audit_sources_by_batch(
    config: dict,
    batches: list[str],
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
) -> pd.DataFrame:
    """Build an evidence-source audit table for selected target batches."""
    batch_set = set(batches)
    rows = [
        build_source_status(target, metadata_root=metadata_root)
        for target in config.get("targets", [])
        if target.get("batch") in batch_set
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_source_audit(audit: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the source audit as a tracked decision CSV."""
    destination = Path(output_path)
    ensure_directory(destination.parent)
    audit.to_csv(destination, index=False)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Batch A local evidence-source acquisition status")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to maneuver annotation target config")
    parser.add_argument("--metadata-root", default=str(DEFAULT_METADATA_ROOT), help="Ignored download metadata root")
    parser.add_argument("--batch", action="append", default=None, help="Batch letter to include; repeatable")
    parser.add_argument(
        "--output",
        default="results/tables/batch_a_source_acquisition_status.csv",
        help="Tracked source acquisition status CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_annotation_config(args.config)
    audit = audit_sources_by_batch(config, batches=args.batch or ["A"], metadata_root=args.metadata_root)
    output = write_source_audit(audit, REPO_ROOT / args.output)
    print(f"wrote {output} rows={len(audit)}")


if __name__ == "__main__":
    main()
