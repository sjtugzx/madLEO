"""Probe reference auxiliary evidence sources without promoting claims."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from downloaders.ilrs_downloader import ILRS_TARGETS, list_cddis_cmr_granules


DEFAULT_CONFIG = REPO_ROOT / "configs" / "targets_maneuver_annotations.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "tables" / "reference_auxiliary_source_probe_status.csv"

PROBE_STATUS_COLUMNS = [
    "sat_id",
    "display_name",
    "batch",
    "source",
    "provider",
    "target_mapping_status",
    "provider_target_name",
    "norad",
    "cospar",
    "sic",
    "probe_start_utc",
    "probe_end_utc",
    "probe_status",
    "candidate_file_count",
    "local_file_count",
    "claim_allowed",
    "source_url_or_provider",
    "required_credentials",
    "next_action",
    "notes",
]


def load_annotation_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the maneuver annotation target config."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def _count_files(path_value: str | Path | None) -> int:
    if path_value is None:
        return 0
    path = resolve_repo_path(str(path_value))
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


def _source_path(target: dict[str, Any], source: str) -> str:
    if source == "slr":
        return str(target.get("slr_raw_path") or f"data/raw/reference_validation/{target['sat_id']}/slr")
    if source == "precise_orbit":
        return str(target.get("precise_orbit_raw_path") or f"data/raw/reference_validation/{target['sat_id']}/precise_orbit")
    if source == "pod":
        return str(target.get("pod_raw_path") or f"data/raw/reference_validation/{target['sat_id']}/pod")
    raise ValueError(f"Unsupported source: {source}")


def _to_utc_iso(value: datetime) -> str:
    value = value.astimezone(timezone.utc)
    if value.microsecond:
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_probe_window() -> tuple[str, str]:
    # A narrow modern probe window is enough to prove archive naming/connectivity
    # without crawling large archives. Historical targets still need per-mission
    # windows before download.
    end = datetime(2023, 1, 10)
    start = end - timedelta(days=9)
    return f"{start:%Y-%m-%dT%H:%M:%SZ}", f"{end:%Y-%m-%dT%H:%M:%SZ}"


def compute_probe_window(target: dict[str, Any], margin_days: int = 1) -> tuple[str, str]:
    """Return a narrow probe window from local event windows when available."""
    normalized_path = target.get("normalized_annotation_path")
    if normalized_path:
        event_path = Path(str(normalized_path)).parent / "event_windows.csv"
        if not event_path.is_absolute():
            event_path = REPO_ROOT / event_path
        if event_path.exists():
            try:
                events = pd.read_csv(event_path)
                starts = pd.to_datetime(events["window_start_utc"], utc=True, errors="coerce").dropna()
                ends = pd.to_datetime(events["window_end_utc"], utc=True, errors="coerce").dropna()
            except (KeyError, pd.errors.EmptyDataError, OSError):
                starts = pd.Series(dtype="datetime64[ns, UTC]")
                ends = pd.Series(dtype="datetime64[ns, UTC]")
            if not starts.empty and not ends.empty:
                start = starts.min().to_pydatetime() - timedelta(days=margin_days)
                end = ends.max().to_pydatetime() + timedelta(days=margin_days)
                return _to_utc_iso(start), _to_utc_iso(end)
    return _default_probe_window()


def _tracking_has_gnss_or_gps(target: dict[str, Any]) -> bool:
    tracking = {str(value).upper() for value in target.get("tracking_systems", [])}
    return bool({"GNSS", "GPS"}.intersection(tracking))


def _base_row(target: dict[str, Any], source: str) -> dict[str, Any]:
    sat_id = target["sat_id"]
    start, end = compute_probe_window(target)
    local_count = _count_files(_source_path(target, source))
    mapping = ILRS_TARGETS.get(sat_id, {}) if source == "slr" else {}
    if source == "slr":
        provider = "ILRS/CDDIS"
        required_credentials = "EARTHDATA_USERNAME;EARTHDATA_PASSWORD"
        source_url_or_provider = "https://cddis.nasa.gov/archive/slr/data/"
        mapping_status = "mapped" if mapping else "mapping_missing"
        probe_status = "planned" if mapping else "provider_mapping_missing"
        next_action = "execute_official_archive_probe" if mapping else "verify_official_target_mapping"
    else:
        provider = "source_discovery_required"
        required_credentials = ""
        source_url_or_provider = target.get("maneuver_history_url", "")
        mapping_status = "unmapped"
        if _tracking_has_gnss_or_gps(target):
            probe_status = "source_discovery_required"
            next_action = "discover_official_precise_orbit_source"
        else:
            probe_status = "not_expected_from_ids_tracking"
            next_action = "record_source_limitation"

    return {
        "sat_id": sat_id,
        "display_name": target.get("display_name", ""),
        "batch": target.get("batch", ""),
        "source": source,
        "provider": provider,
        "target_mapping_status": mapping_status,
        "provider_target_name": mapping.get("name", ""),
        "norad": mapping.get("norad", str(target.get("norad", ""))),
        "cospar": mapping.get("cospar", ""),
        "sic": mapping.get("sic", ""),
        "probe_start_utc": start,
        "probe_end_utc": end,
        "probe_status": probe_status,
        "candidate_file_count": 0,
        "local_file_count": local_count,
        "claim_allowed": local_count > 0,
        "source_url_or_provider": source_url_or_provider,
        "required_credentials": required_credentials,
        "next_action": next_action,
        "notes": "",
    }


def _execute_slr_probe(row: dict[str, Any]) -> dict[str, Any]:
    if row["source"] != "slr" or row["target_mapping_status"] != "mapped":
        return row
    start = datetime.fromisoformat(row["probe_start_utc"].replace("Z", "+00:00")).replace(tzinfo=None)
    end = datetime.fromisoformat(row["probe_end_utc"].replace("Z", "+00:00")).replace(tzinfo=None)
    try:
        granules = list_cddis_cmr_granules(row["sat_id"], start, end, verbose=False)
    except Exception as exc:  # network or CMR failures are probe status, not source absence.
        return {
            **row,
            "probe_status": "probe_error",
            "notes": str(exc),
            "next_action": "retry_probe_or_check_provider",
        }
    count = len(granules)
    return {
        **row,
        "probe_status": "probe_available" if count > 0 else "probe_no_hits",
        "candidate_file_count": count,
        "next_action": "download_verified_slr_files" if count > 0 else "probe_mission_specific_window",
    }


def build_probe_plan(
    config: dict[str, Any],
    batches: list[str] | None = None,
    sources: list[str] | None = None,
    execute_probe: bool = False,
    include: list[str] | None = None,
) -> pd.DataFrame:
    """Build optional live probe status for reference auxiliary evidence."""
    batch_filter = set(batches or [])
    selected_sources = sources or ["slr", "precise_orbit"]
    include_filter = set(include or [])
    rows = []
    for target in config.get("targets", []):
        if batch_filter and target.get("batch") not in batch_filter:
            continue
        if include_filter and target.get("sat_id") not in include_filter:
            continue
        for source in selected_sources:
            row = _base_row(target, source)
            if execute_probe:
                row = _execute_slr_probe(row)
            rows.append(row)
    plan = pd.DataFrame(rows, columns=PROBE_STATUS_COLUMNS)
    for column in ["claim_allowed"]:
        plan[column] = plan[column].astype(object)
    return plan


def write_probe_status(status: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write source probe status as a small decision table."""
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = REPO_ROOT / destination
    ensure_directory(destination.parent)
    status.to_csv(destination, index=False)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe reference auxiliary evidence sources")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Maneuver annotation target config")
    parser.add_argument("--batch", action="append", default=None, help="Batch letter to include; repeatable")
    parser.add_argument("--include", action="append", default=None, help="sat_id to include; repeatable")
    parser.add_argument(
        "--source",
        action="append",
        choices=["slr", "precise_orbit", "pod"],
        default=None,
        help="Source to probe or plan; repeatable",
    )
    parser.add_argument("--execute-probe", action="store_true", help="Execute live official archive probes")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_annotation_config(args.config)
    status = build_probe_plan(
        config,
        batches=args.batch,
        sources=args.source,
        execute_probe=args.execute_probe,
        include=args.include,
    )
    output = write_probe_status(status, args.output)
    print(f"wrote {output} rows={len(status)}")


if __name__ == "__main__":
    main()
