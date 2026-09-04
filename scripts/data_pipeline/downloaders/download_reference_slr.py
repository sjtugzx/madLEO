"""Controlled SLR downloads for reference validation targets."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from downloaders.ilrs_downloader import (
    ILRS_TARGETS,
    download_cddis_cmr_granules,
    download_edc_file,
    download_ilrs_data,
    list_cddis_cmr_granules,
)
from downloaders.probe_reference_evidence_sources import compute_probe_window, load_annotation_config


DEFAULT_METADATA_ROOT = REPO_ROOT / "data" / "validation" / "download_status" / "benchmark"
DEFAULT_ALIGNMENT_TABLE = REPO_ROOT / "docs" / "tables" / "reference_recent_window_alignment_status.csv"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def build_slr_download_plan(
    target: dict[str, Any],
    max_files: int | None = None,
    data_type: str = "npt",
    transport: str = "cddis_cmr",
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
) -> dict[str, Any]:
    """Build a controlled SLR download plan for one target."""
    sat_id = target["sat_id"]
    output_dir = resolve_repo_path(target.get("slr_raw_path") or f"data/raw/reference_validation/{sat_id}/slr")
    metadata_dir = Path(metadata_root)
    if not metadata_dir.is_absolute():
        metadata_dir = REPO_ROOT / metadata_dir
    metadata_dir = metadata_dir / sat_id / "slr"

    base = {
        "sat_id": sat_id,
        "output_dir": str(output_dir),
        "metadata_dir": str(metadata_dir),
        "max_files": max_files if max_files is not None else "",
        "download_scope": "partial_probe_download" if max_files else "full_window_download",
        "data_type": data_type,
        "transport": transport,
        "provider": "ILRS/EDC" if transport == "edc" else "ILRS/CDDIS",
        "next_action": "download_verified_slr_files",
    }
    if sat_id not in ILRS_TARGETS:
        return {
            **base,
            "status": "provider_mapping_missing",
            "download_scope": "none",
            "window_start_utc": "",
            "window_end_utc": "",
            "next_action": "verify_official_target_mapping",
        }

    start, end = compute_probe_window(target)
    return {
        **base,
        "status": "planned",
        "window_start_utc": start,
        "window_end_utc": end,
    }


def _event_windows_path(target: dict[str, Any]) -> Path:
    normalized_path = target.get("normalized_annotation_path")
    if normalized_path:
        path = Path(str(normalized_path)).parent / "event_windows.csv"
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path
    return REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation" / target["sat_id"] / "event_windows.csv"


def _format_utc(value: pd.Timestamp) -> str:
    if value.microsecond:
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def build_slr_window_plans(
    target: dict[str, Any],
    per_window_max_files: int | None = None,
    window_margin_days: int = 1,
    start_window: int = 1,
    max_windows: int | None = None,
    data_type: str = "npt",
    transport: str = "cddis_cmr",
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
) -> list[dict[str, Any]]:
    """Build SLR download plans around each mission-reported maneuver window."""
    event_path = _event_windows_path(target)
    if not event_path.exists():
        return [
            build_slr_download_plan(
                target,
                max_files=per_window_max_files,
                data_type=data_type,
                transport=transport,
                metadata_root=metadata_root,
            )
        ]

    events = pd.read_csv(event_path)
    starts = pd.to_datetime(events["window_start_utc"], utc=True, errors="coerce")
    ends = pd.to_datetime(events["window_end_utc"], utc=True, errors="coerce")
    base_plan = build_slr_download_plan(
        target,
        max_files=per_window_max_files,
        data_type=data_type,
        transport=transport,
        metadata_root=metadata_root,
    )
    plans: list[dict[str, Any]] = []
    start_index = max(0, start_window - 1)
    end_index = None if max_windows is None else start_index + max_windows
    selected_events = events.iloc[start_index:end_index]
    for position, (_, row) in enumerate(selected_events.iterrows()):
        index = start_index + position
        start = starts.iloc[index]
        end = ends.iloc[index]
        if pd.isna(start) or pd.isna(end):
            continue
        event_id = str(row.get("annotation_id") or row.get("event_window_id") or f"event_window_{index + 1}")
        margin = pd.Timedelta(days=window_margin_days)
        plans.append(
            {
                **base_plan,
                "download_scope": "event_window_partial_download" if per_window_max_files else "event_window_full_download",
                "event_window_id": event_id,
                "event_window_index": int(index + 1),
                "window_start_utc": _format_utc(start - margin),
                "window_end_utc": _format_utc(end + margin),
                "max_files": per_window_max_files if per_window_max_files is not None else "",
            }
        )
    return plans or [base_plan]


def build_slr_gap_window_plans(
    target: dict[str, Any],
    alignment_table: str | Path = DEFAULT_ALIGNMENT_TABLE,
    per_window_max_files: int | None = None,
    window_margin_days: int = 1,
    data_type: str = "npt",
    transport: str = "cddis_cmr",
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
) -> list[dict[str, Any]]:
    """Build SLR download plans from alignment rows still missing SLR."""
    alignment_path = Path(alignment_table)
    if not alignment_path.is_absolute():
        alignment_path = REPO_ROOT / alignment_path
    if not alignment_path.exists():
        return []

    gaps = pd.read_csv(alignment_path)
    if gaps.empty:
        return []
    gaps["missing_sources"] = gaps["missing_sources"].fillna("")
    gaps = gaps[
        (gaps["sat_id"].astype(str) == str(target["sat_id"]))
        & gaps["missing_sources"].str.split(",").apply(lambda values: "slr" in {value.strip() for value in values})
    ]
    if gaps.empty:
        return []

    base_plan = build_slr_download_plan(
        target,
        max_files=per_window_max_files,
        data_type=data_type,
        transport=transport,
        metadata_root=metadata_root,
    )
    plans: list[dict[str, Any]] = []
    margin = pd.Timedelta(days=window_margin_days)
    for index, row in enumerate(gaps.itertuples(index=False), start=1):
        start = pd.to_datetime(row.window_start_utc, utc=True, errors="coerce")
        end = pd.to_datetime(row.window_end_utc, utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        plans.append(
            {
                **base_plan,
                "download_scope": "alignment_gap_partial_download"
                if per_window_max_files is not None
                else "alignment_gap_full_download",
                "event_window_id": str(row.annotation_id),
                "event_window_index": index,
                "window_start_utc": _format_utc(start - margin),
                "window_end_utc": _format_utc(end + margin),
                "max_files": per_window_max_files if per_window_max_files is not None else "",
            }
        )
    return plans


def write_download_metadata(metadata: dict[str, Any], metadata_root: str | Path = DEFAULT_METADATA_ROOT) -> dict[str, Any]:
    """Write SLR download metadata under the ignored validation tree."""
    sat_id = metadata["sat_id"]
    root = Path(metadata_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    metadata_dir = ensure_directory(root / sat_id / "slr")
    (metadata_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def download_remote_slr_files(
    sat_id: str,
    remote_paths: list[str],
    output_dir: str | Path,
    data_type: str,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
) -> dict[str, Any]:
    """Download a vetted list of EDC remote SLR files and record exact outcomes."""
    downloaded_files = []
    failed_files = []
    for remote_path in remote_paths:
        local_path = download_edc_file(remote_path, str(output_dir), decompress=True, verbose=True)
        if local_path:
            downloaded_files.append(local_path)
        else:
            failed_files.append(remote_path)
    if downloaded_files and failed_files:
        status = "partial_download"
    elif downloaded_files:
        status = "downloaded"
    else:
        status = "no_files_downloaded"
    metadata = {
        "sat_id": sat_id,
        "provider": "ILRS/EDC",
        "transport": "edc_remote_list",
        "data_type": data_type,
        "download_scope": "explicit_remote_file_list",
        "output_dir": str(output_dir),
        "status": status,
        "requested_file_count": len(remote_paths),
        "downloaded_file_count": len(downloaded_files),
        "failed_file_count": len(failed_files),
        "downloaded_files": downloaded_files,
        "failed_files": failed_files,
    }
    return write_download_metadata(metadata, metadata_root=metadata_root)


def _dated_slr_files(output_dir: str | Path, start_utc: str, end_utc: str) -> list[str]:
    path = Path(output_dir)
    if not path.exists():
        return []
    start_date = _parse_utc(start_utc).date()
    end_date = _parse_utc(end_utc).date()
    files = []
    for item in path.iterdir():
        if not item.is_file() or item.stat().st_size <= 0 or item.name.startswith(".") or ".part" in item.name:
            continue
        match = re.search(r"_(\d{8})(?:\D|$)", item.name)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if start_date <= file_date <= end_date:
            files.append(str(item))
    return sorted(files)


def _granule_date(granule: dict[str, Any]) -> datetime.date | None:
    values = [str(granule.get("producer_granule_id", ""))]
    values.extend(str(link.get("href", "")) for link in granule.get("links", []))
    for value in values:
        match = re.search(r"_(\d{8})(?:\D|$)", value)
        if not match:
            continue
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
    return None


def _annual_probe_counts(plan: list[dict[str, Any]]) -> dict[str, int]:
    years = sorted(
        {
            year
            for row in plan
            for year in range(_parse_utc(row["window_start_utc"]).year, _parse_utc(row["window_end_utc"]).year + 1)
        }
    )
    granule_dates: dict[datetime.date, int] = {}
    for year in years:
        try:
            granules = list_cddis_cmr_granules(
                plan[0]["sat_id"],
                datetime(year, 1, 1),
                datetime(year, 12, 31),
                verbose=False,
            )
        except Exception:
            continue
        for granule in granules:
            date = _granule_date(granule)
            if date is not None:
                granule_dates[date] = granule_dates.get(date, 0) + 1

    counts = {}
    for row in plan:
        start_date = _parse_utc(row["window_start_utc"]).date()
        end_date = _parse_utc(row["window_end_utc"]).date()
        counts[row.get("event_window_id", "")] = sum(
            count for date, count in granule_dates.items() if start_date <= date <= end_date
        )
    return counts


def summarize_slr_window_plan(
    plan: list[dict[str, Any]],
    execute_probe: bool = False,
    probe_strategy: str = "window",
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
) -> dict[str, Any]:
    """Summarize local and optional remote SLR coverage for event-window plans."""
    if not plan:
        raise ValueError("At least one SLR download plan is required")
    if probe_strategy not in {"window", "annual"}:
        raise ValueError("probe_strategy must be 'window' or 'annual'")

    rows = []
    remote_candidate_file_count = 0
    remote_candidate_window_count = 0
    local_files = set()
    annual_counts = _annual_probe_counts(plan) if execute_probe and probe_strategy == "annual" else {}
    for row in plan:
        local = _dated_slr_files(row["output_dir"], row["window_start_utc"], row["window_end_utc"])
        local_files.update(local)
        probe_status = "not_probed"
        candidate_count = 0
        if execute_probe and row.get("status") == "planned":
            if probe_strategy == "annual":
                candidate_count = annual_counts.get(row.get("event_window_id", ""), 0)
                probe_status = "probe_available" if candidate_count else "probe_no_hits"
            else:
                try:
                    granules = list_cddis_cmr_granules(
                        row["sat_id"],
                        _parse_utc(row["window_start_utc"]),
                        _parse_utc(row["window_end_utc"]),
                        verbose=False,
                    )
                    candidate_count = len(granules)
                    probe_status = "probe_available" if candidate_count else "probe_no_hits"
                except Exception as exc:
                    probe_status = "probe_error"
                    row = {**row, "probe_error": str(exc)}
        if candidate_count:
            remote_candidate_window_count += 1
            remote_candidate_file_count += candidate_count
        rows.append(
            {
                "event_window_id": row.get("event_window_id", ""),
                "event_window_index": row.get("event_window_index", ""),
                "window_start_utc": row.get("window_start_utc", ""),
                "window_end_utc": row.get("window_end_utc", ""),
                "local_file_count": len(local),
                "local_files": local,
                "probe_status": probe_status,
                "remote_candidate_file_count": candidate_count,
            }
        )

    first = plan[0]
    summary = {
        "sat_id": first["sat_id"],
        "provider": first["provider"],
        "download_scope": first["download_scope"],
        "probe_strategy": probe_strategy if execute_probe else "not_probed",
        "event_window_count": len(plan),
        "local_file_count": len(local_files),
        "remote_candidate_window_count": remote_candidate_window_count,
        "remote_candidate_file_count": remote_candidate_file_count,
        "event_windows": rows,
    }

    root = Path(metadata_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    metadata_dir = ensure_directory(root / first["sat_id"] / "slr")
    (metadata_dir / "window_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def execute_slr_download_plan(
    plan: dict[str, Any] | list[dict[str, Any]],
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
) -> dict[str, Any]:
    """Execute a planned SLR download and write metadata."""
    if isinstance(plan, list):
        if not plan:
            raise ValueError("At least one SLR download plan is required")
        window_results = []
        downloaded_files: list[str] = []
        seen_files = set()
        for window_plan in plan:
            metadata = _execute_single_slr_download_plan(window_plan)
            window_results.append(
                {
                    "event_window_id": window_plan.get("event_window_id", ""),
                    "window_start_utc": window_plan.get("window_start_utc", ""),
                    "window_end_utc": window_plan.get("window_end_utc", ""),
                    "status": metadata["status"],
                    "downloaded_file_count": metadata["downloaded_file_count"],
                    "downloaded_files": metadata["downloaded_files"],
                }
            )
            for path in metadata["downloaded_files"]:
                if path not in seen_files:
                    seen_files.add(path)
                    downloaded_files.append(path)

        first = plan[0]
        aggregate = {
            **first,
            "status": "downloaded" if downloaded_files else "no_files_downloaded",
            "event_window_count": len(plan),
            "downloaded_file_count": len(downloaded_files),
            "downloaded_files": downloaded_files,
            "event_windows": window_results,
        }
        return write_download_metadata(aggregate, metadata_root=metadata_root)

    return write_download_metadata(_execute_single_slr_download_plan(plan), metadata_root=metadata_root)


def _execute_single_slr_download_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute one planned SLR window without writing metadata."""
    if plan.get("status") != "planned":
        return {**plan, "downloaded_file_count": 0, "downloaded_files": []}

    try:
        max_files = int(plan["max_files"]) if plan.get("max_files") != "" else None
        if plan.get("transport") == "edc":
            downloaded = download_ilrs_data(
                satellite=plan["sat_id"],
                start_date=_parse_utc(plan["window_start_utc"]),
                end_date=_parse_utc(plan["window_end_utc"]),
                output_dir=plan["output_dir"],
                data_type=plan.get("data_type", "npt"),
                max_files=max_files,
                include_cddis_supplement=not bool(plan.get("skip_cddis_supplement", False)),
                verbose=True,
            )
        else:
            downloaded = download_cddis_cmr_granules(
                satellite=plan["sat_id"],
                start_date=_parse_utc(plan["window_start_utc"]),
                end_date=_parse_utc(plan["window_end_utc"]),
                output_dir=plan["output_dir"],
                max_files=max_files,
                verbose=True,
            )
        metadata = {
            **plan,
            "status": "downloaded" if downloaded else "no_files_downloaded",
            "downloaded_file_count": len(downloaded),
            "downloaded_files": downloaded,
        }
    except Exception as exc:
        metadata = {
            **plan,
            "status": "download_failed",
            "downloaded_file_count": 0,
            "downloaded_files": [],
            "error": str(exc),
        }
    return metadata


def _iter_targets(config: dict[str, Any], batches: list[str] | None, include: list[str] | None) -> list[dict[str, Any]]:
    batch_filter = set(batches or [])
    include_filter = set(include or [])
    rows = []
    for target in config.get("targets", []):
        if batch_filter and target.get("batch") not in batch_filter:
            continue
        if include_filter and target.get("sat_id") not in include_filter:
            continue
        rows.append(target)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download reference SLR evidence with explicit scope metadata")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "targets_maneuver_annotations.json"))
    parser.add_argument("--batch", action="append", default=None)
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--data-type", choices=["npt", "frd"], default="npt")
    parser.add_argument("--transport", choices=["cddis_cmr", "edc"], default="cddis_cmr")
    parser.add_argument(
        "--skip-cddis-supplement",
        action="store_true",
        help="For EDC NPT downloads, skip the extra CDDIS CMR supplement pass",
    )
    parser.add_argument(
        "--window-mode",
        choices=["target", "events", "gaps"],
        default="target",
        help="Download one target-level window, mission-reported event windows, or current alignment gap windows",
    )
    parser.add_argument("--per-window-max-files", type=int, default=None)
    parser.add_argument("--window-margin-days", type=int, default=1)
    parser.add_argument("--start-window", type=int, default=1, help="1-based event-window index to start from")
    parser.add_argument("--max-windows", type=int, default=None, help="Maximum event windows to process in this run")
    parser.add_argument("--metadata-root", default=str(DEFAULT_METADATA_ROOT))
    parser.add_argument("--alignment-table", default=str(DEFAULT_ALIGNMENT_TABLE))
    parser.add_argument("--remote-list", default=None, help="Text file containing one EDC remote path per line")
    parser.add_argument("--sat-id", default=None, help="sat_id for --remote-list mode")
    parser.add_argument("--output-dir", default=None, help="output directory for --remote-list mode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize-windows", action="store_true")
    parser.add_argument("--execute-probe", action="store_true")
    parser.add_argument(
        "--probe-strategy",
        choices=["window", "annual"],
        default="window",
        help="Probe each event window directly or query CMR once per year and map granule dates back to windows",
    )
    args = parser.parse_args()
    if args.remote_list and not (args.sat_id and args.output_dir):
        parser.error("--remote-list requires --sat-id and --output-dir")
    active_modes = sum(bool(value) for value in [args.dry_run, args.execute, args.summarize_windows])
    if active_modes > 1:
        parser.error("--dry-run, --execute, and --summarize-windows are mutually exclusive")
    if args.window_mode != "events" and (args.max_windows is not None or args.start_window != 1):
        parser.error("--start-window and --max-windows require --window-mode events")
    if args.skip_cddis_supplement and args.transport != "edc":
        parser.error("--skip-cddis-supplement requires --transport edc")
    return args


def _with_skip_cddis_supplement(
    plan: dict[str, Any] | list[dict[str, Any]],
    skip_cddis_supplement: bool,
) -> dict[str, Any] | list[dict[str, Any]]:
    if not skip_cddis_supplement:
        return plan
    if isinstance(plan, list):
        return [{**row, "skip_cddis_supplement": True} for row in plan]
    return {**plan, "skip_cddis_supplement": True}


def main() -> None:
    args = parse_args()
    if args.remote_list:
        remote_paths = [
            line.strip()
            for line in Path(args.remote_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if args.execute:
            metadata = download_remote_slr_files(
                sat_id=args.sat_id,
                remote_paths=remote_paths,
                output_dir=args.output_dir,
                data_type=args.data_type,
                metadata_root=args.metadata_root,
            )
            print(f"{metadata['sat_id']} status={metadata['status']} files={metadata['downloaded_file_count']}")
        else:
            print(f"{args.sat_id} remote_files={len(remote_paths)} output={args.output_dir}")
        return

    config = load_annotation_config(args.config)
    for target in _iter_targets(config, args.batch, args.include):
        if args.window_mode == "events":
            plan = build_slr_window_plans(
                target,
                per_window_max_files=args.per_window_max_files if args.per_window_max_files is not None else args.max_files,
                window_margin_days=args.window_margin_days,
                start_window=args.start_window,
                max_windows=args.max_windows,
                data_type=args.data_type,
                transport=args.transport,
                metadata_root=args.metadata_root,
            )
        elif args.window_mode == "gaps":
            plan = build_slr_gap_window_plans(
                target,
                alignment_table=args.alignment_table,
                per_window_max_files=args.per_window_max_files if args.per_window_max_files is not None else args.max_files,
                window_margin_days=args.window_margin_days,
                data_type=args.data_type,
                transport=args.transport,
                metadata_root=args.metadata_root,
            )
        else:
            plan = build_slr_download_plan(
                target,
                max_files=args.max_files,
                data_type=args.data_type,
                transport=args.transport,
                metadata_root=args.metadata_root,
            )
        plan = _with_skip_cddis_supplement(plan, args.skip_cddis_supplement)
        if isinstance(plan, list) and not plan:
            print(f"{target['sat_id']} status=no_gap_windows files=0")
            continue
        if args.execute:
            metadata = execute_slr_download_plan(plan, metadata_root=args.metadata_root)
            print(f"{metadata['sat_id']} status={metadata['status']} files={metadata['downloaded_file_count']}")
        elif args.summarize_windows:
            rows = plan if isinstance(plan, list) else [plan]
            summary = summarize_slr_window_plan(
                rows,
                execute_probe=args.execute_probe,
                probe_strategy=args.probe_strategy,
                metadata_root=args.metadata_root,
            )
            print(
                f"{summary['sat_id']} windows={summary['event_window_count']} "
                f"local_files={summary['local_file_count']} "
                f"remote_candidate_windows={summary['remote_candidate_window_count']} "
                f"remote_candidate_files={summary['remote_candidate_file_count']}"
            )
        else:
            rows = plan if isinstance(plan, list) else [plan]
            for row in rows:
                event = f" event={row.get('event_window_id')}" if row.get("event_window_id") else ""
                print(
                    f"{row['sat_id']}{event} status={row['status']} scope={row['download_scope']} "
                    f"window={row.get('window_start_utc', '')}..{row.get('window_end_utc', '')} "
                    f"max_files={row.get('max_files', '')} output={row['output_dir']}"
                )


if __name__ == "__main__":
    main()
