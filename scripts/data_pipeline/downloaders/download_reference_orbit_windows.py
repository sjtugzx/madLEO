"""Download reference orbit products for event-window alignment gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from downloaders.download_batch_tle_history import DEFAULT_CONFIG, load_annotation_config
from downloaders.cdse_downloader import (
    download_cdse_product,
    get_cdse_token,
    list_cdse_sentinel3_orbits,
)
from downloaders.download_utils import load_repo_env
from downloaders.net_guard import assert_path_within, sanitize_remote_filename
from downloaders.download_reference_orbit_samples import (
    SAMPLE_SOURCE_CONTRACTS,
    _candidate_download_urls,
    _pick_https_data_link,
    _validate_download,
    download_https_earthdata_file,
    list_cmr_granules,
)


DEFAULT_ALIGNMENT_TABLE = REPO_ROOT / "docs" / "tables" / "reference_recent_window_alignment_status.csv"
DEFAULT_METADATA_ROOT = REPO_ROOT / "data" / "validation" / "download_status" / "benchmark"


ORBIT_WINDOW_CONTRACTS: dict[str, dict[str, Any]] = {
    **SAMPLE_SOURCE_CONTRACTS,
    "sentinel-3a": {
        "source": "pod",
        "provider": "Copernicus Data Space Ecosystem AUX_POEORB",
        "collection_concept_id": "",
        "data_type": "sentinel3_aux_poeorb_eof",
        "output_dir": "data/raw/reference_validation/sentinel-3a/pod",
        "satellite": "S3A",
        "product_type": "AUX_POEORB",
        "product_types": ["AUX_POEORB", "AUX_MOEORB", "AUX_RESORB"],
        "expected_signature": "xml",
        "download_backend": "cdse_sentinel3",
    },
    "sentinel-3b": {
        "source": "pod",
        "provider": "Copernicus Data Space Ecosystem AUX_POEORB",
        "collection_concept_id": "",
        "data_type": "sentinel3_aux_poeorb_eof",
        "output_dir": "data/raw/reference_validation/sentinel-3b/pod",
        "satellite": "S3B",
        "product_type": "AUX_POEORB",
        "product_types": ["AUX_POEORB", "AUX_MOEORB", "AUX_RESORB"],
        "expected_signature": "xml",
        "download_backend": "cdse_sentinel3",
    },
    "sentinel-6a": {
        **SAMPLE_SOURCE_CONTRACTS["sentinel-6a"],
        "temporal": "",
    },
    "jason-3": {
        "source": "pod",
        "provider": "NASA/JPL PO.DAAC Jason-3 OGDR GPS",
        "collection_concept_id": "C2205122298-POCLOUD",
        "data_type": "netcdf_gps_orbit_related",
        "output_dir": "data/raw/reference_validation/jason-3/pod",
        "link_suffix": ".nc",
        "expected_signature": "hdf5",
    },
    "jason-1": {
        "source": "precise_orbit",
        "provider": "NASA CDDIS DORIS_IDS_orbit_prod",
        "collection_concept_id": "C1602851234-CDDIS",
        "producer_granule_id_pattern": "*ja1*",
        "data_type": "compressed_sp3",
        "output_dir": "data/raw/reference_validation/jason-1/precise_orbit",
        "link_suffix": ".Z",
        "expected_signature": "unix_compress",
    },
}


def _iso(value: object) -> str:
    ts = pd.to_datetime(value, utc=True)
    if ts.microsecond:
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def _contract_for_target(target: dict[str, Any]) -> dict[str, Any] | None:
    contract = ORBIT_WINDOW_CONTRACTS.get(target["sat_id"])
    if contract is None:
        return None
    output_dir = target.get("pod_raw_path") if contract["source"] == "pod" else target.get("precise_orbit_raw_path")
    if output_dir:
        return {**contract, "output_dir": output_dir}
    return contract


def build_orbit_window_plan(
    config: dict[str, Any],
    alignment_table: str | Path = DEFAULT_ALIGNMENT_TABLE,
    batches: list[str] | None = None,
    include: list[str] | None = None,
) -> pd.DataFrame:
    """Build a window-level orbit download plan from local alignment gaps."""
    alignment_path = Path(alignment_table)
    if not alignment_path.is_absolute():
        alignment_path = REPO_ROOT / alignment_path
    gaps = pd.read_csv(alignment_path)
    if gaps.empty:
        return pd.DataFrame()
    gaps["missing_sources"] = gaps["missing_sources"].fillna("")
    gaps = gaps[gaps["missing_sources"].str.contains("orbit")]

    batch_filter = set(batches or [])
    include_filter = set(include or [])
    targets = {
        target["sat_id"]: target
        for target in config.get("targets", [])
        if (not batch_filter or target.get("batch") in batch_filter)
        and (not include_filter or target.get("sat_id") in include_filter)
    }

    rows = []
    for gap in gaps.itertuples(index=False):
        sat_id = str(gap.sat_id)
        target = targets.get(sat_id)
        if target is None:
            continue
        contract = _contract_for_target(target)
        base = {
            "sat_id": sat_id,
            "batch": target.get("batch", ""),
            "annotation_id": gap.annotation_id,
            "window_start_utc": _iso(gap.window_start_utc),
            "window_end_utc": _iso(gap.window_end_utc),
            "status": "planned",
            "provider": "",
            "source": "",
            "collection_concept_id": "",
            "output_dir": "",
            "data_type": "",
            "next_action": "download_window_orbit_product",
        }
        if contract is None:
            rows.append({**base, "status": "unsupported_target", "next_action": "discover_official_orbit_source"})
            continue
        rows.append(
            {
                **base,
                "provider": contract["provider"],
                "source": contract["source"],
                "collection_concept_id": contract["collection_concept_id"],
                "output_dir": contract["output_dir"],
                "data_type": contract["data_type"],
            }
        )
    return pd.DataFrame(rows)


def _slice_plan(plan: pd.DataFrame, start_window: int = 1, max_windows: int | None = None) -> pd.DataFrame:
    """Return a deterministic 1-based slice of a window-level plan."""
    if start_window < 1:
        raise ValueError("start_window must be 1 or greater")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be 1 or greater")
    start_index = start_window - 1
    if max_windows is None:
        return plan.iloc[start_index:].reset_index(drop=True)
    return plan.iloc[start_index : start_index + max_windows].reset_index(drop=True)


def _window_contract(contract: dict[str, Any], start_utc: str, end_utc: str) -> dict[str, Any]:
    return {**contract, "temporal": f"{start_utc},{end_utc}"}


def _intervals_overlap(start_utc: str, end_utc: str, granule_start: object, granule_end: object) -> bool:
    if not granule_start or not granule_end:
        return True
    start = pd.to_datetime(start_utc, utc=True)
    end = pd.to_datetime(end_utc, utc=True)
    candidate_start = pd.to_datetime(granule_start, utc=True, errors="coerce")
    candidate_end = pd.to_datetime(granule_end, utc=True, errors="coerce")
    if pd.isna(candidate_start) or pd.isna(candidate_end):
        return True
    return bool(candidate_start < end and candidate_end > start)


def _parse_sp3_year_doy(two_digit_year: str, doy: str) -> pd.Timestamp | None:
    try:
        year_fragment = int(two_digit_year)
        day = int(doy)
    except ValueError:
        return None
    year = 2000 + year_fragment if year_fragment < 57 else 1900 + year_fragment
    return pd.Timestamp(datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1))


def _filename_interval_from_href(href: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    match = re.search(r"\.b(\d{2})(\d{3})\.e(\d{2})(\d{3})\.", href.rsplit("/", 1)[-1])
    if not match:
        return None
    start = _parse_sp3_year_doy(match.group(1), match.group(2))
    end = _parse_sp3_year_doy(match.group(3), match.group(4))
    if start is None or end is None:
        return None
    return start, end + pd.Timedelta(days=1)


def _href_overlaps_window(href: str, start_utc: str, end_utc: str) -> bool:
    interval = _filename_interval_from_href(href)
    if interval is None:
        return True
    return _intervals_overlap(start_utc, end_utc, interval[0], interval[1])


def _href_requires_filename_interval(contract: dict[str, Any], href: str) -> bool:
    """Return true when CMR time metadata is too broad to prove file overlap."""
    if contract.get("data_type") != "compressed_sp3":
        return False
    if contract.get("collection_concept_id") != "C1602851234-CDDIS":
        return False
    return "JasonOrbits" in href or href.endswith(".Z")


def _granule_overlaps_window(granule: dict[str, Any], start_utc: str, end_utc: str) -> bool:
    return _intervals_overlap(
        start_utc,
        end_utc,
        granule.get("content_start") or granule.get("time_start") or granule.get("start_date"),
        granule.get("content_end") or granule.get("time_end") or granule.get("end_date"),
    )


def _default_list_granules(contract: dict[str, Any], start_utc: str, end_utc: str) -> list[dict[str, Any]]:
    load_repo_env()
    if contract.get("download_backend") == "cdse_sentinel3":
        for product_type in contract.get("product_types", [contract.get("product_type", "AUX_POEORB")]):
            granules = list_cdse_sentinel3_orbits(
                contract["satellite"],
                start_date=pd.to_datetime(start_utc, utc=True).to_pydatetime().replace(tzinfo=None),
                end_date=pd.to_datetime(end_utc, utc=True).to_pydatetime().replace(tzinfo=None),
                product_type=product_type,
                verbose=False,
            )
            if granules:
                return granules
        return []
    if contract.get("data_type") == "compressed_sp3" and contract.get("producer_granule_id_pattern"):
        matched_granules: list[dict[str, Any]] = []
        seen_granules: set[str] = set()
        for pattern in _doris_sp3_window_patterns(contract, start_utc, end_utc):
            granules = _list_cmr_granules_pages(
                _window_contract({**contract, "producer_granule_id_pattern": pattern}, start_utc, end_utc),
                page_size=25,
            )
            for granule in granules:
                identity = str(granule.get("producer_granule_id") or granule.get("id") or granule.get("title") or granule)
                if identity in seen_granules:
                    continue
                seen_granules.add(identity)
                matched_granules.append(granule)
        if matched_granules:
            return matched_granules
    page_size = 2000 if contract.get("producer_granule_id_pattern") else 25
    granules = _list_cmr_granules_pages(_window_contract(contract, start_utc, end_utc), page_size=page_size)
    if granules:
        return granules
    for fallback in contract.get("fallback_contracts", []):
        fallback_contract = {**contract, **fallback}
        granules = _list_cmr_granules_pages(_window_contract(fallback_contract, start_utc, end_utc), page_size=page_size)
        if granules:
            return granules
    return []


def _list_cmr_granules_pages(contract: dict[str, Any], page_size: int = 25, max_pages: int = 20) -> list[dict[str, Any]]:
    """List CMR granules across numbered pages until the first short page."""
    granules: list[dict[str, Any]] = []
    for page_num in range(1, max_pages + 1):
        page = list_cmr_granules(contract, page_size=page_size, page_num=page_num)
        granules.extend(page)
        if len(page) < page_size:
            break
    return granules


def _doris_sp3_window_patterns(contract: dict[str, Any], start_utc: str, end_utc: str) -> list[str]:
    """Build narrow CMR producer-id patterns for DORIS SP3 files near a window."""
    base_pattern = str(contract.get("producer_granule_id_pattern") or "*")
    radius_days = _doris_sp3_pattern_radius_days()
    start = pd.to_datetime(start_utc, utc=True)
    end = pd.to_datetime(end_utc, utc=True)
    start_day = start.floor("D")
    end_day = end.ceil("D")
    candidate_days = set(
        pd.date_range(
            start_day - pd.Timedelta(days=radius_days),
            end_day + pd.Timedelta(days=radius_days),
            freq="D",
        )
    )
    offsets = [0]
    for offset in range(1, radius_days + 1):
        offsets.extend([-offset, offset])
    ordered_days = []
    for offset in offsets:
        day = start_day + pd.Timedelta(days=offset)
        if day in candidate_days:
            ordered_days.append(day)
    for day in sorted(candidate_days):
        if day not in ordered_days:
            ordered_days.append(day)
    patterns: list[str] = []
    for day in ordered_days:
        token = f"{day.year % 100:02d}{day.dayofyear:03d}"
        patterns.append(f"{base_pattern}b{token}*")
    deduped: list[str] = []
    seen = set()
    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        deduped.append(pattern)
    max_patterns = _doris_sp3_max_patterns()
    if max_patterns is not None:
        return deduped[:max_patterns]
    return deduped


def _doris_sp3_pattern_radius_days() -> int:
    """Return the SP3 producer-id search radius around each window start day."""
    raw_value = os.environ.get("DORIS_SP3_PATTERN_RADIUS_DAYS", "14")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 14


def _doris_sp3_max_patterns() -> int | None:
    """Return an optional cap for bounded DORIS SP3 CMR probes."""
    raw_value = os.environ.get("DORIS_SP3_MAX_PATTERNS", "").strip()
    if not raw_value:
        return None
    try:
        return max(1, int(raw_value))
    except ValueError:
        return None


def _write_metadata(metadata: dict[str, Any], metadata_root: str | Path) -> dict[str, Any]:
    root = Path(metadata_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    metadata_dir = ensure_directory(root / metadata["sat_id"] / metadata["source"])
    (metadata_dir / "window_download_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def download_orbit_windows(
    config: dict[str, Any],
    alignment_table: str | Path = DEFAULT_ALIGNMENT_TABLE,
    batches: list[str] | None = None,
    include: list[str] | None = None,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    list_granules: Callable[[dict[str, Any], str, str], list[dict[str, Any]]] = _default_list_granules,
    downloader: Callable[[str, str | Path], Path] = download_https_earthdata_file,
    max_files_per_window: int | None = 1,
    start_window: int = 1,
    max_windows: int | None = None,
) -> dict[str, Any]:
    """Download validated orbit products for windows missing orbit coverage."""
    load_repo_env()
    plan = build_orbit_window_plan(config, alignment_table=alignment_table, batches=batches, include=include)
    plan = _slice_plan(plan, start_window=start_window, max_windows=max_windows)
    if plan.empty:
        return {"status": "no_windows", "downloaded_file_count": 0, "downloaded_files": [], "windows": []}

    downloaded_files: list[str] = []
    windows: list[dict[str, Any]] = []
    saw_download_failure = False

    targets = {target["sat_id"]: target for target in config.get("targets", [])}
    cdse_token: str | None = None
    total_windows = len(plan)
    for window_index, row in enumerate(plan.itertuples(index=False), start=1):
        print(
            f"Processing orbit window {window_index}/{total_windows}: "
            f"{row.sat_id} {row.annotation_id} {row.window_start_utc}..{row.window_end_utc}",
            flush=True,
        )
        if row.status != "planned":
            windows.append(row._asdict())
            continue
        target = targets[str(row.sat_id)]
        contract = _contract_for_target(target)
        if contract is None:
            windows.append({**row._asdict(), "status": "unsupported_target"})
            continue
        try:
            granules = list_granules(contract, row.window_start_utc, row.window_end_utc)
        except Exception as exc:
            saw_download_failure = True
            windows.append({**row._asdict(), "status": "cmr_query_failed", "error": str(exc)})
            continue
        if not granules:
            windows.append({**row._asdict(), "status": "no_cmr_granules"})
            continue

        window_downloads: list[str] = []
        errors: list[str] = []
        skip_reasons: list[str] = []
        for granule in granules:
            if not _granule_overlaps_window(granule, row.window_start_utc, row.window_end_utc):
                skip_reasons.append("granule_content_outside_window")
                continue
            if contract.get("download_backend") == "cdse_sentinel3":
                try:
                    if cdse_token is None:
                        import os

                        username = os.environ.get("CDSE_USERNAME")
                        password = os.environ.get("CDSE_PASSWORD")
                        if not username or not password:
                            raise RuntimeError("CDSE_USERNAME and CDSE_PASSWORD must be set for Sentinel-3 orbit downloads")
                        cdse_token = get_cdse_token(username, password)
                    downloaded_path = download_cdse_product(
                        granule,
                        output_dir=str(resolve_repo_path(contract["output_dir"])),
                        access_token=cdse_token,
                        verbose=True,
                    )
                    if downloaded_path:
                        window_downloads.append(str(downloaded_path))
                    if str(downloaded_path) not in downloaded_files:
                        downloaded_files.append(str(downloaded_path))
                except Exception as exc:
                    errors.append(f"{granule.get('filename', 'cdse_product')}: {exc}")
                if max_files_per_window is not None and len(window_downloads) >= max_files_per_window:
                    break
                continue
            href = _pick_https_data_link(granule, contract["link_suffix"])
            if not href:
                skip_reasons.append("no_matching_data_link")
                continue
            if _filename_interval_from_href(href) is None and _href_requires_filename_interval(contract, href):
                skip_reasons.append("filename_span_unparseable")
                continue
            if not _href_overlaps_window(href, row.window_start_utc, row.window_end_utc):
                skip_reasons.append("filename_span_outside_window")
                continue
            for candidate_href in _candidate_download_urls(href):
                output_path = resolve_repo_path(contract["output_dir"]) / sanitize_remote_filename(
                    candidate_href.rsplit("/", 1)[-1]
                )
                assert_path_within(output_path, resolve_repo_path(contract["output_dir"]))
                try:
                    downloaded_path = Path(downloader(candidate_href, output_path))
                except Exception as exc:
                    errors.append(f"{candidate_href}: {exc}")
                    continue
                validation_error = _validate_download(downloaded_path, contract["expected_signature"])
                if validation_error:
                    downloaded_path.unlink(missing_ok=True)
                    errors.append(f"{candidate_href}: {validation_error}")
                    continue
                window_downloads.append(str(downloaded_path))
                if str(downloaded_path) not in downloaded_files:
                    downloaded_files.append(str(downloaded_path))
                break
            if max_files_per_window is not None and len(window_downloads) >= max_files_per_window:
                break
        if window_downloads:
            windows.append(
                {
                    **row._asdict(),
                    "status": "downloaded",
                    "downloaded_files": window_downloads,
                    "downloaded_file_count": len(window_downloads),
                    "skip_reasons": sorted(set(skip_reasons)),
                    "candidate_granule_count": len(granules),
                    "skipped_granule_count": len(skip_reasons),
                    "errors": errors,
                    "failed_candidate_count": len(errors),
                }
            )
        else:
            saw_download_failure = True
            windows.append(
                {
                    **row._asdict(),
                    "status": "download_failed",
                    "errors": errors,
                    "skip_reasons": sorted(set(skip_reasons)),
                    "candidate_granule_count": len(granules),
                }
            )

    if downloaded_files:
        status = "partial_download" if saw_download_failure else "downloaded"
    else:
        status = "download_failed" if saw_download_failure else "no_files_downloaded"

    first_planned = plan.iloc[0].to_dict()
    sha256 = {}
    for file_path in downloaded_files:
        path = Path(file_path)
        if path.exists():
            sha256[file_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _write_metadata(
        {
            "sat_id": first_planned["sat_id"] if len(set(plan["sat_id"])) == 1 else "multiple",
            "source": first_planned.get("source", "orbit"),
            "status": status,
            "start_window": start_window,
            "max_windows": max_windows if max_windows is not None else "",
            "planned_window_count": len(plan),
            "downloaded_file_count": len(downloaded_files),
            "downloaded_files": downloaded_files,
            "sha256": sha256,
            "windows": windows,
        },
        metadata_root=metadata_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download orbit products for reference alignment gap windows")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--alignment-table", default=str(DEFAULT_ALIGNMENT_TABLE))
    parser.add_argument("--batch", action="append", default=None)
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--metadata-root", default=str(DEFAULT_METADATA_ROOT))
    parser.add_argument("--max-files-per-window", type=int, default=1)
    parser.add_argument("--start-window", type=int, default=1, help="1-based planned-window index to start from")
    parser.add_argument("--max-windows", type=int, default=None, help="Maximum planned windows to process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    if args.start_window < 1:
        parser.error("--start-window must be 1 or greater")
    if args.max_windows is not None and args.max_windows < 1:
        parser.error("--max-windows must be 1 or greater")
    return args


def main() -> None:
    args = parse_args()
    config = load_annotation_config(args.config)
    if args.execute:
        metadata = download_orbit_windows(
            config,
            alignment_table=args.alignment_table,
            batches=args.batch,
            include=args.include,
            metadata_root=args.metadata_root,
            max_files_per_window=args.max_files_per_window,
            start_window=args.start_window,
            max_windows=args.max_windows,
        )
        print(json.dumps({"status": metadata["status"], "files": metadata["downloaded_file_count"]}, sort_keys=True))
        return

    plan = build_orbit_window_plan(config, alignment_table=args.alignment_table, batches=args.batch, include=args.include)
    plan = _slice_plan(plan, start_window=args.start_window, max_windows=args.max_windows)
    print(plan.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
