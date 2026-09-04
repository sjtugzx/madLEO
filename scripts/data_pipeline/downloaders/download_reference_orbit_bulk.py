"""Bulk-download CDDIS DORIS SP3 orbit products for reference alignment gaps.

The per-window downloader (`download_reference_orbit_windows.py`) issues up to
~30 CMR pattern queries per gap window, which is slow and fragile through
proxied networks. This script instead pages the whole DORIS IDS orbit
collection once per satellite, parses the SP3 begin/end dates encoded in
producer granule IDs (CMR granule time metadata is series-level and
unreliable for these products), matches them locally against the alignment
gap windows (default `reference_recent_window_alignment_status.csv`, the
mission-tail gap view; pass `--alignment-table
results/tables/reference_all_window_alignment_status.csv` for the full-mission
audit), and downloads
the best file per window straight to the target's raw directory.
"""

from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarking.config import REPO_ROOT, resolve_repo_path
from downloaders.download_reference_orbit_samples import (
    _candidate_download_urls,
    download_https_earthdata_file,
    list_cmr_granules,
)
from downloaders.download_reference_orbit_windows import (
    DEFAULT_ALIGNMENT_TABLE,
    ORBIT_WINDOW_CONTRACTS,
    _validate_download,
)
from downloaders.net_guard import assert_path_within, sanitize_remote_filename

CDDIS_SP3_SAT_PATTERNS: dict[str, str] = {
    "topex-poseidon": "*top*",
    "jason-1": "*ja1*",
    "jason-2": "*ja2*",
    "jason-3": "*ja3*",
    "cryosat-2": "*cs2*",
    "hy-2a": "*h2a*",
    "saral": "*srl*",
}

# Satellites whose window contract in `download_reference_orbit_windows` does
# not point at the CDDIS DORIS SP3 collection. jason-3's window contract uses
# the PO.DAAC OGDR GPS product, which only starts 2020-10-29; the CDDIS DORIS
# SP3 series reaches back to its 2016 launch, so the bulk backfill overrides
# the contract with this CDDIS-side one.
CDDIS_SP3_CONTRACT_OVERRIDES: dict[str, dict[str, Any]] = {
    "jason-3": {
        "source": "precise_orbit",
        "provider": "NASA CDDIS DORIS_IDS_orbit_prod",
        "collection_concept_id": "C1602851234-CDDIS",
        "producer_granule_id_pattern": "*ja3*",
        "data_type": "compressed_sp3",
        "output_dir": "data/raw/reference_validation/jason-3/precise_orbit",
        "link_suffix": ".Z",
        "expected_signature": "unix_compress",
    },
}


def _cddis_contract(sat_id: str) -> dict[str, Any]:
    """Return the CDDIS DORIS SP3 contract for a satellite."""
    return {**ORBIT_WINDOW_CONTRACTS[sat_id], **CDDIS_SP3_CONTRACT_OVERRIDES.get(sat_id, {})}

# Analysis-center series preference: SSALTO/GSFC multi-technique combos first.
SERIES_PRIORITY = {"ssa": 0, "gsc": 1, "grg": 2, "lca": 3}

_SP3_SPAN_RE = re.compile(r"\.b(\d{2})(\d{3})\.e(\d{2})(\d{3})\.")
_SERIES_RE = re.compile(r"doris_products_orbits_([a-z]+)_")


def _yy_doy_to_date(year2: str, doy: str) -> date:
    """Convert a 2-digit-year + day-of-year token to a date (85 cutoff)."""
    year = int(year2)
    year += 1900 if year > 85 else 2000
    return date(year, 1, 1) + timedelta(days=int(doy) - 1)


def parse_sp3_span(name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Parse the begin/end interval from a DORIS SP3-style file name."""
    match = _SP3_SPAN_RE.search(name)
    if not match:
        return None
    begin = _yy_doy_to_date(match.group(1), match.group(2))
    end = _yy_doy_to_date(match.group(3), match.group(4))
    return pd.Timestamp(begin, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)


def _series_priority(producer_granule_id: str) -> int:
    match = _SERIES_RE.search(producer_granule_id)
    if not match:
        return len(SERIES_PRIORITY)
    return SERIES_PRIORITY.get(match.group(1), len(SERIES_PRIORITY))


def _product_version(producer_granule_id: str) -> int:
    """Extract the product version digits from a DORIS file name (ssaja120 -> 20)."""
    filename = producer_granule_id.split("_")[-1]
    match = re.match(r"[a-z]{3}[a-z0-9]{2}(\d{2})\.b", filename)
    return int(match.group(1)) if match else 0


def fetch_sp3_index(sat_id: str, pattern: str) -> pd.DataFrame:
    """Page the full CDDIS DORIS orbit collection for one satellite."""
    contract = {
        "collection_concept_id": _cddis_contract(sat_id)["collection_concept_id"],
        "producer_granule_id_pattern": pattern,
        "temporal": "",
    }
    rows: list[dict[str, Any]] = []
    page_num = 1
    while True:
        entries = list_cmr_granules(contract, page_size=500, page_num=page_num)
        for entry in entries:
            pgid = str(entry.get("producer_granule_id") or "")
            if not (pgid.endswith(".Z") and (".sp3." in pgid or ".sp1." in pgid)):
                continue
            span = parse_sp3_span(pgid)
            if span is None:
                continue
            href = next(
                (
                    link.get("href", "")
                    for link in entry.get("links", [])
                    if link.get("href", "").startswith("https://") and link.get("href", "").endswith(".Z")
                ),
                "",
            )
            if not href:
                continue
            rows.append(
                {
                    "producer_granule_id": pgid,
                    "href": href,
                    "start": span[0],
                    "end": span[1],
                    "series_priority": _series_priority(pgid),
                    "product_version": _product_version(pgid),
                    "format_rank": 0 if ".sp3." in pgid else 1,
                    "version_rank": 1 if pgid.endswith(".001.002.Z") else 0,
                }
            )
        if len(entries) < 500:
            break
        page_num += 1
        if page_num > 20:
            raise RuntimeError(f"CMR paging for {sat_id} exceeded 20 pages")
    return pd.DataFrame(rows)


def local_file_spans(output_dir: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse filename intervals of orbit files already present locally."""
    spans = []
    if not output_dir.is_dir():
        return spans
    for path in output_dir.iterdir():
        span = parse_sp3_span(path.name)
        if span is not None:
            spans.append(span)
    return spans


def load_gap_windows(sat_id: str, alignment_table: str | Path) -> pd.DataFrame:
    """Load alignment gap windows that still miss orbit evidence."""
    path = Path(alignment_table)
    if not path.is_absolute():
        path = REPO_ROOT / path
    gaps = pd.read_csv(path)
    gaps["missing_sources"] = gaps["missing_sources"].fillna("")
    gaps = gaps[(gaps["sat_id"] == sat_id) & gaps["missing_sources"].str.contains("orbit")]
    gaps["window_start"] = pd.to_datetime(gaps["window_start_utc"], utc=True)
    gaps["window_end"] = pd.to_datetime(gaps["window_end_utc"], utc=True)
    return gaps.sort_values("window_start").reset_index(drop=True)


def select_granules(index: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> list[dict[str, Any]]:
    """Rank all granules overlapping a window; CMR links often 404 for older
    product versions, so callers should try candidates in order."""
    overlapping = index[(index["start"] < window_end) & (index["end"] > window_start)]
    if overlapping.empty:
        return []
    ranked = overlapping.sort_values(
        ["series_priority", "format_rank", "product_version", "version_rank", "producer_granule_id"],
        ascending=[True, True, False, True, True],
    )
    return ranked.to_dict("records")


def select_granule(index: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> dict[str, Any] | None:
    """Pick the best-indexed granule overlapping a window."""
    candidates = select_granules(index, window_start, window_end)
    return candidates[0] if candidates else None


def backfill_satellite(
    sat_id: str,
    alignment_table: str | Path,
    execute: bool = False,
    max_windows: int | None = None,
) -> dict[str, Any]:
    """Download one orbit file per uncovered gap window for one satellite."""
    contract = _cddis_contract(sat_id)
    output_dir = resolve_repo_path(contract["output_dir"])
    gaps = load_gap_windows(sat_id, alignment_table)
    local_spans = local_file_spans(output_dir)
    uncovered = [
        row
        for row in gaps.itertuples(index=False)
        if not any(start < row.window_end and end > row.window_start for start, end in local_spans)
    ]
    if max_windows is not None:
        uncovered = uncovered[:max_windows]
    print(f"{sat_id}: {len(gaps)} gap windows, {len(uncovered)} without local coverage", flush=True)
    if not uncovered:
        return {"sat_id": sat_id, "downloaded": 0, "no_granule": 0, "failed": 0}

    index = fetch_sp3_index(sat_id, CDDIS_SP3_SAT_PATTERNS[sat_id])
    print(f"{sat_id}: CMR index holds {len(index)} usable SP3 granules", flush=True)

    downloaded = no_granule = failed = 0
    total = len(uncovered)
    for position, row in enumerate(uncovered, start=1):
        candidates = select_granules(index, row.window_start, row.window_end)
        if not candidates:
            no_granule += 1
            print(f"[{position}/{total}] {row.window_start_utc[:10]} no_granule", flush=True)
            continue
        candidates = [
            c
            for c in candidates
            if not (output_dir / sanitize_remote_filename(c["href"].rsplit("/", 1)[-1])).exists()
        ]
        if not candidates:
            print(f"[{position}/{total}] {row.window_start_utc[:10]} already_local", flush=True)
            continue
        print(
            f"[{position}/{total}] {row.window_start_utc[:10]} -> "
            f"{candidates[0]['href'].rsplit('/', 1)[-1]} (+{len(candidates) - 1} alternates)",
            flush=True,
        )
        if not execute:
            continue
        ok = False
        for granule in candidates:
            for candidate in _candidate_download_urls(granule["href"]):
                target = output_dir / sanitize_remote_filename(candidate.rsplit("/", 1)[-1])
                assert_path_within(target, output_dir)
                try:
                    path = Path(download_https_earthdata_file(candidate, target))
                    validation_error = _validate_download(path, contract["expected_signature"])
                    if validation_error:
                        path.unlink(missing_ok=True)
                        print(f"    validation failed: {validation_error}", flush=True)
                        continue
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001 - keep backfill moving
                    print(f"    download error: {exc}", flush=True)
            if ok:
                break
        if ok:
            downloaded += 1
        else:
            failed += 1
    return {"sat_id": sat_id, "downloaded": downloaded, "no_granule": no_granule, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-download CDDIS DORIS SP3 files for orbit gap windows")
    parser.add_argument("--alignment-table", default=str(DEFAULT_ALIGNMENT_TABLE))
    parser.add_argument("--include", action="append", default=None, help="sat_id filter (repeatable)")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    sat_ids = args.include or list(CDDIS_SP3_SAT_PATTERNS)
    summary = []
    for sat_id in sat_ids:
        if sat_id not in CDDIS_SP3_SAT_PATTERNS:
            print(f"{sat_id}: no CDDIS SP3 contract, skipped", flush=True)
            continue
        summary.append(backfill_satellite(sat_id, args.alignment_table, execute=args.execute, max_windows=args.max_windows))
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    main()
