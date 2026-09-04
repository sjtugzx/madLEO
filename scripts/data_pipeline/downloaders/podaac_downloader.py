"""PO.DAAC downloader for Jason-3 GPS-based orbit granules."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from downloaders.download_utils import atomic_output_file, cleanup_temp_file, temp_path_for, retry_with_backoff
from downloaders.net_guard import (
    assert_allowed_url,
    assert_allowed_url_args,
    assert_path_within,
    guarded_request,
    sanitize_remote_filename,
)
from processors.jason3_ogdr_processor import validate_jason3_ogdr_file


CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
JASON3_COLLECTION_ID = "C2205122298-POCLOUD"
PAGE_SIZE = 2000
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LEO-Orbit-Dataset/1.0)",
}
CURSOR_FILENAME = "_cursor.json"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
JASON3_FILENAME_RE = re.compile(
    r"_(\d{8})_(\d{6})_(\d{8})_(\d{6})\.(?:nc|nc4)$",
    re.IGNORECASE,
)


def _curl_json(args: List[str]) -> Dict:
    """Run curl and parse a JSON response."""
    assert_allowed_url_args(args)
    result = retry_with_backoff(
        lambda: subprocess.run(
            ["curl", "-sS", "--retry", "5", "--retry-all-errors", "--retry-delay", "2", *args],
            check=True,
            capture_output=True,
            text=True,
        ),
        attempts=6,
        base_delay_sec=2.0,
        max_delay_sec=20.0,
        label="PO.DAAC curl json",
        verbose=False,
    )
    return json.loads(result.stdout)


def _write_cursor(cursor_path: str, granule: Dict, output_root: str) -> None:
    """Persist the latest successfully downloaded Jason-3 granule marker.

    The cursor file must live inside the configured output root; anything
    resolving outside it is rejected as a path traversal.
    """
    assert_path_within(cursor_path, output_root)
    with open(cursor_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "last_successful_filename": granule["filename"],
                "last_successful_time_end": granule.get("time_end"),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def _cursor_datetime(value: str) -> datetime:
    """Parse cursor timestamps into naive UTC datetimes for local comparisons."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _effective_incremental_start(start_date: datetime, end_date: datetime, last_end: str | None) -> datetime | None:
    """Return a safe incremental cursor inside the requested window."""
    if not last_end:
        return start_date
    cursor_end = _cursor_datetime(last_end)
    if cursor_end > end_date:
        return None
    return max(start_date, cursor_end - timedelta(minutes=30))


def _jason3_granule_window_from_filename(filename: str) -> tuple[datetime, datetime]:
    """Extract the encoded start/end timestamps from a Jason-3 POD filename."""
    match = JASON3_FILENAME_RE.search(filename)
    if not match:
        raise ValueError(f"Unable to parse Jason-3 POD timestamps from filename: {filename}")
    start = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
    end = datetime.strptime(f"{match.group(3)}{match.group(4)}", "%Y%m%d%H%M%S")
    return start, end


def list_podaac_jason3_granules(
    start_date: datetime,
    end_date: datetime,
    collection_id: str = JASON3_COLLECTION_ID,
    verbose: bool = True,
) -> List[Dict]:
    """List Jason-3 OGDR GPS granules from CMR."""
    temporal = f"{start_date.strftime('%Y-%m-%dT00:00:00Z')},{(end_date + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')}"
    page_num = 1
    rows: List[Dict] = []

    if verbose:
        print("Querying PO.DAAC CMR for Jason-3 GPS OGDR granules...")

    while True:
        params = {
            "collection_concept_id": collection_id,
            "temporal": temporal,
            "page_size": str(PAGE_SIZE),
            "page_num": str(page_num),
            "sort_key": "+start_date",
        }
        try:
            response = retry_with_backoff(
                lambda: guarded_request("GET", CMR_GRANULES_URL, params=params, headers=HEADERS, timeout=120),
                attempts=6,
                base_delay_sec=2.0,
                max_delay_sec=20.0,
                label=f"PO.DAAC listing page={page_num}",
                verbose=False,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            curl_args = ["-G", CMR_GRANULES_URL]
            for key, value in params.items():
                curl_args.extend(["--data-urlencode", f"{key}={value}"])
            payload = _curl_json(curl_args)
        entries = payload.get("feed", {}).get("entry", [])
        if not entries:
            break
        rows.extend(entries)
        if len(entries) < PAGE_SIZE:
            break
        page_num += 1

    granules: List[Dict] = []
    for entry in rows:
        href = None
        for link in entry.get("links", []):
            candidate = link.get("href")
            if not candidate:
                continue
            if candidate.endswith(".nc") or candidate.endswith(".nc4"):
                href = candidate
                break
        if href is None:
            continue
        granules.append(
            {
                "id": entry.get("id"),
                "filename": sanitize_remote_filename(href.rstrip("/").split("/")[-1]),
                "url": href,
                "time_start": entry.get("time_start"),
                "time_end": entry.get("time_end"),
                "updated": entry.get("updated"),
                "collection_concept_id": collection_id,
            }
        )

    if verbose:
        print(f"  Found {len(granules)} granules")

    return granules


def find_podaac_jason3_granule(filename: str, verbose: bool = True) -> Dict:
    """Find a specific Jason-3 POD granule by filename via narrow CMR listing windows."""
    start, end = _jason3_granule_window_from_filename(filename)
    windows = [
        (start, end),
        (start - timedelta(days=1), end + timedelta(days=1)),
    ]
    seen_ranges: set[tuple[datetime, datetime]] = set()
    for window_start, window_end in windows:
        key = (window_start, window_end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        granules = list_podaac_jason3_granules(window_start, window_end, verbose=verbose)
        for granule in granules:
            if granule["filename"] == filename:
                return granule
    raise FileNotFoundError(f"Unable to locate Jason-3 POD granule in CMR for filename: {filename}")


def download_podaac_granule(
    granule: Dict,
    output_dir: str,
    username: str,
    password: str,
    verbose: bool = True,
) -> Optional[str]:
    """Download one PO.DAAC granule using Earthdata credentials."""
    os.makedirs(output_dir, exist_ok=True)
    filename = sanitize_remote_filename(str(granule["filename"]))
    local_path = os.path.join(output_dir, filename)
    assert_path_within(local_path, output_dir)

    if os.path.exists(local_path):
        validation_error = None
        if _looks_like_hdf5(local_path):
            validation_error = validate_jason3_ogdr_file(local_path)
        if validation_error is None and _looks_like_hdf5(local_path):
            if verbose:
                print(f"  Skipping {filename} (already exists)")
            return local_path
        if verbose:
            reason = validation_error or "invalid HDF5 signature"
            print(f"  Existing {filename} is invalid ({reason}), re-downloading")
        try:
            os.remove(local_path)
        except OSError:
            pass

    if verbose:
        print(f"  Downloading {filename}...")

    try:
        response = retry_with_backoff(
            lambda: guarded_request(
                "GET",
                granule["url"],
                auth=(username, password),
                headers=HEADERS,
                timeout=300,
                stream=True,
            ),
            attempts=6,
            base_delay_sec=2.0,
            max_delay_sec=20.0,
            label=f"PO.DAAC granule {filename}",
            verbose=False,
        )
        response.raise_for_status()
        with atomic_output_file(local_path, "wb") as (handle, _):
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
    except requests.RequestException:
        _download_with_earthdata_curl(granule["url"], local_path, username, password)
    except Exception:
        cleanup_temp_file(local_path)
        raise

    if _looks_like_html(local_path):
        os.remove(local_path)
        _download_with_earthdata_curl(granule["url"], local_path, username, password)
        if _looks_like_html(local_path):
            raise RuntimeError(f"Earthdata download returned HTML instead of data for {filename}")
    if not _looks_like_hdf5(local_path):
        try:
            os.remove(local_path)
        except OSError:
            pass
        raise RuntimeError(f"Downloaded PO.DAAC granule is not a valid HDF5/netCDF file for {filename}")
    validation_error = validate_jason3_ogdr_file(local_path)
    if validation_error is not None:
        try:
            os.remove(local_path)
        except OSError:
            pass
        raise RuntimeError(f"Downloaded PO.DAAC granule failed OGDR validation for {filename}: {validation_error}")

    return local_path


def _download_with_earthdata_curl(url: str, output_path: str, username: str, password: str) -> None:
    """Download a protected Earthdata file using netrc and cookie jar handling."""
    url = assert_allowed_url(url)
    cookie_path = os.path.join(tempfile.gettempdir(), "earthdata_urs_cookies.txt")
    temp_output_path = temp_path_for(output_path)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as netrc_file:
        netrc_file.write(f"machine urs.earthdata.nasa.gov login {username} password {password}\n")
        netrc_path = netrc_file.name
    os.chmod(netrc_path, 0o600)
    try:
        def _attempt() -> None:
            cleanup_temp_file(output_path)
            subprocess.run(
                [
                    "curl",
                    "-sS",
                    "-L",
                    "--retry",
                    "5",
                    "--retry-all-errors",
                    "--retry-delay",
                    "2",
                    "--netrc-file",
                    netrc_path,
                    "-b",
                    cookie_path,
                    "-c",
                    cookie_path,
                    "-o",
                    temp_output_path,
                    url,
                ],
                check=True,
            )
            if not os.path.exists(temp_output_path):
                if os.path.exists(output_path) and _looks_like_hdf5(output_path):
                    return
                raise RuntimeError(f"Missing temp download file after curl: {temp_output_path}")
            try:
                os.replace(temp_output_path, output_path)
            except FileNotFoundError:
                if os.path.exists(output_path) and _looks_like_hdf5(output_path):
                    return
                raise

        retry_with_backoff(
            _attempt,
            attempts=6,
            base_delay_sec=2.0,
            max_delay_sec=20.0,
            label=f"PO.DAAC curl download {output_path}",
            verbose=False,
        )
    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        try:
            os.remove(netrc_path)
        except OSError:
            pass


def _looks_like_html(path: str) -> bool:
    """Cheap check for accidental HTML login pages saved as data files."""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(256).lower()
    except OSError:
        return False
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _looks_like_hdf5(path: str) -> bool:
    """Cheap validation for netCDF4/HDF5 granules."""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(len(HDF5_SIGNATURE))
    except OSError:
        return False
    return prefix == HDF5_SIGNATURE


def download_podaac_jason3_ogdr_gps(
    start_date: datetime,
    end_date: datetime,
    output_dir: str,
    max_files: int | None = None,
    incremental: bool = True,
    verbose: bool = True,
) -> List[str]:
    """Download Jason-3 OGDR GPS granules from PO.DAAC."""
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set for PO.DAAC downloads")

    os.makedirs(output_dir, exist_ok=True)
    effective_start = start_date
    cursor_path = os.path.join(output_dir, CURSOR_FILENAME)
    if incremental and os.path.exists(cursor_path):
        try:
            cursor = json.load(open(cursor_path, "r", encoding="utf-8"))
            last_end = cursor.get("last_successful_time_end")
            effective_start = _effective_incremental_start(start_date, end_date, last_end)
            if effective_start is None:
                if verbose:
                    print(f"Local Jason-3 cursor is already beyond {end_date.isoformat()}, skipping listing for this window")
                return []
            if last_end and effective_start:
                if verbose:
                    print(f"Resuming Jason-3 POD listing from {effective_start.isoformat()} based on local cursor")
        except Exception as exc:
            if verbose:
                print(f"Warning: failed to load Jason-3 cursor: {exc}")

    granules = list_podaac_jason3_granules(effective_start, end_date, verbose=verbose)
    if max_files is not None:
        granules = granules[:max_files]

    downloaded = []
    last_successful = None
    for granule in granules:
        local_path = download_podaac_granule(granule, output_dir, username, password, verbose=verbose)
        if local_path:
            downloaded.append(local_path)
            last_successful = granule
            _write_cursor(cursor_path, granule, output_root=str(output_dir))

    if verbose:
        print(f"Downloaded {len(downloaded)} files")

    if last_successful is not None:
        _write_cursor(cursor_path, last_successful, output_root=str(output_dir))

    return downloaded
