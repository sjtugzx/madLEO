"""Download small official orbit-source samples for reference targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from downloaders.download_utils import load_repo_env, retry_with_backoff, temp_path_for
from downloaders.net_guard import assert_allowed_url, assert_path_within, guarded_request, sanitize_remote_filename


CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
DEFAULT_METADATA_ROOT = REPO_ROOT / "data" / "validation" / "download_status" / "benchmark"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
GZIP_SIGNATURE = b"\x1f\x8b"

SAMPLE_SOURCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "topex-poseidon": {
        "source": "precise_orbit",
        "provider": "NASA CDDIS DORIS_IDS_orbit_prod",
        "collection_concept_id": "C1602851234-CDDIS",
        "temporal": "1992-09-20T00:00:00Z,1992-10-20T00:00:00Z",
        "producer_granule_id_pattern": "*top*",
        "data_type": "compressed_sp3",
        "output_dir": "data/raw/reference_validation/topex-poseidon/precise_orbit",
        "link_suffix": ".Z",
        "expected_signature": "unix_compress",
    },
    "swot": {
        "source": "precise_orbit",
        "provider": "NASA PO.DAAC SWOT_POE_2.0",
        "collection_concept_id": "C2799438359-POCLOUD",
        "temporal": "2023-01-14T00:00:00Z,2023-01-17T00:00:00Z",
        "data_type": "netcdf_precise_orbit",
        "output_dir": "data/raw/reference_validation/swot/precise_orbit",
        "link_suffix": ".nc",
        "expected_signature": "hdf5",
        "fallback_contracts": [
            {
                "provider": "NASA PO.DAAC SWOT_MOE_1.0",
                "collection_concept_id": "C2296989401-POCLOUD",
                "data_type": "netcdf_medium_orbit",
            }
        ],
    },
    "jason-1": {
        "source": "pod",
        "provider": "NASA PO.DAAC JASON-1_L2_OST_GPN_E",
        "collection_concept_id": "C1940470304-POCLOUD",
        "temporal": "2002-01-15T06:00:00Z,2002-01-15T08:00:00Z",
        "data_type": "netcdf_gdr_orbit_related",
        "output_dir": "data/raw/reference_validation/jason-1/pod",
        "link_suffix": ".nc",
        "expected_signature": "hdf5",
        "notes": "GDR orbit-related sample; not a full mission precise-orbit SP3 backfill.",
    },
    "jason-2": {
        "source": "precise_orbit",
        "provider": "NASA CDDIS DORIS_IDS_orbit_prod",
        "collection_concept_id": "C1602851234-CDDIS",
        "temporal": "2008-07-01T00:00:00Z,2008-07-20T00:00:00Z",
        "producer_granule_id_pattern": "*ja2*",
        "data_type": "compressed_sp3",
        "output_dir": "data/raw/reference_validation/jason-2/precise_orbit",
        "link_suffix": ".Z",
        "expected_signature": "unix_compress",
    },
    "sentinel-6a": {
        "source": "pod",
        "provider": "NASA PO.DAAC JASON_CS_S6A_L1B_GNSS_POD_DAILY",
        "collection_concept_id": "C1968980591-POCLOUD",
        "temporal": "2020-11-26T00:00:00Z,2020-12-10T00:00:00Z",
        "data_type": "gnss_pod_rinex_daily",
        "output_dir": "data/raw/reference_validation/sentinel-6a/pod",
        "link_suffix": ".rnx.gz",
        "expected_signature": "gzip",
        "notes": "GNSS-POD tracking RINEX sample; not a direct precise-orbit SP3 product.",
    },
    "hy-2a": {
        "source": "precise_orbit",
        "provider": "NASA CDDIS DORIS_IDS_orbit_prod",
        "collection_concept_id": "C1602851234-CDDIS",
        "temporal": "2012-01-01T00:00:00Z,2012-01-20T00:00:00Z",
        "producer_granule_id_pattern": "*h2a*",
        "data_type": "compressed_sp3",
        "output_dir": "data/raw/reference_validation/hy-2a/precise_orbit",
        "link_suffix": ".Z",
        "expected_signature": "unix_compress",
    },
    "cryosat-2": {
        "source": "precise_orbit",
        "provider": "NASA CDDIS DORIS_IDS_orbit_prod",
        "collection_concept_id": "C1602851234-CDDIS",
        "temporal": "2010-05-25T00:00:00Z,2010-06-10T00:00:00Z",
        "producer_granule_id_pattern": "*cs2*",
        "data_type": "compressed_sp3",
        "output_dir": "data/raw/reference_validation/cryosat-2/precise_orbit",
        "link_suffix": ".Z",
        "expected_signature": "unix_compress",
    },
    "saral": {
        "source": "precise_orbit",
        "provider": "NASA CDDIS DORIS_IDS_orbit_prod",
        "collection_concept_id": "C1602851234-CDDIS",
        "temporal": "2013-03-10T00:00:00Z,2013-03-25T00:00:00Z",
        "producer_granule_id_pattern": "*srl*",
        "data_type": "compressed_sp3",
        "output_dir": "data/raw/reference_validation/saral/precise_orbit",
        "link_suffix": ".Z",
        "expected_signature": "unix_compress",
    },
}


def _looks_like_html(path: str | Path) -> bool:
    try:
        prefix = Path(path).read_bytes()[:256].lower().lstrip()
    except OSError:
        return False
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _validate_download(path: str | Path, expected_signature: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size <= 0:
        return "empty_or_missing_file"
    if _looks_like_html(file_path):
        return "html_redirect_or_login_page"
    if expected_signature == "hdf5" and file_path.read_bytes()[: len(HDF5_SIGNATURE)] != HDF5_SIGNATURE:
        return "not_hdf5_netcdf"
    if expected_signature == "unix_compress" and file_path.read_bytes()[:2] != b"\x1f\x9d":
        return "not_unix_compress"
    if expected_signature == "gzip" and file_path.read_bytes()[: len(GZIP_SIGNATURE)] != GZIP_SIGNATURE:
        return "not_gzip"
    return None


def list_cmr_granules(contract: dict[str, Any], page_size: int = 1, page_num: int | None = None) -> list[dict[str, Any]]:
    """List candidate CMR granules for one sample source contract."""
    params = {
        "collection_concept_id": contract["collection_concept_id"],
        "page_size": str(page_size),
        "sort_key": "+start_date",
    }
    if contract.get("temporal"):
        params["temporal"] = contract["temporal"]
    if page_num is not None:
        params["page_num"] = str(page_num)
    if contract.get("producer_granule_id_pattern"):
        params["producer_granule_id"] = contract["producer_granule_id_pattern"]
        params["options[producer_granule_id][pattern]"] = "true"
    response = retry_with_backoff(
        lambda: guarded_request("GET", CMR_GRANULES_URL, params=params, timeout=_cmr_request_timeout_seconds()),
        attempts=4,
        base_delay_sec=1.0,
        max_delay_sec=8.0,
        label="CMR granule listing",
        verbose=False,
    )
    response.raise_for_status()
    return response.json().get("feed", {}).get("entry", [])


def _cmr_request_timeout_seconds() -> float:
    """Return the per-request CMR timeout, with a safe environment override."""
    raw_value = os.environ.get("CMR_REQUEST_TIMEOUT_SECONDS", "60")
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        return 60.0


def _pick_https_data_link(granule: dict[str, Any], suffix: str) -> str | None:
    for link in granule.get("links", []):
        href = link.get("href", "")
        if href.startswith("https://") and href.endswith(suffix):
            return href
    return None


def _candidate_download_urls(url: str) -> list[str]:
    """Return equivalent official URLs to try for known CDDIS path migrations."""
    urls = [url]
    if "https://cddis.nasa.gov/archive/pub/doris/" in url:
        urls.append(url.replace("https://cddis.nasa.gov/archive/pub/doris/", "https://cddis.nasa.gov/archive/doris/"))
    if "https://cddis.nasa.gov/archive/pub/misc/" in url:
        urls.append(url.replace("https://cddis.nasa.gov/archive/pub/misc/", "https://cddis.nasa.gov/archive/misc/"))
    return urls


def _earthdata_netrc() -> tuple[str, str]:
    load_repo_env()
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set")
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as netrc_file:
        netrc_file.write(f"machine urs.earthdata.nasa.gov login {username} password {password}\n")
        netrc_path = netrc_file.name
    os.chmod(netrc_path, 0o600)
    cookie_path = str(Path(tempfile.gettempdir()) / "earthdata_urs_cookies_reference_orbits.txt")
    return netrc_path, cookie_path


def _earthdata_curl_max_time_seconds() -> str:
    """Return bounded curl max-time for Earthdata downloads."""
    raw_value = os.environ.get("EARTHDATA_CURL_MAX_TIME_SECONDS", "180")
    try:
        return str(max(1, int(float(raw_value))))
    except ValueError:
        return "180"


def _earthdata_curl_speed_options() -> list[str]:
    """Return optional curl low-speed abort options for screening slow links."""
    raw_value = os.environ.get("EARTHDATA_CURL_SPEED_TIME_SECONDS", "").strip()
    if not raw_value:
        return []
    try:
        speed_time = str(max(1, int(float(raw_value))))
    except ValueError:
        return []
    return ["--speed-limit", "1", "--speed-time", speed_time]


def download_https_earthdata_file(url: str, output_path: str | Path) -> Path:
    """Download one Earthdata-protected HTTPS URL with validation-safe temp writes."""
    url = assert_allowed_url(url)
    destination = Path(output_path)
    ensure_directory(destination.parent)
    temp_path = Path(temp_path_for(destination))
    netrc_path, cookie_path = _earthdata_netrc()
    try:
        subprocess.run(
            [
                "curl",
                "-sS",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--retry-all-errors",
                "--continue-at",
                "-",
                *_earthdata_curl_speed_options(),
                "--netrc-file",
                netrc_path,
                "-b",
                cookie_path,
                "-c",
                cookie_path,
                "--max-redirs",
                "10",
                "--max-time",
                _earthdata_curl_max_time_seconds(),
                "-o",
                str(temp_path),
                url,
            ],
            check=True,
        )
        os.replace(temp_path, destination)
    finally:
        Path(netrc_path).unlink(missing_ok=True)
        temp_path.unlink(missing_ok=True)
    return destination


def write_metadata(metadata: dict[str, Any], metadata_root: str | Path = DEFAULT_METADATA_ROOT) -> dict[str, Any]:
    root = Path(metadata_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    metadata_dir = ensure_directory(root / metadata["sat_id"] / metadata["source"])
    (metadata_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def download_sample_for_target(
    sat_id: str,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    downloader=download_https_earthdata_file,
) -> dict[str, Any]:
    """Download one small official orbit-source sample for a configured target."""
    if sat_id not in SAMPLE_SOURCE_CONTRACTS:
        raise ValueError(f"Unsupported orbit sample target: {sat_id}")
    contract = SAMPLE_SOURCE_CONTRACTS[sat_id]
    metadata = {
        "sat_id": sat_id,
        "source": contract["source"],
        "provider": contract["provider"],
        "collection_concept_id": contract["collection_concept_id"],
        "data_type": contract["data_type"],
        "status": "planned",
        "downloaded_files": [],
        "downloaded_file_count": 0,
        "notes": contract.get("notes", ""),
    }
    granules = list_cmr_granules(contract, page_size=10)
    if not granules:
        return write_metadata({**metadata, "status": "no_cmr_granules"}, metadata_root=metadata_root)

    output_dir = resolve_repo_path(contract["output_dir"])
    errors = []
    last_source_url = ""
    last_time_start = None
    last_time_end = None
    last_validation_error = ""
    saw_download_candidate = False

    for granule in granules:
        href = _pick_https_data_link(granule, contract["link_suffix"])
        if not href:
            errors.append(f"{granule.get('producer_granule_id', 'granule')}: no_https_data_link")
            continue

        saw_download_candidate = True
        last_source_url = href
        last_time_start = granule.get("time_start")
        last_time_end = granule.get("time_end")
        for candidate_href in _candidate_download_urls(href):
            output_path = Path(output_dir) / sanitize_remote_filename(candidate_href.rsplit("/", 1)[-1])
            assert_path_within(output_path, output_dir)
            try:
                downloaded_path = downloader(candidate_href, output_path)
            except Exception as exc:
                output_path.unlink(missing_ok=True)
                Path(temp_path_for(output_path)).unlink(missing_ok=True)
                errors.append(f"{candidate_href}: {exc}")
                continue

            validation_error = _validate_download(downloaded_path, contract["expected_signature"])
            if validation_error:
                Path(downloaded_path).unlink(missing_ok=True)
                last_validation_error = validation_error
                last_source_url = candidate_href
                errors.append(f"{candidate_href}: {validation_error}")
                continue

            file_path = Path(downloaded_path)
            return write_metadata(
                {
                    **metadata,
                    "status": "downloaded",
                    "source_url": candidate_href,
                    "time_start": granule.get("time_start"),
                    "time_end": granule.get("time_end"),
                    "downloaded_files": [str(file_path)],
                    "downloaded_file_count": 1,
                    "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                },
                metadata_root=metadata_root,
            )

    if not saw_download_candidate:
        return write_metadata(
            {**metadata, "status": "no_https_data_link", "notes": " | ".join(errors)},
            metadata_root=metadata_root,
        )

    status = last_validation_error or "download_failed"
    return write_metadata(
        {
            **metadata,
            "status": status,
            "source_url": last_source_url,
            "time_start": last_time_start,
            "time_end": last_time_end,
            "notes": " | ".join(errors),
        },
        metadata_root=metadata_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download small official orbit-source samples")
    parser.add_argument("--sat-id", action="append", choices=sorted(SAMPLE_SOURCE_CONTRACTS), required=True)
    parser.add_argument("--metadata-root", default=str(DEFAULT_METADATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for sat_id in args.sat_id:
        metadata = download_sample_for_target(sat_id, metadata_root=args.metadata_root)
        print(f"{sat_id} {metadata['source']} status={metadata['status']} files={metadata['downloaded_file_count']}")


if __name__ == "__main__":
    main()
