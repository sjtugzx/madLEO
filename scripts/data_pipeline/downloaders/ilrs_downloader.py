"""
ILRS (International Laser Ranging Service) SLR Data Downloader

Downloads Satellite Laser Ranging data from:
- CDDIS (Crustal Dynamics Data Information System)
- EDC (EUROLAS Data Center)

Transport note: the EDC mirror is reached via explicit FTPS (AUTH TLS + PROT P,
``ftplib.FTP_TLS`` with ``ssl.create_default_context()``). Plaintext
``ftplib.FTP`` is no longer used anywhere; historical plaintext endpoints can
only be reached through guarded curl subprocesses with
``MADLEO_ALLOW_INSECURE=1``.

Data formats:
- CRD (Consolidated laser Ranging Data) - raw data
- NPT (Normal Points) - condensed data
"""

import os
import ftplib
import ssl
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import gzip
import shutil
from pathlib import Path
from downloaders.download_utils import atomic_output_file, cleanup_temp_file, retry_with_backoff, temp_path_for
from downloaders.net_guard import (
    assert_allowed_url,
    assert_allowed_url_args,
    assert_path_within,
    guarded_request,
    sanitize_remote_filename,
)


# CDDIS FTP configuration (requires NASA Earthdata login for new data)
CDDIS_FTP_HOST = 'cddis.nasa.gov'
CDDIS_SLR_PATH = '/archive/slr'
CDDIS_HTTPS_BASE = 'https://cddis.nasa.gov/archive/slr/data'
CDDIS_FTPS_HOST = 'gdc.cddis.eosdis.nasa.gov'
CDDIS_FTPS_TIMEOUT = 8
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
CDDIS_NPT_DAILY_COLLECTION_ID = "C1537476328-CDDIS"

# EDC FTPS configuration (alternative, no auth required)
EDC_FTP_HOST = 'edc.dgfi.tum.de'
EDC_SLR_PATH = '/pub/slr'
EDC_FTP_TIMEOUT = 30


def _edc_ftps_client() -> ftplib.FTP_TLS:
    """Return an explicit-TLS FTPS client for the EDC mirror (no connection made).

    The EDC transport was upgraded from plaintext FTP to explicit FTPS;
    callers connect, login anonymously, and call ``prot_p()`` to protect the
    data channel.
    """
    return ftplib.FTP_TLS(context=ssl.create_default_context())

# Target satellite ILRS IDs
ILRS_TARGETS = {
    'sentinel-3a': {
        'cospar': '1601101',
        'sic': '8811',
        'norad': '41335',
        'name': 'sentinel3a'
    },
    'sentinel-3b': {
        'cospar': '1803901',
        'sic': '8812',
        'norad': '43437',
        'name': 'sentinel3b'
    },
    'jason-3': {
        'cospar': '1600201',
        'sic': '8808',
        'norad': '41240',
        'name': 'jason3'
    },
    'sentinel-6a': {
        'cospar': '2008601',
        'sic': '4380',
        'norad': '46984',
        'name': 'sentinel6a'
    },
    'cryosat-2': {
        'cospar': '1001301',
        'sic': '8006',
        'norad': '36508',
        'name': 'cryosat2'
    },
    'saral': {
        'cospar': '1300901',
        'sic': '3201',
        'norad': '39086',
        'name': 'saral'
    },
    'jason-1': {
        'cospar': '0105501',
        'sic': '4378',
        'norad': '26997',
        'name': 'jas1',
        'edc_name': 'jason1'
    },
    'jason-2': {
        'cospar': '0803201',
        'sic': '1025',
        'norad': '33105',
        'name': 'jas2',
        'edc_name': 'jason2'
    },
    'topex-poseidon': {
        'cospar': '9205201',
        'sic': '4377',
        'norad': '22076',
        'name': 'topx',
        'edc_name': 'topex'
    },
    'hy-2a': {
        'cospar': '1104301',
        'sic': '2201',
        'norad': '37781',
        'name': 'hy2a'
    },
    'swot': {
        'cospar': '2217301',
        'sic': '4381',
        'norad': '54754',
        'name': 'swot'
    },
    'swarm-a': {
        'cospar': '1306701',
        'sic': '8307',
        'norad': '39452',
        'name': 'swarma'
    },
    'swarm-b': {
        'cospar': '1306702',
        'sic': '8308',
        'norad': '39451',
        'name': 'swarmb'
    },
    'swarm-c': {
        'cospar': '1306703',
        'sic': '8309',
        'norad': '39453',
        'name': 'swarmc'
    },
    'grace-fo-1': {
        'cospar': '1804701',
        'sic': '8813',
        'norad': '43476',
        'name': 'gracefo1'
    },
    'grace-fo-2': {
        'cospar': '1804702',
        'sic': '8814',
        'norad': '43477',
        'name': 'gracefo2'
    },
}


def _file_in_date_range(filename: str,
                        start_date: datetime = None,
                        end_date: datetime = None) -> bool:
    """
    Check whether an EDC filename falls inside the requested date range.

    Supported patterns:
    - name_YYYYMMDD.ext
    - name_YYYYMM.ext
    - legacy EDC name.YYYYMMDD and name.YYYYMM.ext
    """
    match = re.search(r'[_.](\d{8}|\d{6})(?:\D|$)', filename)
    if not match:
        return False

    token = match.group(1)
    try:
        if len(token) == 8:
            file_start = datetime.strptime(token, '%Y%m%d')
            file_end = file_start + timedelta(days=1) - timedelta(microseconds=1)
        else:
            file_start = datetime.strptime(token, '%Y%m')
            if file_start.month == 12:
                next_month = datetime(file_start.year + 1, 1, 1)
            else:
                next_month = datetime(file_start.year, file_start.month + 1, 1)
            file_end = next_month - timedelta(microseconds=1)
    except ValueError:
        return False

    if start_date and file_end < start_date:
        return False
    if end_date and file_start > end_date:
        return False
    return True


def _edc_data_dirs(target_name: str, data_type: str) -> List[str]:
    if data_type == 'npt':
        return [
            f"{EDC_SLR_PATH}/data/npt_crd/{target_name}",
            f"{EDC_SLR_PATH}/data/npt_crd_v2/{target_name}",
            f"{EDC_SLR_PATH}/data/npt/{target_name}",
        ]
    if data_type in {'crd', 'frd'}:
        return [
            f"{EDC_SLR_PATH}/data/fr_crd/{target_name}",
            f"{EDC_SLR_PATH}/data/fr/{target_name}",
        ]
    raise ValueError("data_type must be 'npt', 'crd', or 'frd'")


def list_edc_files(satellite: str,
                   data_type: str = 'npt',
                   start_date: datetime = None,
                   end_date: datetime = None,
                   verbose: bool = True) -> List[str]:
    """
    List available SLR files on EDC FTP server

    Args:
        satellite: Satellite name
        data_type: 'npt' for normal points, 'crd' for full rate
        start_date: Optional start date
        end_date: Optional end date
        verbose: Print progress

    Returns:
        List of file paths on FTP
    """
    if satellite.lower() not in ILRS_TARGETS:
        raise ValueError(f"Unknown satellite: {satellite}. Available: {list(ILRS_TARGETS.keys())}")

    target_info = ILRS_TARGETS[satellite.lower()]
    available_files = []

    try:
        if verbose:
            print(f"Connecting to EDC FTP...")

        assert_allowed_url(f"ftps://{EDC_FTP_HOST}/")
        ftp = _edc_ftps_client()
        ftp.connect(EDC_FTP_HOST, timeout=EDC_FTP_TIMEOUT)
        ftp.login()
        ftp.prot_p()

        target_name = target_info.get('edc_name', target_info['name']).lower()
        base_candidates = _edc_data_dirs(target_name, data_type)

        for candidate in base_candidates:
            try:
                ftp.cwd(candidate)
            except ftplib.error_perm:
                continue
            except Exception:
                continue
            years = ftp.nlst()
            for year in years:
                try:
                    year_int = int(year)
                except ValueError:
                    continue

                if start_date and year_int < start_date.year:
                    continue
                if end_date and year_int > end_date.year:
                    continue

                year_path = f"{candidate}/{year}"
                try:
                    ftp.cwd(year_path)
                    files = ftp.nlst()
                except ftplib.error_perm:
                    continue
                except Exception:
                    continue

                for f in files:
                    if (f.endswith('.npt') or f.endswith('.np2') or f.endswith('.crd') or f.endswith('.frd') or f.endswith('.gz') or re.search(r'\.\d{8}(?:\D|$)', f)) and \
                            _file_in_date_range(f, start_date, end_date):
                        available_files.append(f"{year_path}/{f}")

        ftp.quit()

    except Exception as e:
        print(f"FTP error: {e}")

    if verbose:
        print(f"Found {len(available_files)} files for {satellite}")

    return sorted(set(available_files))


def download_edc_file(remote_path: str,
                      output_dir: str,
                      decompress: bool = True,
                      verbose: bool = True) -> Optional[str]:
    """
    Download a single file from EDC FTP

    Args:
        remote_path: Full path on FTP
        output_dir: Local output directory
        decompress: Decompress .gz files
        verbose: Print progress

    Returns:
        Local file path or None
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = sanitize_remote_filename(remote_path)
    local_path = os.path.join(output_dir, filename)
    assert_path_within(local_path, output_dir)

    # Skip if exists
    decompressed_name = filename[:-3] if filename.endswith('.gz') else filename
    if os.path.exists(os.path.join(output_dir, decompressed_name)):
        if verbose:
            print(f"  Skipping {filename} (already exists)")
        return os.path.join(output_dir, decompressed_name)

    try:
        assert_allowed_url(f"ftps://{EDC_FTP_HOST}/")
        ftp = _edc_ftps_client()
        ftp.connect(EDC_FTP_HOST, timeout=EDC_FTP_TIMEOUT)
        ftp.login()
        ftp.prot_p()

        if verbose:
            print(f"  Downloading {filename}...")

        with atomic_output_file(local_path, 'wb') as (f, _):
            ftp.retrbinary(f'RETR {remote_path}', f.write)

        ftp.quit()

        # Decompress if needed
        if decompress and local_path.endswith('.gz'):
            decompressed_path = local_path[:-3]
            with gzip.open(local_path, 'rb') as f_in:
                with atomic_output_file(decompressed_path, 'wb') as (f_out, _):
                    shutil.copyfileobj(f_in, f_out)
            os.remove(local_path)
            local_path = decompressed_path

        return local_path

    except Exception as e:
        print(f"  Download failed: {e}")
        cleanup_temp_file(local_path)
        return None


def download_ilrs_data(satellite: str,
                       start_date: datetime,
                       end_date: datetime,
                       output_dir: str,
                       data_type: str = 'npt',
                       max_files: int = None,
                       include_cddis_supplement: bool = True,
                       verbose: bool = True) -> List[str]:
    """
    Download ILRS SLR data for a satellite

    Args:
        satellite: Satellite name
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        data_type: 'npt' or 'crd'
        max_files: Maximum files to download
        include_cddis_supplement: Also query CDDIS CMR after EDC NPT downloads
        verbose: Print progress

    Returns:
        List of downloaded file paths
    """
    if verbose:
        print(f"Downloading ILRS {data_type.upper()} data for {satellite}")
        print(f"  Period: {start_date.date()} to {end_date.date()}")

    # List available files
    files = list_edc_files(satellite, data_type, start_date, end_date, verbose=verbose)

    downloaded = []
    for remote_path in files:
        filename = sanitize_remote_filename(remote_path)
        final_name = filename[:-3] if filename.endswith('.gz') else filename
        final_path = os.path.join(output_dir, final_name)
        if os.path.exists(final_path):
            if verbose:
                print(f"  Skipping {final_name} (already exists)")
            continue
        local_path = download_edc_file(remote_path, output_dir, verbose=verbose)
        if local_path:
            downloaded.append(local_path)
            if max_files is not None and len(downloaded) >= max_files:
                break

    if data_type == 'npt' and include_cddis_supplement:
        try:
            cddis_downloaded = download_cddis_cmr_granules(
                satellite=satellite,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                max_files=max_files,
                verbose=verbose,
            )
            downloaded.extend(path for path in cddis_downloaded if path not in downloaded)
        except Exception as exc:
            if verbose:
                print(f"CDDIS CMR supplement skipped: {exc}")
    elif data_type == 'npt' and verbose:
        print("CDDIS CMR supplement skipped by request")
    elif verbose:
        print("CDDIS CMR supplement skipped for non-NPT EDC download")

    if verbose:
        print(f"Downloaded {len(downloaded)} files")

    return downloaded


def download_cddis_data(satellite: str,
                        start_date: datetime,
                        end_date: datetime,
                        output_dir: str,
                        earthdata_token: str = None,
                        data_type: str = 'npt',
                        max_files: int = None,
                        verbose: bool = True) -> List[str]:
    """
    Download ILRS data from CDDIS (requires NASA Earthdata login)

    Args:
        satellite: Satellite name
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        earthdata_token: NASA Earthdata bearer token
        data_type: 'npt' or 'crd'
        verbose: Print progress

    Returns:
        List of downloaded file paths
    """
    if satellite.lower() not in ILRS_TARGETS:
        raise ValueError(f"Unknown satellite: {satellite}")

    available = list_cddis_files(
        satellite=satellite,
        data_type=data_type,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
    )
    if max_files is not None:
        available = available[:max_files]

    downloaded = []
    for entry in available:
        local_path = download_cddis_file(entry["url"], output_dir, verbose=verbose)
        if local_path:
            downloaded.append(local_path)

    if verbose:
        print(f"Downloaded {len(downloaded)} CDDIS files")

    return downloaded


def _earthdata_credentials() -> tuple[str, str]:
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set for CDDIS access")
    return username, password


def _looks_like_html_redirect(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(256).lower().lstrip()
    except OSError:
        return False
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _non_negative_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _curl_earthdata(url: str, write_to: str = None) -> str:
    url = assert_allowed_url(url)
    username, password = _earthdata_credentials()
    max_time = _positive_int_env("CDDIS_CURL_MAX_TIME_SECONDS", 300)
    retry_attempts = _positive_int_env("CDDIS_CURL_RETRY_ATTEMPTS", 6)
    curl_retry_count = _non_negative_int_env("CDDIS_CURL_RETRY_COUNT", 5)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as netrc_file:
        netrc_file.write(f"machine urs.earthdata.nasa.gov login {username} password {password}\n")
        netrc_path = netrc_file.name
    Path(netrc_path).chmod(0o600)
    cookie_path = tempfile.NamedTemporaryFile(delete=False).name
    temp_output_path = temp_path_for(write_to) if write_to else None
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--http1.1",
        "--retry",
        str(curl_retry_count),
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        str(max_time),
        "--netrc-file",
        netrc_path,
        "-b",
        cookie_path,
        "-c",
        cookie_path,
        url,
    ]
    assert_allowed_url_args(cmd)
    if write_to:
        cleanup_temp_file(write_to)
        cmd.extend(["-o", temp_output_path])
    def _run() -> subprocess.CompletedProcess:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"curl failed with exit code {result.returncode}: {result.stderr.strip()}"
            )
        if write_to:
            if not temp_output_path or not os.path.exists(temp_output_path):
                raise RuntimeError(f"Missing temp download file after curl: {temp_output_path}")
            if _looks_like_html_redirect(temp_output_path):
                os.remove(temp_output_path)
                raise RuntimeError(f"CDDIS download returned an HTML redirect/login page for {url}")
            os.replace(temp_output_path, write_to)
        return result

    try:
        result = retry_with_backoff(
            _run,
            attempts=retry_attempts,
            base_delay_sec=2.0,
            max_delay_sec=20.0,
            label=f"Earthdata fetch {url}",
            verbose=False,
        )
    finally:
        try:
            os.remove(netrc_path)
        except OSError:
            pass
        try:
            os.remove(cookie_path)
        except OSError:
            pass
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    return result.stdout if write_to is None else ""


def _earthdata_head(url: str) -> tuple[str, str]:
    url = assert_allowed_url(url)
    username, password = _earthdata_credentials()
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as netrc_file:
        netrc_file.write(f"machine urs.earthdata.nasa.gov login {username} password {password}\n")
        netrc_path = netrc_file.name
    Path(netrc_path).chmod(0o600)
    cookie_path = tempfile.NamedTemporaryFile(delete=False).name
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--max-time",
        "20",
        "--netrc-file",
        netrc_path,
        "-b",
        cookie_path,
        "-c",
        cookie_path,
        "-o",
        "/dev/null",
        "-w",
        "%{http_code} %{url_effective}",
        url,
    ]
    assert_allowed_url_args(cmd)
    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)

    try:
        result = retry_with_backoff(
            _run,
            attempts=6,
            base_delay_sec=2.0,
            max_delay_sec=20.0,
            label=f"Earthdata head {url}",
            verbose=False,
        )
    finally:
        try:
            os.remove(netrc_path)
        except OSError:
            pass
        try:
            os.remove(cookie_path)
        except OSError:
            pass
    code, final_url = result.stdout.strip().split(" ", 1)
    return code, final_url


def _cddis_download_url_variants(url: str) -> list[str]:
    migrated = url.replace("https://cddis.nasa.gov/archive/pub/slr/", "https://cddis.nasa.gov/archive/slr/")
    if migrated != url:
        return [migrated, url]
    return [url]


def _candidate_tokens(start_date: datetime,
                      end_date: datetime,
                      granularity: str = "both") -> List[tuple[str, str]]:
    tokens = []
    monthly_seen = set()
    daily_seen = set()
    if granularity in {"both", "monthly"}:
        current = datetime(start_date.year, start_date.month, 1)
        while current <= end_date:
            monthly_token = current.strftime("%Y%m")
            if monthly_token not in monthly_seen:
                monthly_seen.add(monthly_token)
                tokens.append(("monthly", monthly_token))
            current += timedelta(days=1)

    if granularity in {"both", "daily"}:
        current = start_date
        while current <= end_date:
            daily_token = current.strftime("%Y%m%d")
            if daily_token not in daily_seen:
                daily_seen.add(daily_token)
                tokens.append(("daily", daily_token))
            current += timedelta(days=1)
    return tokens


def _allsat_hourly_tokens(start_date: datetime, end_date: datetime) -> List[str]:
    tokens = []
    seen = set()
    current = start_date.replace(minute=0, second=0, microsecond=0)
    while current <= end_date:
        token = current.strftime("%Y%m%d%H%M")
        if token not in seen:
            seen.add(token)
            tokens.append(token)
        current += timedelta(hours=1)
    return tokens


def _cddis_ftps_password() -> str:
    for key in ["CDDIS_FTPS_EMAIL", "EARTHDATA_USERNAME"]:
        value = os.environ.get(key)
        if value:
            return value
    return "anonymous@example.com"


def _cddis_ftps_dirs(target_name: str, year: int, data_type: str) -> List[str]:
    roots = [f"/archive/slr/data/{data_type}_crd", f"/archive/slr/data/{data_type}_crd_v2"]
    dirs = []
    for root in roots:
        dirs.append(f"{root}/{year}/{target_name}")
        dirs.append(f"{root}/{target_name}/{year}")
    return dirs


def list_cddis_ftps_files(satellite: str,
                          data_type: str = 'npt',
                          start_date: datetime = None,
                          end_date: datetime = None,
                          verbose: bool = True) -> List[Dict]:
    if satellite.lower() not in ILRS_TARGETS:
        raise ValueError(f"Unknown satellite: {satellite}")
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required for CDDIS FTPS probing")

    target_name = ILRS_TARGETS[satellite.lower()]['name']
    password = _cddis_ftps_password()
    available: List[Dict] = []

    for year in range(start_date.year, end_date.year + 1):
        for directory in _cddis_ftps_dirs(target_name, year, data_type):
            ftps = ftplib.FTP_TLS()
            try:
                assert_allowed_url(f"ftps://{CDDIS_FTPS_HOST}/")
                ftps.connect(CDDIS_FTPS_HOST, 21, timeout=CDDIS_FTPS_TIMEOUT)
                ftps.login('anonymous', password)
                ftps.prot_p()
                ftps.cwd(directory)
                files = ftps.nlst()
                for filename in files:
                    name = os.path.basename(filename)
                    if _file_in_date_range(name, start_date, end_date):
                        available.append(
                            {
                                "satellite": satellite.lower(),
                                "filename": name,
                                "url": f"ftps://{CDDIS_FTPS_HOST}{directory}/{name}",
                                "directory": directory,
                                "transport": "ftps",
                            }
                        )
                if files and verbose:
                    print(f"  FTPS listing succeeded: {directory} ({len(files)} entries)")
            except Exception:
                continue
            finally:
                try:
                    ftps.quit()
                except Exception:
                    pass

    dedup = {(row["directory"], row["filename"]): row for row in available}
    rows = list(dedup.values())
    if verbose:
        print(f"Found {len(rows)} CDDIS FTPS candidate files for {satellite}")
    return rows


def list_cddis_cmr_granules(satellite: str,
                            start_date: datetime,
                            end_date: datetime,
                            collection_concept_id: str = CDDIS_NPT_DAILY_COLLECTION_ID,
                            verbose: bool = True) -> List[Dict]:
    if satellite.lower() not in ILRS_TARGETS:
        raise ValueError(f"Unknown satellite: {satellite}")

    sat_name = ILRS_TARGETS[satellite.lower()]["name"]
    temporal = f"{start_date.strftime('%Y-%m-%dT00:00:00Z')},{(end_date + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')}"
    page_num = 1
    rows: List[Dict] = []
    while True:
        params = {
            "collection_concept_id": collection_concept_id,
            "page_size": "2000",
            "page_num": str(page_num),
            "temporal": temporal,
            "options[producer_granule_id][pattern]": "true",
            "producer_granule_id": f"*{sat_name}*",
        }
        response = retry_with_backoff(
            lambda: guarded_request("GET", CMR_GRANULES_URL, params=params, timeout=120),
            attempts=3,
            base_delay_sec=2.0,
            max_delay_sec=20.0,
            label=f"CDDIS CMR granule search for {satellite}",
            verbose=verbose,
        )
        response.raise_for_status()
        entries = response.json().get("feed", {}).get("entry", [])
        if not entries:
            break
        rows.extend(entries)
        if len(entries) < 2000:
            break
        page_num += 1
    if verbose:
        print(f"Found {len(rows)} CDDIS CMR granules for {satellite}")
    return rows


def _pick_cddis_granule_link(granule: Dict) -> str | None:
    for link in granule.get("links", []):
        href = link.get("href")
        if not href:
            continue
        if href.endswith(".np2") or href.endswith(".npt") or href.endswith(".np2.gz") or href.endswith(".npt.gz"):
            return href
    return None


def download_cddis_cmr_granules(satellite: str,
                                start_date: datetime,
                                end_date: datetime,
                                output_dir: str,
                                max_files: int = None,
                                verbose: bool = True) -> List[str]:
    granules = list_cddis_cmr_granules(
        satellite=satellite,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
    )
    downloaded = []
    for granule in granules:
        href = _pick_cddis_granule_link(granule)
        if not href:
            continue
        if not _file_in_date_range(Path(href).name, start_date=start_date, end_date=end_date):
            continue
        filename = sanitize_remote_filename(href)
        final_name = filename[:-3] if filename.endswith('.gz') else filename
        final_path = os.path.join(output_dir, final_name)
        if os.path.exists(final_path):
            if verbose:
                print(f"  Skipping {final_name} (already exists)")
            continue
        local_path = download_cddis_file(href, output_dir, decompress=True, verbose=verbose)
        if local_path:
            downloaded.append(local_path)
            if max_files is not None and len(downloaded) >= max_files:
                break
    if verbose:
        print(f"Downloaded {len(downloaded)} CDDIS CMR granules for {satellite}")
    return downloaded


def build_cddis_candidate_urls(satellite: str,
                               start_date: datetime,
                               end_date: datetime,
                               data_type: str = 'npt',
                               granularity: str = "both") -> List[Dict]:
    if satellite.lower() not in ILRS_TARGETS:
        raise ValueError(f"Unknown satellite: {satellite}")
    if data_type not in {'npt', 'crd'}:
        raise ValueError("data_type must be 'npt' or 'crd'")

    target_name = ILRS_TARGETS[satellite.lower()]['name']
    rows = []
    for granularity_name, token in _candidate_tokens(start_date, end_date, granularity=granularity):
        year = token[:4]
        base = f"{target_name}_{token}.{data_type}"
        for suffix in ["", ".gz"]:
            rows.append(
                {
                    "satellite": satellite.lower(),
                    "granularity": granularity_name,
                    "token": token,
                    "compressed": bool(suffix),
                    "filename": base + suffix,
                    "url": f"{CDDIS_HTTPS_BASE}/{data_type}_crd/{year}/{target_name}/{base}{suffix}",
                }
            )
        if data_type == "npt" and granularity_name == "daily":
            np2_base = f"{target_name}_{token}.np2"
            rows.append(
                {
                    "satellite": satellite.lower(),
                    "granularity": granularity_name,
                    "token": token,
                    "compressed": False,
                    "filename": np2_base,
                    "url": f"{CDDIS_HTTPS_BASE}/npt_crd_v2/{target_name}/{year}/{np2_base}",
                }
            )
    return rows


def build_cddis_allsat_candidate_urls(start_date: datetime,
                                      end_date: datetime,
                                      data_type: str = 'npt') -> List[Dict]:
    if data_type != 'npt':
        raise ValueError("allsat probing is currently only implemented for npt")
    rows = []
    for token in _allsat_hourly_tokens(start_date, end_date):
        year = token[:4]
        filename = f"allsat_{token}.npt"
        rows.append(
            {
                "satellite": "allsat",
                "granularity": "hourly",
                "token": token,
                "compressed": False,
                "filename": filename,
                "url": f"{CDDIS_HTTPS_BASE}/npt_crd/{year}/allsat/{filename}",
            }
        )
    return rows


def probe_cddis_files(satellite: str,
                      start_date: datetime,
                      end_date: datetime,
                      data_type: str = 'npt',
                      granularity: str = "both",
                      scope: str = "satellite",
                      verbose: bool = True) -> List[Dict]:
    if scope == "allsat":
        candidates = build_cddis_allsat_candidate_urls(
            start_date,
            end_date,
            data_type=data_type,
        )
    else:
        candidates = build_cddis_candidate_urls(
            satellite,
            start_date,
            end_date,
            data_type=data_type,
            granularity=granularity,
        )
    available = []
    for row in candidates:
        try:
            code, final_url = _earthdata_head(row["url"])
        except subprocess.CalledProcessError:
            continue
        if code != "200":
            continue
        # A redirect to login or a generic site page is not a real archive hit.
        if "earthdata.nasa.gov" in final_url or "urs.earthdata.nasa.gov" in final_url:
            continue
        available.append({**row, "final_url": final_url, "transport": "https"})
        if verbose:
            print(f"  Found CDDIS file: {row['filename']}")
    if verbose:
        print(f"Found {len(available)} CDDIS candidate files for {satellite}")
    return available


def list_cddis_files(satellite: str,
                     data_type: str = 'npt',
                     start_date: datetime = None,
                     end_date: datetime = None,
                     transport: str = "auto",
                     granularity: str = "both",
                     scope: str = "satellite",
                     verbose: bool = True) -> List[Dict]:
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required for CDDIS probing")
    if transport not in {"auto", "https", "ftps"}:
        raise ValueError("transport must be one of: auto, https, ftps")
    if scope not in {"satellite", "allsat"}:
        raise ValueError("scope must be one of: satellite, allsat")
    if transport in {"auto", "https"}:
        rows = probe_cddis_files(
            satellite=satellite,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type,
            granularity=granularity,
            scope=scope,
            verbose=verbose,
        )
        if rows or transport == "https":
            return rows
        if verbose:
            print("  HTTPS probe returned no files, trying FTPS directory listing...")
    if scope == "allsat":
        if verbose:
            print("  FTPS fallback is not implemented for allsat probing.")
        return []
    return list_cddis_ftps_files(
        satellite=satellite,
        data_type=data_type,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
    )


def download_cddis_file(url: str,
                        output_dir: str,
                        decompress: bool = True,
                        verbose: bool = True) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    filename = sanitize_remote_filename(url)
    local_path = os.path.join(output_dir, filename)
    assert_path_within(local_path, output_dir)
    final_name = filename[:-3] if filename.endswith('.gz') else filename
    final_path = os.path.join(output_dir, final_name)
    if os.path.exists(final_path):
        if verbose:
            print(f"  Skipping {final_name} (already exists)")
        return final_path

    temp_target = local_path
    errors = []
    for candidate_url in _cddis_download_url_variants(url):
        if verbose:
            print(f"  Downloading {filename} from CDDIS...")
        try:
            _curl_earthdata(candidate_url, write_to=temp_target)
            if decompress and temp_target.endswith('.gz'):
                with gzip.open(temp_target, 'rb') as f_in:
                    with atomic_output_file(final_path, 'wb') as (f_out, _):
                        shutil.copyfileobj(f_in, f_out)
                os.remove(temp_target)
                return final_path
            return temp_target
        except Exception as exc:
            errors.append(str(exc))
            cleanup_temp_file(temp_target)
            try:
                os.remove(temp_target)
            except OSError:
                pass
    if verbose:
        print(f"  CDDIS download failed for {filename}: {'; '.join(errors)}")
    return None


def get_station_passes(satellite: str,
                       date: datetime,
                       verbose: bool = True) -> List[Dict]:
    """
    Get SLR station passes for a satellite on a specific date

    This can help identify which stations tracked the satellite.

    Args:
        satellite: Satellite name
        date: Date to check
        verbose: Print progress

    Returns:
        List of pass information
    """
    # This would query ILRS for tracking predictions
    # For now, return empty list
    return []


def download_batch(satellites: List[str],
                   start_date: datetime,
                   end_date: datetime,
                   base_output_dir: str,
                   data_type: str = 'npt',
                   verbose: bool = True) -> Dict[str, List[str]]:
    """
    Download SLR data for multiple satellites

    Args:
        satellites: List of satellite names
        start_date: Start date
        end_date: End date
        base_output_dir: Base output directory
        data_type: 'npt' or 'crd'
        verbose: Print progress

    Returns:
        Dictionary mapping satellite to downloaded files
    """
    results = {}

    for satellite in satellites:
        sat_dir = os.path.join(base_output_dir, satellite.lower())
        downloaded = download_ilrs_data(
            satellite, start_date, end_date, sat_dir, data_type, verbose=verbose
        )
        results[satellite] = downloaded

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download ILRS SLR data')
    parser.add_argument('satellite', help='Satellite name (e.g., sentinel-3a, jason-3)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument(
        '--output',
        default='./data/raw/reference_validation/sentinel-3a/slr',
        help='Output directory',
    )
    parser.add_argument('--type', default='npt', choices=['npt', 'crd'],
                        help='Data type (npt=Normal Points, crd=full rate)')
    parser.add_argument('--max-files', type=int, help='Max files (for testing)')
    parser.add_argument('--list-only', action='store_true', help='List without downloading')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    if args.list_only:
        files = list_edc_files(args.satellite, args.type, start_date, end_date)
        for f in files:
            print(f)
    else:
        download_ilrs_data(
            args.satellite,
            start_date,
            end_date,
            args.output,
            args.type,
            args.max_files
        )
