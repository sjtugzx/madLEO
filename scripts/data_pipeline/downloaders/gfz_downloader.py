"""
GFZ/ISDC Precision Orbit Downloader

Downloads precision orbit products (SP3 files) from:
- GFZ ISDC (HTTPS) - Primary source
- GFZ FTP server - Fallback

Product types:
- GNSS orbits (GPS, GLONASS, Galileo): Available via ISDC
- LEO satellite orbits (Sentinel, Jason, Swarm): See dedicated downloaders

Note: For LEO satellites like Sentinel-3A, use sentinel_downloader.py
      For GNSS constellation orbits, use this downloader.
"""

import os
import re
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import gzip
import shutil
from downloaders.download_utils import atomic_output_file, cleanup_temp_file
from downloaders.net_guard import assert_path_within, guarded_request, sanitize_remote_filename


# ISDC HTTPS endpoints (preferred, no auth required for older data)
ISDC_BASE_URL = 'https://isdc-data.gfz.de/gnss/products/'
ISDC_RAPID_URL = ISDC_BASE_URL + 'rapid/'
ISDC_FINAL_URL = ISDC_BASE_URL + 'final/'
ISDC_ULTRA_URL = ISDC_BASE_URL + 'ultra/'

# Product configurations
PRODUCT_TYPES = {
    'rapid': {
        'url': ISDC_RAPID_URL,
        'latency': '1 day',
        'format': 'SP3-d',
        'systems': ['GPS', 'GLONASS', 'Galileo']
    },
    'final': {
        'url': ISDC_FINAL_URL,
        'latency': '12-18 days',
        'format': 'SP3-d',
        'systems': ['GPS', 'GLONASS']
    },
    'ultra': {
        'url': ISDC_ULTRA_URL,
        'latency': '3 hours',
        'format': 'SP3-c',
        'systems': ['GPS', 'GLONASS', 'Galileo', 'BeiDou', 'QZSS']
    }
}

# Request headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; LEO-Orbit-Dataset/1.0)'
}


def gps_week_from_date(date: datetime) -> Tuple[int, int]:
    """
    Calculate GPS week and day of week from date

    Args:
        date: datetime object

    Returns:
        Tuple of (GPS week number, day of week 0-6)
    """
    gps_epoch = datetime(1980, 1, 6)
    delta = date - gps_epoch
    gps_week = delta.days // 7
    day_of_week = delta.days % 7
    return gps_week, day_of_week


def date_from_gps_week(gps_week: int, day_of_week: int = 0) -> datetime:
    """
    Calculate date from GPS week and day

    Args:
        gps_week: GPS week number
        day_of_week: Day of week (0-6)

    Returns:
        datetime object
    """
    gps_epoch = datetime(1980, 1, 6)
    return gps_epoch + timedelta(weeks=gps_week, days=day_of_week)


def list_isdc_weeks(product_type: str = 'rapid',
                    verbose: bool = True) -> List[int]:
    """
    List available GPS weeks on ISDC server

    Args:
        product_type: 'rapid', 'final', or 'ultra'
        verbose: Print progress

    Returns:
        List of available GPS week numbers
    """
    if product_type not in PRODUCT_TYPES:
        raise ValueError(f"Unknown product type: {product_type}")

    base_url = PRODUCT_TYPES[product_type]['url']
    weeks = []

    try:
        if verbose:
            print(f"Querying ISDC {product_type} products...")

        response = guarded_request("GET", base_url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        # Parse directory listing
        # Looking for links like "w2295/" or "2295/"
        pattern = r'href="w?(\d{4})/"'
        matches = re.findall(pattern, response.text)

        weeks = sorted(set(int(w) for w in matches))

        if verbose:
            print(f"  Found {len(weeks)} weeks available")
            if weeks:
                print(f"  Range: w{weeks[0]} to w{weeks[-1]}")

    except Exception as e:
        print(f"Error querying ISDC: {e}")

    return weeks


def list_isdc_files(gps_week: int,
                    product_type: str = 'rapid',
                    verbose: bool = True) -> List[Dict]:
    """
    List SP3 files available for a specific GPS week

    Args:
        gps_week: GPS week number
        product_type: 'rapid', 'final', or 'ultra'
        verbose: Print progress

    Returns:
        List of file info dictionaries
    """
    if product_type not in PRODUCT_TYPES:
        raise ValueError(f"Unknown product type: {product_type}")

    base_url = PRODUCT_TYPES[product_type]['url']
    week_url = f"{base_url}w{gps_week}/"

    files = []

    try:
        response = guarded_request("GET", week_url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        # Parse for SP3 files
        # Pattern: GFZ0MGXRAP_20240107000_01D_05M_ORB.SP3.gz
        pattern = r'href="([^"]+\.(?:sp3|SP3)(?:\.gz)?)"'
        matches = re.findall(pattern, response.text)

        for filename in matches:
            file_info = {
                'filename': filename,
                'url': week_url + filename,
                'gps_week': gps_week,
                'compressed': filename.endswith('.gz')
            }
            files.append(file_info)

        if verbose and files:
            print(f"  Week {gps_week}: {len(files)} SP3 files")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            if verbose:
                print(f"  Week {gps_week}: not available")
        else:
            print(f"  Week {gps_week}: HTTP error {e}")
    except Exception as e:
        print(f"  Week {gps_week}: error {e}")

    return files


def list_gfz_files(start_date: datetime,
                   end_date: datetime,
                   product_type: str = 'rapid',
                   verbose: bool = True) -> List[Dict]:
    """
    List available SP3 files for a date range

    Args:
        start_date: Start date
        end_date: End date
        product_type: 'rapid', 'final', or 'ultra'
        verbose: Print progress

    Returns:
        List of file info dictionaries
    """
    all_files = []

    # Get GPS weeks for date range
    start_week, _ = gps_week_from_date(start_date)
    end_week, _ = gps_week_from_date(end_date)

    if verbose:
        print(f"Searching ISDC {product_type} products")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        print(f"  GPS weeks: {start_week} to {end_week}")

    for week in range(start_week, end_week + 1):
        files = list_isdc_files(week, product_type, verbose=verbose)
        all_files.extend(files)

    if verbose:
        print(f"Found {len(all_files)} total files")

    return all_files


def download_isdc_file(file_info: Dict,
                       output_dir: str,
                       decompress: bool = True,
                       verbose: bool = True) -> Optional[str]:
    """
    Download a single file from ISDC

    Args:
        file_info: File info dictionary with 'url' and 'filename'
        output_dir: Local output directory
        decompress: Decompress .gz files
        verbose: Print progress

    Returns:
        Local file path or None if failed
    """
    os.makedirs(output_dir, exist_ok=True)

    url = file_info['url']
    filename = sanitize_remote_filename(file_info['filename'])
    local_path = os.path.join(output_dir, filename)
    assert_path_within(local_path, output_dir)

    # Check if already exists (possibly decompressed)
    if decompress and filename.endswith('.gz'):
        decompressed_path = local_path[:-3]
        if os.path.exists(decompressed_path):
            if verbose:
                print(f"  Skipping {filename} (already exists)")
            return decompressed_path
    elif os.path.exists(local_path):
        if verbose:
            print(f"  Skipping {filename} (already exists)")
        return local_path

    try:
        if verbose:
            print(f"  Downloading {filename}...")

        response = guarded_request("GET", url, headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        with atomic_output_file(local_path, 'wb') as (f, _):
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

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
        if os.path.exists(local_path):
            os.remove(local_path)
        cleanup_temp_file(local_path)
        return None


def download_gfz_orbit(start_date: datetime,
                       end_date: datetime,
                       output_dir: str,
                       product_type: str = 'rapid',
                       max_files: int = None,
                       verbose: bool = True) -> List[str]:
    """
    Download GNSS orbit data from GFZ ISDC

    Args:
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        product_type: 'rapid', 'final', or 'ultra'
        max_files: Maximum number of files to download (for testing)
        verbose: Print progress

    Returns:
        List of downloaded file paths
    """
    if verbose:
        print(f"Downloading GFZ {product_type} GNSS orbits")
        print(f"  Period: {start_date.date()} to {end_date.date()}")

    # List available files
    files = list_gfz_files(start_date, end_date, product_type, verbose=verbose)

    if max_files:
        files = files[:max_files]

    downloaded = []
    for file_info in files:
        local_path = download_isdc_file(file_info, output_dir, verbose=verbose)
        if local_path:
            downloaded.append(local_path)

    if verbose:
        print(f"Downloaded {len(downloaded)} files")

    return downloaded


def download_orbit_batch(start_date: datetime,
                         end_date: datetime,
                         base_output_dir: str,
                         product_types: List[str] = None,
                         verbose: bool = True) -> Dict[str, List[str]]:
    """
    Download orbits for multiple product types

    Args:
        start_date: Start date
        end_date: End date
        base_output_dir: Base output directory
        product_types: List of product types (default: ['rapid'])
        verbose: Print progress

    Returns:
        Dictionary mapping product type to list of downloaded files
    """
    if product_types is None:
        product_types = ['rapid']

    results = {}

    for ptype in product_types:
        output_dir = os.path.join(base_output_dir, ptype)
        downloaded = download_gfz_orbit(
            start_date, end_date, output_dir, ptype, verbose=verbose
        )
        results[ptype] = downloaded

    return results


# Legacy compatibility - satellite info for reference
SATELLITE_INFO = {
    'sentinel-3a': {
        'cospar': '2016-011A',
        'norad': '41335',
        'sp3_id': 'S3A',
        'note': 'Use Copernicus POD Service for Sentinel-3A orbits'
    },
    'sentinel-3b': {
        'cospar': '2018-039A',
        'norad': '43437',
        'sp3_id': 'S3B',
        'note': 'Use Copernicus POD Service for Sentinel-3B orbits'
    },
    'jason-3': {
        'cospar': '2016-002A',
        'norad': '41240',
        'sp3_id': 'JA3',
        'note': 'Use CNES AVISO for Jason-3 orbits'
    },
    'swarm-a': {
        'cospar': '2013-067A',
        'norad': '39452',
        'sp3_id': 'SWA',
        'note': 'Use ESA Swarm Data Access for Swarm orbits'
    },
}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download GFZ/ISDC GNSS orbits')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument(
        '--output',
        default='./data/raw/expansion_candidates/gnss/pod',
        help='Output directory',
    )
    parser.add_argument('--type', default='rapid', choices=['rapid', 'final', 'ultra'],
                        help='Product type')
    parser.add_argument('--max-files', type=int, help='Max files to download (for testing)')
    parser.add_argument('--list-only', action='store_true', help='List files without downloading')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    if args.list_only:
        files = list_gfz_files(start_date, end_date, args.type)
        for f in files:
            print(f"{f['filename']}")
    else:
        download_gfz_orbit(
            start_date,
            end_date,
            args.output,
            args.type,
            args.max_files
        )
