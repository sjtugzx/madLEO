"""
Sentinel POD Orbit Downloader

Downloads precision orbit data for Sentinel satellites from:
- ASF (Alaska Satellite Facility) - Primary for S1A/S1B
- Copernicus Data Space - Alternative source

Orbit types:
- RESORB: Restituted (near real-time, ~3cm accuracy)
- POEORB: Precise (10-21 days latency, ~5mm accuracy)
"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from lxml import etree
from urllib.parse import urljoin, quote
import re
from downloaders.download_utils import atomic_output_file, cleanup_temp_file
from downloaders.net_guard import assert_path_within, guarded_request, sanitize_remote_filename


def _hardened_xml_parser() -> etree.XMLParser:
    """Return a hardened parser for provider XML (no entities, no network)."""
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        dtd_validation=False,
        load_dtd=False,
    )


# ASF (Alaska Satellite Facility) - requires authentication now
ASF_POEORB_URL = 'https://s1qc.asf.alaska.edu/aux_poeorb/'
ASF_RESORB_URL = 'https://s1qc.asf.alaska.edu/aux_resorb/'

# AWS S3 (Free, no authentication) - Primary source
AWS_S3_BUCKET = 's1-orbits'
AWS_S3_REGION = 'us-west-2'
AWS_S3_BASE_URL = f'https://{AWS_S3_BUCKET}.s3.{AWS_S3_REGION}.amazonaws.com/'
AWS_POEORB_PREFIX = 'AUX_POEORB/'
AWS_RESORB_PREFIX = 'AUX_RESORB/'

# Request configuration
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; LEO-Orbit-Dataset/1.0)'
}

# Retry configuration
RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)


def get_session() -> requests.Session:
    """Create a requests session with retry logic"""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Sentinel satellite info
SENTINEL_SATELLITES = {
    'S1A': {
        'name': 'Sentinel-1A',
        'launch': datetime(2014, 4, 3),
        'norad': '39634',
        'mission': 'S1A'
    },
    'S1B': {
        'name': 'Sentinel-1B',
        'launch': datetime(2016, 4, 25),
        'end': datetime(2022, 12, 23),  # Retired
        'norad': '41456',
        'mission': 'S1B'
    },
    'S2A': {
        'name': 'Sentinel-2A',
        'launch': datetime(2015, 6, 23),
        'norad': '40697',
        'mission': 'S2A'
    },
    'S2B': {
        'name': 'Sentinel-2B',
        'launch': datetime(2017, 3, 7),
        'norad': '42063',
        'mission': 'S2B'
    },
    'S3A': {
        'name': 'Sentinel-3A',
        'launch': datetime(2016, 2, 16),
        'norad': '41335',
        'mission': 'S3A'
    },
    'S3B': {
        'name': 'Sentinel-3B',
        'launch': datetime(2018, 4, 25),
        'norad': '43437',
        'mission': 'S3B'
    },
}


def parse_orbit_filename(filename: str) -> Dict:
    """
    Parse Sentinel orbit filename to extract metadata

    Filename format:
    S1A_OPER_AUX_POEORB_OPOD_20240101T120000_V20231230T225942_20240101T005942.EOF

    Args:
        filename: EOF filename

    Returns:
        Dictionary with parsed metadata
    """
    metadata = {
        'satellite': None,
        'orbit_type': None,
        'creation_time': None,
        'validity_start': None,
        'validity_stop': None
    }

    # Pattern for Sentinel orbit files
    pattern = r'(S[123][AB])_OPER_AUX_(POEORB|RESORB)_OPOD_(\d{8}T\d{6})_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF'
    match = re.match(pattern, filename)

    if match:
        metadata['satellite'] = match.group(1)
        metadata['orbit_type'] = match.group(2)
        metadata['creation_time'] = datetime.strptime(match.group(3), '%Y%m%dT%H%M%S')
        metadata['validity_start'] = datetime.strptime(match.group(4), '%Y%m%dT%H%M%S')
        metadata['validity_stop'] = datetime.strptime(match.group(5), '%Y%m%dT%H%M%S')

    return metadata


def list_aws_orbits(satellite: str,
                    orbit_type: str = 'POEORB',
                    start_date: datetime = None,
                    end_date: datetime = None,
                    verbose: bool = True) -> List[Dict]:
    """
    List available orbit files from AWS S3 (free, no auth required)

    Args:
        satellite: Satellite ID (S1A, S1B, etc.)
        orbit_type: POEORB or RESORB
        start_date: Optional start date filter
        end_date: Optional end date filter
        verbose: Print progress

    Returns:
        List of orbit file metadata dicts
    """
    satellite = satellite.upper()

    # Build S3 list URL
    prefix = AWS_POEORB_PREFIX if orbit_type == 'POEORB' else AWS_RESORB_PREFIX

    available = []
    continuation_token = None

    try:
        if verbose:
            print(f"Querying AWS S3 for {satellite} {orbit_type} orbits...")

        session = get_session()

        while True:
            # Build URL with continuation token
            url = f"{AWS_S3_BASE_URL}?list-type=2&prefix={prefix}&max-keys=1000"
            if continuation_token:
                url += f"&continuation-token={quote(continuation_token, safe='')}"

            response = guarded_request("GET", url, headers=HEADERS, timeout=60, session=session)
            response.raise_for_status()

            # Parse XML response
            root = etree.fromstring(response.content, parser=_hardened_xml_parser())
            ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}

            for content in root.findall('.//s3:Contents', ns):
                key = content.find('s3:Key', ns).text
                filename = key.split('/')[-1]

                # Filter by satellite
                if not filename.startswith(satellite):
                    continue

                metadata = parse_orbit_filename(filename)
                if metadata['satellite']:
                    metadata['url'] = AWS_S3_BASE_URL + key
                    metadata['filename'] = filename

                    # Get file size
                    size_elem = content.find('s3:Size', ns)
                    if size_elem is not None:
                        metadata['size'] = int(size_elem.text)

                    # Apply date filter
                    if start_date and metadata['validity_stop'] < start_date:
                        continue
                    if end_date and metadata['validity_start'] > end_date:
                        continue

                    available.append(metadata)

            # Check for more results
            is_truncated = root.find('.//s3:IsTruncated', ns)
            if is_truncated is not None and is_truncated.text == 'true':
                next_token = root.find('.//s3:NextContinuationToken', ns)
                if next_token is not None:
                    continuation_token = next_token.text
                else:
                    break
            else:
                break

        if verbose:
            print(f"  Found {len(available)} orbit files")

    except Exception as e:
        print(f"Error querying AWS S3: {e}")

    return sorted(available, key=lambda x: x.get('validity_start', datetime.min))


def list_asf_orbits(satellite: str,
                    orbit_type: str = 'POEORB',
                    start_date: datetime = None,
                    end_date: datetime = None,
                    verbose: bool = True) -> List[Dict]:
    """
    List available orbit files from ASF

    Args:
        satellite: Satellite ID (S1A, S1B, etc.)
        orbit_type: POEORB or RESORB
        start_date: Optional start date filter
        end_date: Optional end date filter
        verbose: Print progress

    Returns:
        List of orbit file metadata dicts
    """
    satellite = satellite.upper()
    if satellite not in SENTINEL_SATELLITES:
        raise ValueError(f"Unknown satellite: {satellite}")

    # Build URL based on orbit type
    if orbit_type == 'POEORB':
        base_url = ASF_POEORB_URL
    else:
        base_url = ASF_RESORB_URL

    available = []

    try:
        if verbose:
            print(f"Querying ASF for {satellite} {orbit_type} orbits...")

        session = get_session()
        response = guarded_request("GET", base_url, headers=HEADERS, timeout=60, session=session)
        response.raise_for_status()

        # Parse HTML to find links
        # Simple regex to find EOF files
        pattern = rf'href="({satellite}_OPER_AUX_{orbit_type}_[^"]+\.EOF)"'
        matches = re.findall(pattern, response.text)

        for filename in matches:
            metadata = parse_orbit_filename(filename)
            if metadata['satellite']:
                metadata['url'] = urljoin(base_url, filename)
                metadata['filename'] = filename

                # Apply date filter
                if start_date and metadata['validity_stop'] < start_date:
                    continue
                if end_date and metadata['validity_start'] > end_date:
                    continue

                available.append(metadata)

        if verbose:
            print(f"  Found {len(available)} orbit files")

    except requests.exceptions.SSLError as e:
        print(f"SSL Error (may be network/firewall issue): {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
    except Exception as e:
        print(f"Error querying ASF: {e}")

    return sorted(available, key=lambda x: x.get('validity_start', datetime.min))


def download_sentinel_orbit(url: str,
                            output_dir: str,
                            verbose: bool = True) -> Optional[str]:
    """
    Download a single Sentinel orbit file

    Args:
        url: URL to download
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Local file path or None if failed
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = sanitize_remote_filename(url)
    local_path = os.path.join(output_dir, filename)
    assert_path_within(local_path, output_dir)

    # Skip if already exists
    if os.path.exists(local_path):
        if verbose:
            print(f"  Skipping {filename} (already exists)")
        return local_path

    try:
        if verbose:
            print(f"  Downloading {filename}...")

        session = get_session()
        response = guarded_request("GET", url, headers=HEADERS, timeout=120, stream=True, session=session)
        response.raise_for_status()

        with atomic_output_file(local_path, 'wb') as (f, _):
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return local_path

    except requests.exceptions.SSLError as e:
        print(f"  SSL Error: {e}")
        cleanup_temp_file(local_path)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  Connection Error: {e}")
        cleanup_temp_file(local_path)
        return None
    except Exception as e:
        print(f"  Download failed: {e}")
        cleanup_temp_file(local_path)
        return None


def download_sentinel_orbits(satellite: str,
                             start_date: datetime,
                             end_date: datetime,
                             output_dir: str,
                             orbit_type: str = 'POEORB',
                             max_files: int = None,
                             source: str = 'aws',
                             verbose: bool = True) -> List[str]:
    """
    Download Sentinel orbit files for a date range

    Args:
        satellite: Satellite ID
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        orbit_type: POEORB or RESORB
        max_files: Maximum files to download (for testing)
        source: Data source - 'aws' (free, recommended) or 'asf' (requires auth)
        verbose: Print progress

    Returns:
        List of downloaded file paths
    """
    if verbose:
        print(f"Downloading {satellite} {orbit_type} orbits")
        print(f"  Period: {start_date.date()} to {end_date.date()}")
        print(f"  Source: {source.upper()}")

    # List available files based on source
    if source == 'aws':
        available = list_aws_orbits(satellite, orbit_type, start_date, end_date, verbose=verbose)
    else:
        available = list_asf_orbits(satellite, orbit_type, start_date, end_date, verbose=verbose)

    if max_files:
        available = available[:max_files]

    downloaded = []
    for orbit_info in available:
        local_path = download_sentinel_orbit(orbit_info['url'], output_dir, verbose=verbose)
        if local_path:
            downloaded.append(local_path)

    if verbose:
        print(f"Downloaded {len(downloaded)} files")

    return downloaded


def download_sentinel_batch(satellites: List[str],
                            start_date: datetime,
                            end_date: datetime,
                            base_output_dir: str,
                            orbit_type: str = 'POEORB',
                            verbose: bool = True) -> Dict[str, List[str]]:
    """
    Download orbits for multiple Sentinel satellites

    Args:
        satellites: List of satellite IDs
        start_date: Start date
        end_date: End date
        base_output_dir: Base output directory
        orbit_type: POEORB or RESORB
        verbose: Print progress

    Returns:
        Dictionary mapping satellite to downloaded files
    """
    results = {}

    for satellite in satellites:
        sat_dir = os.path.join(base_output_dir, satellite.lower())
        downloaded = download_sentinel_orbits(
            satellite, start_date, end_date, sat_dir, orbit_type, verbose=verbose
        )
        results[satellite] = downloaded

    return results


def find_orbit_for_date(satellite: str,
                        target_date: datetime,
                        orbit_type: str = 'POEORB') -> Optional[Dict]:
    """
    Find the orbit file that covers a specific date

    Args:
        satellite: Satellite ID
        target_date: Date to find coverage for
        orbit_type: POEORB or RESORB

    Returns:
        Orbit file metadata or None
    """
    # Search with a small window around target date
    start = target_date - timedelta(days=1)
    end = target_date + timedelta(days=1)

    available = list_asf_orbits(satellite, orbit_type, start, end, verbose=False)

    for orbit in available:
        if orbit['validity_start'] <= target_date <= orbit['validity_stop']:
            return orbit

    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download Sentinel orbit files')
    parser.add_argument('satellite', help='Satellite ID (S1A, S1B, S2A, S2B, S3A, S3B)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument(
        '--output',
        default='./data/raw/reference_validation/sentinel-3a/pod',
        help='Output directory',
    )
    parser.add_argument('--type', default='POEORB', choices=['POEORB', 'RESORB'],
                        help='Orbit type')
    parser.add_argument('--max-files', type=int, help='Max files (for testing)')
    parser.add_argument('--list-only', action='store_true', help='List without downloading')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    if args.list_only:
        orbits = list_asf_orbits(args.satellite, args.type, start_date, end_date)
        for orbit in orbits:
            print(f"{orbit['filename']}")
            print(f"  Valid: {orbit['validity_start']} to {orbit['validity_stop']}")
    else:
        download_sentinel_orbits(
            args.satellite,
            start_date,
            end_date,
            args.output,
            args.type,
            args.max_files
        )
