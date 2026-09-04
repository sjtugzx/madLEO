"""
Satellite Laser Ranging (SLR) Data Processing Module

This module provides parsers for SLR data formats:
1. CRD (Consolidated laser Ranging Data) format - ILRS standard
2. Historic ILRS normal-point format ('99999' indicator records)
3. MERIT-II / ILRS full-rate V2 merged files (see slr_formats.py)

Files are routed by CONTENT via ``slr_formats.detect_slr_format``; there is
deliberately no fallback guess parser (root-cause fix A4: the retired
``parse_normal_point_file`` force-fitted arbitrary token lines and
fabricated records).

SLR provides independent validation of satellite orbits through
ground-based laser ranging measurements.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from astropy.time import Time

from processors.slr_formats import (
    UnsupportedSLRFormatError,
    detect_slr_format,
    mission_window_for_target,
    parse_merit_monthly_file,
    read_slr_text,
)


# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s

# Corrupt-parse range floor, m: one-way ranges below this cannot be real
# slant ranges of any dataset target (lowest target altitude ~540 km) --
# they are corrupt parses (e.g. the 3 topex-poseidon legacy records at
# 0.058-223.6 m with tof ~4e-10 s from the original review).  Same value
# and derivation as physical_qc.SLR_RANGE_MIN_M; kept local here because
# physical_qc imports this module (no import cycle).
_CORRUPT_RANGE_FLOOR_M = 5.0e5


class CRDHeader:
    """Container for CRD file header information"""

    def __init__(self):
        self.format_version: str = ''
        self.production_date: datetime = None
        self.station_name: str = ''
        self.station_id: str = ''  # CDP (COSPAR) ID
        self.system_id: str = ''
        self.target_name: str = ''
        self.target_id: str = ''  # COSPAR ID
        self.target_class: int = 0
        self.target_norad: str = ''
        self.sic: str = ''
        self.start_time: datetime = None
        self.end_time: datetime = None
        self.wavelength: float = 0.0  # nm
        self.time_scale: str = 'UTC'
        self.epoch_format: str = ''
        # Parse ledger (P1: counted rejections, never silent skips).
        self.records_rejected: int = 0
        self.reject_reasons: Dict[str, int] = {}
        # True iff undecodable bytes were replaced while reading the file
        # (see slr_formats.read_slr_text).
        self.non_utf8_sanitized_copy: bool = False


class NormalPoint:
    """Container for a Normal Point record"""

    def __init__(self):
        self.epoch: datetime = None
        self.time_of_flight_s: float = 0.0  # seconds (two-way)
        self.range: float = 0.0  # meters (one-way)
        self.sigma: float = 0.0  # meters, RMS
        self.num_returns: int = 0
        self.window_length: float = 0.0  # seconds
        self.bin_rms: float = 0.0  # meters
        self.skew: float = 0.0
        self.kurtosis: float = 0.0
        self.peak_minus_mean: float = 0.0
        self.calibration_delay: float = 0.0  # seconds
        self.system_delay: float = 0.0  # seconds
        self.atmospheric_delay: float = 0.0  # meters


def parse_crd_file(filepath: str) -> Tuple[CRDHeader, List[Dict]]:
    """
    Parse CRD (Consolidated Ranging Data) format file

    CRD format specification: https://ilrs.gsfc.nasa.gov/data_and_products/formats/crd.html

    Multi-session files (merged monthlies): per the CRD spec H8/H9 are
    SESSION-boundary records, not end-of-file markers -- parsing continues
    past them, and each H8 resets the session context (the following H4
    re-anchors the seconds-of-day epoch base).  Data records seen without
    a session date are counted rejections ('epoch_reference_missing'),
    never silently mis-epoched.  Every structurally invalid or unknown
    line is likewise a counted rejection in ``header.reject_reasons``
    (P1: no silent skips); files are read through
    ``slr_formats.read_slr_text`` so binary junk in ``60`` free-text
    fields sanitizes instead of crashing (header.non_utf8_sanitized_copy).

    Args:
        filepath: Path to CRD file

    Returns:
        Tuple of (CRDHeader, list of range records)
    """
    header = CRDHeader()
    range_records = []

    text, non_utf8 = read_slr_text(filepath)
    header.non_utf8_sanitized_copy = non_utf8

    def _reject(reason: str) -> None:
        header.records_rejected += 1
        header.reject_reasons[reason] = header.reject_reasons.get(reason, 0) + 1

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Record type is the first field
        parts = line.split()
        if not parts:
            continue

        record_type = parts[0].upper()

        # Header records
        if record_type == 'H1':
            # Format header
            if len(parts) >= 3:
                header.format_version = parts[1]

        elif record_type == 'H2':
            # Station header
            if len(parts) >= 3:
                header.station_name = parts[1]
                header.station_id = parts[2]
                header.system_id = parts[3] if len(parts) > 3 else ''

        elif record_type == 'H3':
            # Target header
            if len(parts) >= 3:
                header.target_name = parts[1]
                header.target_id = parts[2]
                header.sic = parts[3] if len(parts) > 3 else ''
                header.target_norad = parts[4] if len(parts) > 4 else ''
                try:
                    header.target_class = int(parts[5]) if len(parts) > 5 else 0
                except ValueError:
                    header.target_class = 0

        elif record_type == 'H4':
            # Session header
            if len(parts) >= 8:
                try:
                    # Downloaded CRD files use:
                    # H4 <data_type> <start_year> <start_month> ...
                    offset = 1 if len(parts[1]) == 4 else 2
                    year = int(parts[offset])
                    month = int(parts[offset + 1])
                    day = int(parts[offset + 2])
                    hour = int(parts[offset + 3]) if len(parts) > offset + 3 else 0
                    minute = int(parts[offset + 4]) if len(parts) > offset + 4 else 0
                    second = int(parts[offset + 5]) if len(parts) > offset + 5 else 0
                    header.start_time = datetime(year, month, day, hour, minute, second)
                except (ValueError, IndexError):
                    pass

        elif record_type == 'H5':
            # End session header
            if len(parts) >= 8:
                try:
                    offset = 1 if len(parts[1]) == 4 else 2
                    year = int(parts[offset])
                    month = int(parts[offset + 1])
                    day = int(parts[offset + 2])
                    hour = int(parts[offset + 3]) if len(parts) > offset + 3 else 23
                    minute = int(parts[offset + 4]) if len(parts) > offset + 4 else 59
                    second = int(parts[offset + 5]) if len(parts) > offset + 5 else 59
                    header.end_time = datetime(year, month, day, hour, minute, second)
                except (ValueError, IndexError):
                    pass

        elif record_type == 'C0':
            # System configuration
            if len(parts) >= 3:
                try:
                    header.wavelength = float(parts[2])  # nm
                except (ValueError, IndexError):
                    pass

        elif record_type == 'C1':
            # Laser configuration
            pass

        elif record_type == 'C2':
            # Detector configuration
            pass

        elif record_type == 'C3':
            # Timing configuration
            if len(parts) >= 2:
                header.time_scale = parts[1]

        elif record_type == 'C4':
            # Transponder configuration (for 2-way)
            pass

        elif record_type == '10':
            # Range record (full rate)
            if len(parts) < 5:
                _reject('short_record')
                continue
            try:
                seconds_of_day = float(parts[1])
                time_of_flight = float(parts[2])  # seconds, two-way
            except ValueError:
                _reject('malformed_record')
                continue
            if header.start_time is None:
                # No session date anchor (data before any H4, or after an
                # H8 session boundary without a new H4): reject loudly
                # instead of silently mis-epoching against another session.
                _reject('epoch_reference_missing')
                continue
            base_date = header.start_time.replace(hour=0, minute=0, second=0)
            epoch = base_date + timedelta(seconds=seconds_of_day)

            # One-way range in meters
            one_way_range = (time_of_flight / 2) * SPEED_OF_LIGHT

            range_records.append({
                'record_type': 'full_rate',
                'epoch': epoch,
                'time_of_flight_s': time_of_flight,
                'range_m': one_way_range,
                'station_id': header.station_id,
                'target_id': header.target_id,
                'target_name': header.target_name
            })

        elif record_type == '11':
            # Normal point record
            if len(parts) < 8:
                _reject('short_record')
                continue
            try:
                seconds_of_day = float(parts[1])
                time_of_flight = float(parts[2])  # seconds, two-way
                # CRD normal point layout in the downloaded files:
                # 11 sod tof cfg_id data_release window_length num_returns bin_rms ...
                window_length = float(parts[5])
                num_returns = int(float(parts[6]))
                bin_rms = float(parts[7])  # picoseconds
            except ValueError:
                _reject('malformed_record')
                continue
            if header.start_time is None:
                _reject('epoch_reference_missing')
                continue

            base_date = header.start_time.replace(hour=0, minute=0, second=0)
            epoch = base_date + timedelta(seconds=seconds_of_day)

            # One-way range in meters
            one_way_range = (time_of_flight / 2) * SPEED_OF_LIGHT

            # Convert bin RMS from picoseconds to meters
            sigma_m = (bin_rms * 1e-12 / 2) * SPEED_OF_LIGHT

            range_records.append({
                'record_type': 'normal_point',
                'epoch': epoch,
                'time_of_flight_s': time_of_flight,
                'range_m': one_way_range,
                'sigma_m': sigma_m,
                'num_returns': num_returns,
                'window_length': window_length,
                'station_id': header.station_id,
                'target_id': header.target_id,
                'target_name': header.target_name
            })

        elif record_type in ('12', '20', '30', '40', '50', '60'):
            # Range supplement / meteorological / pointing / calibration /
            # session statistics / compatibility records: parsed-but-unused
            # known record types -- not rejections.
            pass

        elif record_type == '00' or (len(record_type) == 2 and record_type[0] == '9'):
            # Spec-defined but free-form, bypassed per the CRD v2 manual:
            # '00' comment records ("Comments as needed"; the archived
            # ilrs-crd-v2.01e3.pdf: the jason-2 monthlies use them for
            # detector-change notes like "New CFD in the STOP channel")
            # and user-defined 9x records ("Just bypass them as you do
            # not know the format").  Not rejections.
            pass

        elif record_type in ('H8', 'H9'):
            # End of SESSION (H8) / end of last session data (H9): per the
            # CRD spec these close the current session; merged multi-session
            # monthlies continue with a new H1..H4 block.  Parsing continues
            # and the session context resets so a session without its own
            # H4 cannot silently reuse the previous session's date.
            header.start_time = None

        elif len(record_type) == 2 and record_type[0] in ('H', 'C') and record_type[1].isdigit():
            # Known header/config family (H5, H6, H7, C1..C9 variants not
            # individually consumed): recognized, ignored.
            pass

        else:
            _reject('unknown_record_type')

    return header, range_records


def calculate_residual(observed_range: float,
                       computed_range: float) -> float:
    """
    Calculate O-C (Observed minus Computed) range residual

    Args:
        observed_range: Measured range (meters)
        computed_range: Predicted range from orbit (meters)

    Returns:
        Residual in meters (O-C)
    """
    return observed_range - computed_range


def compute_range_from_orbit(station_ecef: np.ndarray,
                             sat_ecef: np.ndarray) -> float:
    """
    Compute geometric range from station to satellite

    Args:
        station_ecef: Station position in ECEF/ITRF (meters)
        sat_ecef: Satellite position in ECEF/ITRF (meters)

    Returns:
        Range in meters
    """
    diff = sat_ecef - station_ecef
    return np.linalg.norm(diff)


def validate_slr_data(df: pd.DataFrame) -> Dict:
    """
    Validate SLR data quality

    Args:
        df: DataFrame with SLR observations

    Returns:
        Dictionary with validation results
    """
    record_types = df['record_type'] if 'record_type' in df.columns else pd.Series(dtype=object)
    validation = {
        'total_records': len(df),
        'normal_points': int((record_types == 'normal_point').sum()),
        'full_rate': int((record_types == 'full_rate').sum()),
        'unique_stations': df['station_id'].nunique() if 'station_id' in df.columns else 0,
        'unique_targets': df['target_id'].nunique() if 'target_id' in df.columns else 0,
        'time_span_start': df['epoch'].min() if len(df) > 0 else None,
        'time_span_end': df['epoch'].max() if len(df) > 0 else None,
        'mean_sigma_m': df['sigma_m'].mean() if 'sigma_m' in df.columns else None,
        'outliers': [],
        'valid': True
    }

    # Check for range outliers (LEO satellites typically 300-2000 km slant range)
    if 'range_m' in df.columns:
        min_range = 300000  # 300 km
        max_range = 3000000  # 3000 km (generous upper bound)
        outliers = df[(df['range_m'] < min_range) | (df['range_m'] > max_range)]
        if len(outliers) > 0:
            validation['outliers'] = len(outliers)
            validation['valid'] = False

    # Check for negative sigmas
    if 'sigma_m' in df.columns:
        negative_sigma = df[df['sigma_m'] < 0]
        if len(negative_sigma) > 0:
            validation['valid'] = False

    return validation


def parse_legacy_ilrs_np_file(filepath: str) -> Tuple[Dict, List[Dict]]:
    """
    Parse the historic (pre-CRD) ILRS normal-point format.

    Fixed-width 54/55-byte records. Data records are preceded by an indicator
    line '99999' followed by a header record carrying the pass date (2-digit
    year + day-of-year) and station pad id. Data record fields used here:
    columns 1-12 time of day in 0.1 microsecond units, 13-24 two-way
    time-of-flight in picoseconds, 25-31 bin RMS in picoseconds, 44-47 number
    of compressed raw ranges.

    Spec: docs/standards/ilrs-npt-legacy-format.html (archived NASA/ILRS
    page; header columns 8-9 are "Year of century").  The spec defines NO
    century pivot for the two-digit year; this parser uses the STATIC rule
    ``year2 > 85 -> 19xx else 20xx`` -- a documented convention for the NPT
    archive era (1986 through the 2012-05-01 CRD cutover), not a spec
    requirement.  REVIEW_FINDINGS 6.2.1 rule 4 prefers mission-window
    pivots; the dormant failure mode (a pre-1986 pass) is accepted and
    documented here.

    WP8 hardening (REVIEW_FINDINGS 6.3.7): every structurally invalid line
    is a COUNTED rejection in the returned metadata
    (``records_parsed``/``records_rejected``/``reject_reasons`` reason ->
    count), never a silent ``continue``.

    Returns:
        Tuple of (metadata dict, list of normal point records)
    """
    metadata = {
        'station_id': '',
        'target_id': '',
        'target_name': '',
        'wavelength': 0.0,
        'records_parsed': 0,
        'records_rejected': 0,
        'reject_reasons': {},
        'non_utf8_sanitized_copy': False,
    }
    records: List[Dict] = []
    current_date = None
    current_station = ''
    current_target = ''

    def _reject(reason: str) -> None:
        metadata['records_rejected'] += 1
        metadata['reject_reasons'][reason] = metadata['reject_reasons'].get(reason, 0) + 1

    text, non_utf8 = read_slr_text(filepath)
    metadata['non_utf8_sanitized_copy'] = non_utf8
    lines = [line.rstrip('\n') for line in text.splitlines()]

    expect_header = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line == '99999':
            expect_header = True
            continue
        if expect_header:
            expect_header = False
            if len(line) < 16 or not line[:12].strip().isdigit():
                _reject('malformed_header_record')
                continue
            try:
                # Static pivot, documented against the archived NPT spec
                # (see docstring): 86-99 -> 19xx, 00-85 -> 20xx.
                year2 = int(line[7:9])
                doy = int(line[9:12])
                current_date = datetime(1900 + year2 if year2 > 85 else 2000 + year2,
                                        1, 1) + timedelta(days=doy - 1)
            except ValueError:
                _reject('malformed_header_record')
                continue
            current_station = line[12:16].strip()
            current_target = line[0:7].strip()
            if not metadata['station_id']:
                metadata['station_id'] = current_station
                metadata['target_id'] = current_target
            continue
        if current_date is None:
            _reject('data_without_active_pass')
            continue
        if len(line) < 47:
            _reject('short_data_record')
            continue
        if not line[:12].isdigit():
            _reject('malformed_record')
            continue
        try:
            seconds_of_day = (int(line[0:12]) % 864000000000) / 1e7
            time_of_flight = int(line[12:24]) * 1e-12  # ps -> seconds (two-way)
            bin_rms_ps = int(line[24:31])
            num_returns = int(line[43:47])
        except ValueError:
            _reject('malformed_record')
            continue

        epoch = current_date + timedelta(seconds=seconds_of_day)
        records.append({
            'record_type': 'normal_point',
            'epoch': epoch,
            'time_of_flight_s': time_of_flight,
            'range_m': (time_of_flight / 2) * SPEED_OF_LIGHT,
            'sigma_m': (bin_rms_ps * 1e-12 / 2) * SPEED_OF_LIGHT,
            'num_returns': num_returns,
            'station_id': current_station,
            'target_id': current_target,
        })
        metadata['records_parsed'] += 1

    return metadata, records


def slr_to_dataframe(
    filepath: str,
    mission_window: Optional[Tuple[object, object]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse an SLR file to a DataFrame, routing strictly by CONTENT.

    Dispatch (``processors.slr_formats.detect_slr_format``):

    - CRD header records (H1/H2/H3/H4/C0..., case-insensitive) ->
      ``parse_crd_file``
    - first non-empty line '99999' -> ``parse_legacy_ilrs_np_file``
    - MERIT-II fixed-width records -> ``slr_formats.parse_merit_monthly_file``
      with the mission window resolved as (in order): the explicit
      ``mission_window`` argument; else the constant-table lookup for the
      ILRS satellite id carried by the file's own first record
      (``slr_formats.SAT_MISSION_WINDOWS`` / ``ILRS_TARGET_TO_SAT``); else
      no window -- the parser falls back to the launch-year anchor and the
      returned validation dict carries the counted ``window_unavailable``
      marker (never a crash).
    - anything else -> ``UnsupportedSLRFormatError``

    There is deliberately NO fallback guess parser: the retired
    ``parse_normal_point_file`` force-fitted any >=8-token line with a Y2K
    pivot and fabricated records (root-cause fix A4).

    Args:
        filepath: Path to SLR data file
        mission_window: Optional ``(start, end)`` datetimes gating the
            MERIT-II two-digit-year resolution.  ``None`` entries inside
            the tuple mean "open" on that side.

    Returns:
        Tuple of (DataFrame, validation_results).  Family parse ledgers
        ride on the validation dict: ``merit_records_rejected`` /
        ``merit_reject_reasons`` / ``merit_window`` / ``window_unavailable``
        (MERIT), ``npt_records_rejected`` / ``npt_reject_reasons`` (legacy
        NPT), ``crd_records_rejected`` / ``crd_reject_reasons`` (CRD).
        Cross-family: ``corrupt_records_rejected`` /
        ``corrupt_reject_reasons`` (rows rejected as corrupt parses,
        currently ``range_below_band_corrupt``: one-way range below the
        physical slant-range floor) and ``non_utf8_sanitized_copy`` (True
        iff undecodable bytes were replaced while reading the file).

    Raises:
        UnsupportedSLRFormatError: if the file content matches no supported
            format (file-level failure; callers must not silently skip it).
    """
    # errors="replace" on the peek too: a binary-junk byte in the first
    # lines must not crash before the format detection even runs.
    with open(filepath, 'r', errors='replace') as handle:
        first_lines = [handle.readline() for _ in range(10)]

    file_format = detect_slr_format(first_lines)
    if file_format == 'html_error_page':
        # Saved provider error page, not SLR data (acquisition artifact).
        # Counted rejection per P1/§8.4: no records fabricated, no crash.
        return pd.DataFrame(), {
            "file_format": "html_error_page",
            "html_error_page_not_data": 1,
            "records": 0,
        }
    merit_meta: Optional[Dict] = None
    npt_meta: Optional[Dict] = None
    crd_meta: Optional[CRDHeader] = None
    if file_format == 'crd':
        crd_meta, records = parse_crd_file(filepath)
    elif file_format == 'legacy_npt':
        npt_meta, records = parse_legacy_ilrs_np_file(filepath)
    elif file_format == 'merit_monthly':
        window = mission_window
        if window is None:
            # Content-driven fallback: the file's own first record carries
            # the ILRS satellite id (columns 1-7).
            target_id = _first_nonempty_line(first_lines)[:7].lstrip(' ')
            window = mission_window_for_target(target_id)
        if window is not None:
            window_start, window_end = window
        else:
            window_start, window_end = None, None
        merit_meta, records = parse_merit_monthly_file(
            filepath,
            window_start=window_start,
            window_end=window_end,
        )
    else:
        raise UnsupportedSLRFormatError(
            f"unrecognized SLR content in {filepath!r}: "
            f"format detection returned 'unknown'; refusing to guess"
        )

    df = pd.DataFrame(records)

    # Corrupt-parse rejection (P1): a one-way range below the physical
    # slant-range floor of every dataset target is a corrupt parse, not a
    # measurement -- rejected with a counted reason, never retained and
    # never silently dropped (the known case: 3 topex-poseidon legacy
    # records at 0.058-223.6 m, tof ~4e-10 s).
    corrupt_reasons: Dict[str, int] = {}
    if len(df) > 0 and 'range_m' in df.columns:
        rng = pd.to_numeric(df['range_m'], errors='coerce')
        bad = rng < _CORRUPT_RANGE_FLOOR_M
        n_bad = int(bad.sum())
        if n_bad:
            df = df[~bad].reset_index(drop=True)
            corrupt_reasons['range_below_band_corrupt'] = n_bad

    if len(df) > 0:
        # Ensure datetime type for epoch
        df['epoch'] = pd.to_datetime(df['epoch'])

        # Sort by epoch
        df = df.sort_values('epoch').reset_index(drop=True)

    validation = validate_slr_data(df)
    if merit_meta is not None:
        validation['merit_records_rejected'] = merit_meta.get('records_rejected', 0)
        validation['merit_reject_reasons'] = dict(merit_meta.get('reject_reasons', {}))
        validation['merit_window'] = merit_meta.get('window', (None, None))
        validation['window_unavailable'] = not merit_meta.get('window_applied', False)
    if npt_meta is not None:
        # Legacy NPT parse ledger (WP8 / REVIEW_FINDINGS 6.3.7): counted
        # rejections surfaced through the same validation channel as MERIT.
        validation['npt_records_rejected'] = npt_meta.get('records_rejected', 0)
        validation['npt_reject_reasons'] = dict(npt_meta.get('reject_reasons', {}))
    if crd_meta is not None:
        # CRD parse ledger (fix round 2): counted rejections + multi-session
        # accounting, same channel as MERIT/NPT.
        validation['crd_records_rejected'] = crd_meta.records_rejected
        validation['crd_reject_reasons'] = dict(crd_meta.reject_reasons)
    if corrupt_reasons:
        validation['corrupt_records_rejected'] = sum(corrupt_reasons.values())
        validation['corrupt_reject_reasons'] = corrupt_reasons
    validation['non_utf8_sanitized_copy'] = bool(
        (crd_meta is not None and crd_meta.non_utf8_sanitized_copy)
        or (npt_meta is not None and npt_meta.get('non_utf8_sanitized_copy'))
        or (merit_meta is not None and merit_meta.get('non_utf8_sanitized_copy'))
    )

    return df, validation


def _first_nonempty_line(lines: List[str]) -> str:
    """First non-empty line (newline stripped) of a peeked head."""
    for raw in lines:
        line = raw.rstrip('\r\n')
        if line.strip():
            return line
    return ''


def process_slr_file(filepath: str, output_dir: str = None) -> pd.DataFrame:
    """
    Process an SLR data file and optionally save to CSV

    Args:
        filepath: Path to SLR file
        output_dir: Optional output directory

    Returns:
        DataFrame with processed data
    """
    df, validation = slr_to_dataframe(filepath)

    print(f"Parsed SLR file: {len(df)} records")
    print(f"  Normal points: {validation['normal_points']}")
    print(f"  Unique stations: {validation['unique_stations']}")
    if not validation['valid']:
        print(f"  Warning: {validation['outliers']} outlier records detected")

    if output_dir and len(df) > 0:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(filepath)
        output_filename = os.path.splitext(filename)[0] + '.csv'
        output_path = os.path.join(output_dir, output_filename)
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return df


def merge_slr_with_orbit(slr_df: pd.DataFrame,
                         orbit_df: pd.DataFrame,
                         station_coords: Dict[str, np.ndarray],
                         time_tolerance_sec: float = 1.0) -> pd.DataFrame:
    """
    Merge SLR observations with orbit data and compute residuals

    Args:
        slr_df: DataFrame with SLR observations
        orbit_df: DataFrame with orbit positions (x, y, z in ITRF)
        station_coords: Dict mapping station_id to ECEF position [x, y, z]
        time_tolerance_sec: Maximum time difference for matching

    Returns:
        DataFrame with SLR observations and computed residuals
    """
    results = []

    for _, slr_row in slr_df.iterrows():
        slr_epoch = slr_row['epoch']
        station_id = slr_row['station_id']

        # Get station coordinates
        if station_id not in station_coords:
            continue
        station_ecef = station_coords[station_id]

        # Find closest orbit epoch
        time_diffs = abs((orbit_df['epoch'] - slr_epoch).dt.total_seconds())
        min_idx = time_diffs.idxmin()
        min_diff = time_diffs[min_idx]

        if min_diff <= time_tolerance_sec:
            orbit_row = orbit_df.loc[min_idx]
            sat_ecef = np.array([orbit_row['x'], orbit_row['y'], orbit_row['z']])

            # Compute geometric range
            computed_range = compute_range_from_orbit(station_ecef, sat_ecef)

            # Compute residual
            observed_range = slr_row['range_m']
            residual = calculate_residual(observed_range, computed_range)

            result = slr_row.to_dict()
            result['computed_range_m'] = computed_range
            result['residual_m'] = residual
            result['time_diff_sec'] = min_diff
            results.append(result)

    return pd.DataFrame(results)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python slr_processor.py <crd_or_npt_file> [output_dir]")
        sys.exit(1)

    filepath = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    df = process_slr_file(filepath, output_dir)

    if len(df) > 0:
        print(f"\nSample data:")
        print(df.head())
