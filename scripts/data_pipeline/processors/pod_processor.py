"""
Precision Orbit Determination (POD) Data Processing Module

This module provides parsers for precision orbit data formats:
1. SP3 format - Standard Product 3 (used by GFZ, IGS, etc.)
2. EOF format - Earth Observation Format (Sentinel XML orbit files)

Outputs unified DataFrame format with columns:
sat_id, epoch, x, y, z, vx, vy, vz, sigma_x, sigma_y, sigma_z
"""

import os
import re
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from lxml import etree
from astropy.time import Time

from processors.timebase import sp3_time_system, tai_datetime_to_utc


# SP3 Format Constants
SP3_POS_ACCURACY_BASE = 1.25  # mm, base accuracy unit
SP3_VEL_ACCURACY_BASE = 0.1  # mm/s, base accuracy unit (for SP3-d)

# WGS-84 Earth gravitational parameter (m^3/s^2) used ONLY for the velocity
# unit arbitration below (same value as physical_qc.EARTH_MU_M3_PER_S2;
# kept local because this module must not import physical_qc for mu).
_SP3_UNIT_MU = 3.986004418e14

# Velocity-unit arbitration bands (|v_raw| / sqrt(mu/r)).  For LEO targets
# |v_ecef| / v_circular lies in ~[0.90, 1.10] (the ECEF/inertial offset is
# the |omega x r| ~ 0.35-0.53 km/s component), so the RAW ratio is ~1 for
# m/s-as-written products and ~10 for spec dm/s products; the two unit
# interpretations differ by a factor of 10 and the bands separate them
# with wide margin.
_SP3_VEL_RATIO_MPS = (0.80, 1.30)
_SP3_VEL_RATIO_DMPS = (8.0, 13.0)
# Sampled epochs used for the per-file median ratio (median is insensitive
# to the exact sampling).
_SP3_VEL_UNIT_MAX_SAMPLES = 99


def _sp3_velocity_unit_scale(lines: List[str]) -> Tuple[float, str]:
    """Decide this file's V-record velocity scale (content-driven).

    The SP3-c/d specification defines V-record velocities in dm/s, but the
    GRG product family in the legacy archive (29 files: hy-2a, jason-2,
    saral, topex-poseidon) writes them in m/s -- measured against the
    vis-viva identity: median |v_raw|/sqrt(mu/r) is ~0.97-1.01 for grg
    (m/s as written) and ~9.7-10.0 for every other series (spec dm/s).
    No header token separates them (grg and lca files share agency=LCA,
    type=FIT, coord=ITR05 while differing in unit), so the unit class is
    arbitrated per FILE by the physics identity itself -- the exact defect
    class REVIEW_FINDINGS 6.2.4's vis-viva bound exists to catch ("速度
    单位错误"), applied here as a two-hypothesis resolution between the
    spec unit and the as-written unit.

    Returns ``(scale, basis)``: scale 1.0 with basis ``m/s_as_written`` when
    the median ratio lands in the m/s band, 0.1 (spec) with basis
    ``dm/s_spec`` when it lands in the dm/s band, and 0.1 with basis
    ``dm/s_spec_default_unresolved`` when there is no P+V evidence or the
    ratio matches neither band (spec default; downstream L2 assertions
    then adjudicate the file).
    """
    if not any(line.startswith('V') for line in lines):
        return 0.1, 'no_v_records'
    ratios: List[float] = []
    position = None
    for line in lines:
        if line.startswith('*'):
            position = None
        elif line.startswith('P'):
            try:
                position = (
                    float(line[4:18]) * 1000.0,
                    float(line[18:32]) * 1000.0,
                    float(line[32:46]) * 1000.0,
                )
            except (ValueError, IndexError):
                position = None
        elif line.startswith('V') and position is not None:
            try:
                vx = float(line[4:18])
                vy = float(line[18:32])
                vz = float(line[32:46])
            except (ValueError, IndexError):
                continue
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            radius = math.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)
            if radius <= 0:
                continue
            ratios.append(speed / math.sqrt(_SP3_UNIT_MU / radius))
            if len(ratios) >= _SP3_VEL_UNIT_MAX_SAMPLES:
                break
    if not ratios:
        return 0.1, 'dm/s_spec_default_unresolved'
    median_ratio = float(np.median(ratios))
    if _SP3_VEL_RATIO_MPS[0] <= median_ratio <= _SP3_VEL_RATIO_MPS[1]:
        return 1.0, 'm/s_as_written'
    if _SP3_VEL_RATIO_DMPS[0] <= median_ratio <= _SP3_VEL_RATIO_DMPS[1]:
        return 0.1, 'dm/s_spec'
    return 0.1, 'dm/s_spec_default_unresolved'

# SP3-c/d spec sentinel ("clock not available / bad clock"): the clock (and
# clock-rate) field value 999999.999999 marks MISSING data and must map to
# NaN, never enter statistics as a measurement (REVIEW_FINDINGS 6.2.4).
# Imported from physical_qc so the constant exists once in the repository.
from processors.physical_qc import SP3_CLOCK_SENTINEL  # noqa: E402


class SP3Header:
    """Container for SP3 file header information"""

    def __init__(self):
        self.version: str = ''
        self.pos_vel_flag: str = 'P'  # P=position only, V=position+velocity
        self.start_epoch: datetime = None
        self.num_epochs: int = 0
        self.data_used: str = ''
        self.coord_system: str = ''
        self.orbit_type: str = ''
        self.agency: str = ''
        self.gps_week: int = 0
        self.seconds_of_week: float = 0.0
        self.epoch_interval: float = 0.0
        self.mjd_start: int = 0
        self.frac_day: float = 0.0
        self.num_sats: int = 0
        self.sat_ids: List[str] = []
        self.sat_accuracy: Dict[str, int] = {}
        self.file_type: str = ''
        self.time_system: str = ''
        # SP3-c/d spec (docs/standards/sp3d-format-spec.pdf), first %c line:
        # col 4 file type, cols 6-7 tracking system, cols 10-12 POSITION time
        # system, cols 14-16 CLOCK time system.  ``time_system`` is the
        # position time system, filled via processors.timebase delegation.
        self.tracking_system: str = ''
        self.clock_time_system: str = ''
        self.base_pos_vel: float = 0.0
        self.base_clk_rate: float = 0.0
        # Raw header lines (everything before the first '*' epoch line) so
        # callers can run authoritative %c time-system parsing (see
        # processors.timebase.sp3_time_system).
        self.header_lines: List[str] = []


def parse_sp3_header(lines: List[str]) -> SP3Header:
    """
    Parse SP3 file header to extract metadata

    Args:
        lines: List of file lines

    Returns:
        SP3Header object with parsed metadata
    """
    header = SP3Header()
    sat_ids = []
    sat_accuracy = {}
    header_lines: List[str] = []
    percent_c_seen = False

    for i, line in enumerate(lines):
        if line.startswith('*'):
            # End of header, start of data
            break

        header_lines.append(line)
        if i == 0:
            # Line 1: Version, P/V flag, start epoch, num epochs, data type
            header.version = line[0:2]
            header.pos_vel_flag = line[2]
            year = int(line[3:7])
            month = int(line[8:10])
            day = int(line[11:13])
            hour = int(line[14:16])
            minute = int(line[17:19])
            second = float(line[20:31])
            header.start_epoch = datetime(year, month, day, hour, minute, int(second),
                                         int((second % 1) * 1e6))
            header.num_epochs = int(line[32:39])
            header.data_used = line[40:45].strip()
            header.coord_system = line[46:51].strip()
            header.orbit_type = line[52:55].strip()
            header.agency = line[56:60].strip()

        elif i == 1:
            # Line 2: GPS week, seconds of week, epoch interval, MJD
            header.gps_week = int(line[3:7])
            header.seconds_of_week = float(line[8:23])
            header.epoch_interval = float(line[24:38])
            header.mjd_start = int(line[39:44])
            header.frac_day = float(line[45:60])

        elif line.startswith('+ '):
            # Satellite ID lines (can be multiple)
            if i == 2:
                header.num_sats = int(line[3:6])
            # Parse satellite IDs (up to 17 per line, 3 chars each)
            for j in range(17):
                start = 9 + j * 3
                if start + 3 <= len(line):
                    sat_id = line[start:start+3].strip()
                    if sat_id and sat_id != '0' and sat_id != '  0':
                        sat_ids.append(sat_id)

        elif line.startswith('++'):
            # Satellite accuracy lines
            for j in range(17):
                start = 9 + j * 3
                if start + 3 <= len(line):
                    try:
                        acc = int(line[start:start+3].strip())
                        if len(sat_ids) > len(sat_accuracy):
                            idx = len(sat_accuracy)
                            if idx < len(sat_ids):
                                sat_accuracy[sat_ids[idx]] = acc
                    except (ValueError, IndexError):
                        pass

        elif line.startswith('%c'):
            # First %c line, SP3-c/d fixed columns (1-based): col 4 file
            # type, cols 6-7 tracking system, cols 10-12 POSITION time
            # system, cols 14-16 CLOCK time system.  The second %c line
            # carries the base numbers and is skipped here.  (The old
            # whitespace-split branch mis-parsed every real product to
            # time_system='cc'; fix per REVIEW_FINDINGS 6.3.2.)
            if not percent_c_seen:
                percent_c_seen = True
                header.file_type = line[3:4].strip()
                header.tracking_system = line[5:7].strip()
                header.clock_time_system = line[13:16].strip()

        elif line.startswith('%f'):
            # Base for position/velocity and clock
            parts = line[3:].split()
            if len(parts) >= 2:
                try:
                    header.base_pos_vel = float(parts[0])
                    header.base_clk_rate = float(parts[1])
                except ValueError:
                    pass

    header.sat_ids = sat_ids[:header.num_sats] if header.num_sats > 0 else sat_ids
    header.sat_accuracy = sat_accuracy
    header.header_lines = header_lines
    # Position time system: delegated to the single authority
    # processors.timebase.sp3_time_system (columns 10-12 of the first %c
    # line).  Raises ValueError when the %c line is missing or still holds
    # the spec placeholder -- an undeclared time scale must block, not
    # default (REVIEW_FINDINGS 6.2.1 rule 2, 6.3.2).
    header.time_system = sp3_time_system(header_lines)

    return header


def parse_sp3_epochs(lines: List[str], header: SP3Header, stats: Optional[Dict] = None) -> List[Dict]:
    """
    Parse SP3 position/velocity data epochs

    Args:
        lines: List of file lines
        header: Parsed SP3Header object
        stats: Optional dict filled with the parse ledger (counted
            rejections, never silent drops; REVIEW_FINDINGS 6.3.2):
            ``epoch_lines``, ``epoch_lines_rejected``, ``p_records``,
            ``p_records_rejected``, ``v_records``, ``v_records_rejected``,
            ``clock_sentinel_rows``.

    Returns:
        List of record dictionaries.  Clock (and clock-rate) fields holding
        the SP3 spec sentinel 999999.999999 are mapped to ``None`` (missing),
        never surfaced as measurements.  V-record velocities are scaled by
        the per-file unit arbitration (``_sp3_velocity_unit_scale``; grg
        products write m/s, the spec default is dm/s) with the decision
        recorded in ``stats['velocity_unit_scale']`` /
        ``stats['velocity_unit_basis']``.
    """
    velocity_scale, velocity_basis = _sp3_velocity_unit_scale(lines)
    records = []
    current_epoch = None
    epoch_positions = {}  # sat_id -> position data for current epoch
    epoch_velocities = {}  # sat_id -> velocity data for current epoch
    local_stats: Dict = {
        'epoch_lines': 0,
        'epoch_lines_rejected': 0,
        'p_records': 0,
        'p_records_rejected': 0,
        'v_records': 0,
        'v_records_rejected': 0,
        'clock_sentinel_rows': 0,
        'velocity_unit_scale': velocity_scale,
        'velocity_unit_basis': velocity_basis,
    }

    def _finish_epoch(target_records: List[Dict], epoch, positions: Dict, velocities: Dict) -> None:
        for sat_id, pos in positions.items():
            record = {
                'sat_id': sat_id,
                'epoch': epoch,
                'x': pos['x'],
                'y': pos['y'],
                'z': pos['z'],
                'clock': pos.get('clock'),
                'sigma_x': pos.get('sigma_x'),
                'sigma_y': pos.get('sigma_y'),
                'sigma_z': pos.get('sigma_z'),
                'sigma_clock': pos.get('sigma_clock'),
            }
            if sat_id in velocities:
                vel = velocities[sat_id]
                record.update({
                    'vx': vel['vx'],
                    'vy': vel['vy'],
                    'vz': vel['vz'],
                    'clock_rate': vel.get('clock_rate'),
                })
            target_records.append(record)

    for line in lines:
        if line.startswith('*'):
            # Save previous epoch data if exists
            if current_epoch and epoch_positions:
                _finish_epoch(records, current_epoch, epoch_positions, epoch_velocities)

            # Parse new epoch
            local_stats['epoch_lines'] += 1
            epoch_positions = {}
            epoch_velocities = {}
            try:
                year = int(line[3:7])
                month = int(line[8:10])
                day = int(line[11:13])
                hour = int(line[14:16])
                minute = int(line[17:19])
                second = float(line[20:31])
                current_epoch = datetime(year, month, day, hour, minute, int(second),
                                        int((second % 1) * 1e6))
            except (ValueError, IndexError):
                local_stats['epoch_lines_rejected'] += 1
                current_epoch = None

        elif line.startswith('P'):
            # Position record
            sat_id = line[1:4].strip()
            try:
                x = float(line[4:18]) * 1000  # km to m
                y = float(line[18:32]) * 1000
                z = float(line[32:46]) * 1000
                clock = float(line[46:60]) if len(line) >= 60 else None
                if clock is not None and clock == SP3_CLOCK_SENTINEL:
                    # Spec sentinel: clock unavailable -> missing, counted
                    clock = None
                    local_stats['clock_sentinel_rows'] += 1

                # Position standard deviations (if present, columns 61-73)
                sigma_x = sigma_y = sigma_z = sigma_clock = None
                if len(line) >= 73:
                    try:
                        sx = int(line[61:63].strip()) if line[61:63].strip() else None
                        sy = int(line[64:66].strip()) if line[64:66].strip() else None
                        sz = int(line[67:69].strip()) if line[67:69].strip() else None
                        sc = int(line[70:73].strip()) if line[70:73].strip() else None

                        base = header.base_pos_vel if header.base_pos_vel > 0 else SP3_POS_ACCURACY_BASE
                        if sx: sigma_x = base ** sx / 1000  # mm to m
                        if sy: sigma_y = base ** sy / 1000
                        if sz: sigma_z = base ** sz / 1000
                    except ValueError:
                        pass

                epoch_positions[sat_id] = {
                    'x': x, 'y': y, 'z': z, 'clock': clock,
                    'sigma_x': sigma_x, 'sigma_y': sigma_y, 'sigma_z': sigma_z,
                    'sigma_clock': sigma_clock
                }
                local_stats['p_records'] += 1
            except (ValueError, IndexError):
                local_stats['p_records_rejected'] += 1

        elif line.startswith('V'):
            # Velocity record (SP3-c/d with V flag); unit per the per-file
            # arbitration (spec dm/s -> x0.1; grg products write m/s).
            sat_id = line[1:4].strip()
            try:
                vx = float(line[4:18]) * velocity_scale
                vy = float(line[18:32]) * velocity_scale
                vz = float(line[32:46]) * velocity_scale
                clock_rate = float(line[46:60]) if len(line) >= 60 else None
                if clock_rate is not None and clock_rate == SP3_CLOCK_SENTINEL:
                    clock_rate = None
                    local_stats['clock_sentinel_rows'] += 1

                epoch_velocities[sat_id] = {
                    'vx': vx, 'vy': vy, 'vz': vz,
                    'clock_rate': clock_rate
                }
                local_stats['v_records'] += 1
            except (ValueError, IndexError):
                local_stats['v_records_rejected'] += 1

        elif line.startswith('EOF'):
            break

    # Don't forget the last epoch
    if current_epoch and epoch_positions:
        _finish_epoch(records, current_epoch, epoch_positions, epoch_velocities)

    if stats is not None:
        stats.update(local_stats)

    return records


def validate_sp3_data(df: pd.DataFrame, max_gap_hours: float = 2.0) -> Dict:
    """
    Validate SP3 data quality

    Args:
        df: DataFrame with SP3 data
        max_gap_hours: Maximum allowed gap between epochs

    Returns:
        Dictionary with validation results

    The caller's frame is NEVER mutated: the outlier radius is a local
    Series (the legacy ``df['r'] = ...`` leaked a helper column into the
    caller's DataFrame).
    """
    validation = {
        'total_records': len(df),
        'unique_satellites': df['sat_id'].nunique(),
        'time_span_start': df['epoch'].min(),
        'time_span_end': df['epoch'].max(),
        'has_velocity': 'vx' in df.columns and df['vx'].notna().any(),
        'has_sigma': 'sigma_x' in df.columns and df['sigma_x'].notna().any(),
        'gaps': [],
        'outliers': [],
        'valid': True
    }

    # Check for time gaps per satellite
    for sat_id in df['sat_id'].unique():
        sat_df = df[df['sat_id'] == sat_id].sort_values('epoch')
        if len(sat_df) > 1:
            time_diffs = sat_df['epoch'].diff().dropna()
            max_gap = time_diffs.max()
            if isinstance(max_gap, timedelta):
                max_gap_hours_found = max_gap.total_seconds() / 3600
                if max_gap_hours_found > max_gap_hours:
                    validation['gaps'].append({
                        'sat_id': sat_id,
                        'max_gap_hours': max_gap_hours_found
                    })

    # Check for position outliers (rough sanity check for LEO: 6400-8000 km
    # from Earth center).  The radius stays a LOCAL Series -- no in-place
    # ``df['r']`` column side effect on the caller's frame.
    radius_m = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    outlier_mask = (radius_m < 6400000) | (radius_m > 8000000)
    if outlier_mask.any():
        outliers = df.loc[outlier_mask, ['sat_id', 'epoch']].copy()
        outliers['r'] = radius_m[outlier_mask]
        validation['outliers'] = outliers.to_dict('records')
        validation['valid'] = False

    if validation['gaps']:
        validation['valid'] = False

    return validation


def sp3_to_dataframe(filepath: str) -> Tuple[pd.DataFrame, SP3Header, Dict]:
    """
    Parse SP3 file and convert to unified DataFrame format

    Args:
        filepath: Path to SP3 file

    Returns:
        Tuple of (DataFrame, SP3Header, validation_results)
    """
    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    header = parse_sp3_header(lines)
    parse_stats: Dict = {}
    records = parse_sp3_epochs(lines, header, stats=parse_stats)

    df = pd.DataFrame(records)

    # P/V record integrity vs header declaration (counted warning, never a
    # silent mismatch; REVIEW_FINDINGS 6.3.2 "P/V record integrity =
    # num_epochs x sats accounting").
    if header.num_epochs > 0 and header.num_sats > 0:
        expected_p = header.num_epochs * header.num_sats
        if parse_stats.get('p_records', 0) != expected_p:
            validation_warning = (
                f"parsed {parse_stats.get('p_records', 0)} P records, "
                f"expected {expected_p} (header {header.num_epochs} epochs x "
                f"{header.num_sats} satellites)"
            )
        else:
            validation_warning = None
    else:
        validation_warning = None

    # Ensure all expected columns exist
    expected_cols = ['sat_id', 'epoch', 'x', 'y', 'z', 'vx', 'vy', 'vz',
                     'sigma_x', 'sigma_y', 'sigma_z', 'clock', 'clock_rate']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # Convert epoch to datetime if needed
    if len(df) > 0 and not pd.api.types.is_datetime64_any_dtype(df['epoch']):
        df['epoch'] = pd.to_datetime(df['epoch'])

    validation = validate_sp3_data(df)
    validation['sp3_parse_stats'] = parse_stats
    if validation_warning is not None:
        validation['p_record_count_warning'] = validation_warning

    return df, header, validation


# EOF (Sentinel) Format Parser

class EOFMetadata:
    """Container for EOF file metadata"""

    def __init__(self):
        self.mission: str = ''
        self.satellite_id: str = ''
        self.validity_start: datetime = None
        self.validity_stop: datetime = None
        self.creation_time: datetime = None
        self.source: str = ''
        self.orbit_type: str = ''  # POE (Precise), MOE (Medium), ROE (Restituted)
        self.fixed_header: Dict = {}
        # Variable_Header declarations (REVIEW_FINDINGS 6.3.3: time base and
        # frame must be read from the product's own metadata).
        self.time_reference: str = ''
        self.ref_frame: str = ''
        # OSV parse ledger: counted, never silent (never continue).
        self.osv_total: int = 0
        self.osv_rejected: int = 0
        self.reject_reasons: Dict[str, int] = {}
        self.non_nominal_quality: int = 0


def parse_eof_xml(filepath: str) -> Tuple[EOFMetadata, List[Dict]]:
    """
    Parse Sentinel EOF (Earth Observation Format) XML orbit file

    Args:
        filepath: Path to EOF XML file

    Returns:
        Tuple of (EOFMetadata, list of OSV records)

    Time base (REVIEW_FINDINGS 6.3.3): the ``<UTC>`` element is preferred;
        when absent the ``<TAI>`` tag is converted to UTC through the
        authoritative IERS/BIPM leap-second table
        (``processors.timebase.tai_datetime_to_utc``) -- a TAI string is
        never used raw.  OSVs with no parsable time element, or malformed
        numeric fields, are counted rejections (metadata.reject_reasons);
        ``Quality != NOMINAL`` keeps the row but increments
        ``metadata.non_nominal_quality``.
    """
    tree = etree.parse(filepath)
    root = tree.getroot()

    # Handle namespace if present
    nsmap = root.nsmap
    ns = {'eof': nsmap.get(None, '')} if nsmap else {}

    def find_element(parent, tag):
        """Find element with or without namespace"""
        elem = parent.find(tag, ns) if ns else parent.find(tag)
        if elem is None:
            elem = parent.find(f'.//{tag}')
        if elem is None:
            # Try without namespace
            for child in parent.iter():
                if child.tag.endswith(tag):
                    return child
        return elem

    def find_text(parent, tag, default=''):
        elem = find_element(parent, tag)
        return elem.text if elem is not None and elem.text else default

    metadata = EOFMetadata()

    # Parse fixed header
    fixed_header = find_element(root, 'Fixed_Header')
    if fixed_header is not None:
        metadata.mission = find_text(fixed_header, 'Mission')
        metadata.source = find_text(fixed_header, 'Source')

        validity = find_element(fixed_header, 'Validity_Period')
        if validity is not None:
            start_str = find_text(validity, 'Validity_Start')
            stop_str = find_text(validity, 'Validity_Stop')
            if start_str:
                metadata.validity_start = parse_eof_datetime(start_str)
            if stop_str:
                metadata.validity_stop = parse_eof_datetime(stop_str)

    # Variable_Header declarations: time base and reference frame are read
    # from the product itself (EOF format / Sentinel POD product specs;
    # REVIEW_FINDINGS 6.2.1 rule 2 + 6.2.2 rule 1 -- no assumption).
    variable_header = find_element(root, 'Variable_Header')
    if variable_header is not None:
        metadata.time_reference = find_text(variable_header, 'Time_Reference').strip()
        metadata.ref_frame = find_text(variable_header, 'Ref_Frame').strip()

    # Parse OSV (Orbit State Vector) records
    osv_records = []
    osv_list = root.findall('.//OSV')
    if not osv_list:
        osv_list = root.findall('.//{*}OSV')

    for osv in osv_list:
        metadata.osv_total += 1

        def _reject(reason: str) -> None:
            metadata.osv_rejected += 1
            metadata.reject_reasons[reason] = metadata.reject_reasons.get(reason, 0) + 1

        try:
            tai_str = find_text(osv, 'TAI')
            utc_str = find_text(osv, 'UTC')

            # Parse time (prefer UTC; TAI converted authoritatively).
            epoch = None
            if utc_str:
                epoch = parse_eof_datetime(utc_str)
            elif tai_str:
                tai_epoch = parse_eof_datetime(tai_str)
                if tai_epoch is not None:
                    epoch = tai_datetime_to_utc(tai_epoch)
            if epoch is None:
                _reject('missing_or_unparsable_epoch')
                continue

            # Position (meters)
            x = float(find_text(osv, 'X', '0'))
            y = float(find_text(osv, 'Y', '0'))
            z = float(find_text(osv, 'Z', '0'))

            # Velocity (m/s)
            vx = float(find_text(osv, 'VX', '0'))
            vy = float(find_text(osv, 'VY', '0'))
            vz = float(find_text(osv, 'VZ', '0'))

            # Quality indicator: non-NOMINAL rows are KEPT but counted
            # (REVIEW_FINDINGS 6.3.3 -- flag, never silently drop).
            quality = find_text(osv, 'Quality').strip()
            if quality and quality != 'NOMINAL':
                metadata.non_nominal_quality += 1

            record = {
                'epoch': epoch,
                'x': x,
                'y': y,
                'z': z,
                'vx': vx,
                'vy': vy,
                'vz': vz,
                'tai_time': tai_str,
                'utc_time': utc_str,
                'quality': quality
            }
            osv_records.append(record)

        except (ValueError, TypeError):
            _reject('malformed_osv')
            continue

    # Extract satellite ID from filename or mission
    if metadata.mission:
        if 'S1A' in metadata.mission or 'SENTINEL-1A' in metadata.mission.upper():
            metadata.satellite_id = 'S1A'
        elif 'S1B' in metadata.mission or 'SENTINEL-1B' in metadata.mission.upper():
            metadata.satellite_id = 'S1B'
        elif 'S2A' in metadata.mission:
            metadata.satellite_id = 'S2A'
        elif 'S2B' in metadata.mission:
            metadata.satellite_id = 'S2B'
        elif 'S3A' in metadata.mission or 'SENTINEL-3A' in metadata.mission.upper():
            metadata.satellite_id = 'S3A'
        elif 'S3B' in metadata.mission:
            metadata.satellite_id = 'S3B'
        else:
            metadata.satellite_id = metadata.mission[:10]

    return metadata, osv_records


def parse_eof_datetime(dt_str: str) -> Optional[datetime]:
    """
    Parse EOF datetime string

    Formats:
    - UTC=2024-01-15T12:00:00
    - TAI=2024-01-15T12:00:37
    - 2024-01-15T12:00:00.000000

    Args:
        dt_str: Datetime string from EOF file

    Returns:
        datetime object or None
    """
    if not dt_str:
        return None

    # Remove TAI= or UTC= prefix
    if '=' in dt_str:
        dt_str = dt_str.split('=')[1]

    # Try different formats
    formats = [
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue

    return None


def eof_to_dataframe(filepath: str) -> Tuple[pd.DataFrame, EOFMetadata]:
    """
    Parse EOF file and convert to unified DataFrame format

    Args:
        filepath: Path to EOF XML file

    Returns:
        Tuple of (DataFrame, EOFMetadata)
    """
    metadata, records = parse_eof_xml(filepath)

    df = pd.DataFrame(records)

    # Add satellite ID to all records
    df['sat_id'] = metadata.satellite_id

    # Reorder columns
    cols = ['sat_id', 'epoch', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'quality']
    df = df[[c for c in cols if c in df.columns]]

    # Convert epoch to datetime
    if len(df) > 0 and not pd.api.types.is_datetime64_any_dtype(df['epoch']):
        df['epoch'] = pd.to_datetime(df['epoch'])

    # Surface the product's own frame/time declarations (best-effort
    # metadata channel; REVIEW_FINDINGS 6.2.2 rule 1).
    df.attrs['eof_ref_frame'] = metadata.ref_frame
    df.attrs['eof_time_reference'] = metadata.time_reference
    df.attrs['eof_parse_stats'] = {
        'osv_total': metadata.osv_total,
        'osv_rejected': metadata.osv_rejected,
        'reject_reasons': dict(metadata.reject_reasons),
        'non_nominal_quality': metadata.non_nominal_quality,
    }

    return df, metadata


def process_pod_file(filepath: str, output_dir: str = None) -> pd.DataFrame:
    """
    Process a POD file (SP3 or EOF format) and optionally save to CSV

    Args:
        filepath: Path to POD file
        output_dir: Optional output directory for CSV

    Returns:
        DataFrame with processed data
    """
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    if ext in ['.sp3', '.eph']:
        df, header, validation = sp3_to_dataframe(filepath)
        print(f"Parsed SP3 file: {len(df)} records, {header.num_sats} satellites")
        if not validation['valid']:
            print(f"  Validation warnings: gaps={len(validation['gaps'])}, "
                  f"outliers={len(validation['outliers'])}")
    elif ext in ['.eof', '.xml']:
        df, metadata = eof_to_dataframe(filepath)
        print(f"Parsed EOF file: {len(df)} records, satellite={metadata.satellite_id}")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.splitext(filename)[0] + '.csv'
        output_path = os.path.join(output_dir, output_filename)
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return df


def process_pod_directory(input_dir: str, output_dir: str = None,
                          pattern: str = '*.sp3') -> pd.DataFrame:
    """
    Process all POD files in a directory

    Args:
        input_dir: Input directory path
        output_dir: Optional output directory
        pattern: Glob pattern for files

    Returns:
        Combined DataFrame
    """
    import glob

    files = glob.glob(os.path.join(input_dir, pattern))
    all_dfs = []

    for filepath in sorted(files):
        try:
            df = process_pod_file(filepath, output_dir)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(['sat_id', 'epoch'])
        return combined

    return pd.DataFrame()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pod_processor.py <sp3_or_eof_file> [output_dir]")
        sys.exit(1)

    filepath = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    df = process_pod_file(filepath, output_dir)
    print(f"\nProcessed {len(df)} records")
    print(df.head())
