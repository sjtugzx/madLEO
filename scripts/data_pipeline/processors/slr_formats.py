"""Content-based SLR format registry and MERIT-II parser.

This module kills the A4 root cause: the retired guess-parser
(``parse_normal_point_file``) force-fitted any whitespace token line with a
Y2K pivot (year < 50 -> 2000s), fabricating records with epochs 1950-2049,
~8e11 m ranges, and glued station/target ids.  SLR files are now routed by
their CONTENT only -- file names and extensions in the wild carry no
reliable signal (the legacy archive contains CRD content in ``.npt`` and
``.frd`` files, MERIT-II content in extension-less files, and mixed
families inside identically named directories).

Supported families (see ``detect_slr_format``):

- ``"crd"``        ILRS Consolidated Ranging Data; first non-empty line
                   starts with an H1/H2/H3/H4/C0-style header record
                   (case-insensitive -- many provider files use lowercase).
- ``"legacy_npt"`` historic pre-CRD ILRS normal-point format; first
                   non-empty line is the ``99999`` pass indicator.
- ``"merit_monthly"`` MERIT-II / ILRS full-rate V2 ("frv2") merged files:
                   fixed-width 130-character records with no marker lines.
                   Layout: docs/standards/merit2-frv2-format.md.
- ``"unknown"``    anything else -> callers must refuse to parse.

Example::

    from processors.slr_formats import detect_slr_format
    with open(path) as handle:
        fmt = detect_slr_format([handle.readline() for _ in range(10)])
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

SPEED_OF_LIGHT_M_PER_S = 299792458.0

# MERIT-II column layout (1-based, inclusive) per the archived ILRS spec.
_MERIT_SAT_ID = (1, 7)        # ILRS satellite id YYXXXAA, leading blank fill
_MERIT_YEAR = (8, 9)          # year of century (blank-padded " 4" occurs in
                              # the archive -- FORTRAN I2 read semantics)
_MERIT_DOY = (10, 12)         # day of year, leading blank fill
_MERIT_TOD = (13, 24)         # seconds of day, 0.1 us granularity
_MERIT_STATION = (25, 28)     # CDP pad number, leading blank fill
_MERIT_AZIMUTH = (33, 39)     # 0.1 millidegree
_MERIT_ELEVATION = (40, 45)   # 0.1 millidegree
_MERIT_TOF = (46, 57)         # two-way time of flight, picoseconds
_MERIT_SIGMA = (58, 64)       # laser range standard deviation, picoseconds
_MERIT_WAVELENGTH = (65, 68)  # 0.1 nm
_MERIT_NP_INDICATOR = (115, 115)
_MERIT_NUM_RETURNS = (116, 119)

# Normal-point window indicator -> window length in seconds
# (MERIT-II spec: 1-4 -> 10 s, 5 -> 5 s, 6 -> 15 s, 7 -> 30 s, 8 -> 60 s,
#  9 -> 120 s, 0 -> not a normal point).
_MERIT_NP_WINDOW_SECONDS = {"1": 10.0, "2": 10.0, "3": 10.0, "4": 10.0,
                            "5": 5.0, "6": 15.0, "7": 30.0, "8": 60.0, "9": 120.0}

# Satellites launched after this two-digit year belong to the 2000s.  The
# boundary is the first artificial satellite (1957): every ILRS target with
# yy in {57..99} launched in the 1900s, {00..56} in the 2000s or later --
# stable well past the operational horizon of this dataset.
_SPUTNIK_BOUNDARY_YEAR2 = 57

# Mission windows (launch -> end of flight operations) per satellite, used
# to resolve MERIT-II two-digit year-of-century fields in production.
# Source of the dates: public mission fact sheets (NASA/CNES/ESA/ESA-CNES/
# ISRO/CNSC launch and passivation announcements) plus the WP4 validation
# contract for jason-1 (launch 2001-12-07, passivation 2013-07-01; the
# validated contract window 2001-04-15..2013-07-31 is kept verbatim).
# Retired satellites carry a padded end (month/quarter boundary past
# passivation -- post-passivation SLR tracking is possible but brief);
# operational satellites are open-ended (``None`` end: no upper gate).
# configs/targets_maneuver_annotations.json carries no launch/passivation
# dates (its release_window is the 2023-2026 acquisition window), so this
# constant table is the single derived lookup, not a second config file.
SAT_MISSION_WINDOWS: Dict[str, Tuple[Optional[datetime], Optional[datetime]]] = {
    "topex-poseidon": (datetime(1992, 8, 10), datetime(2006, 6, 30)),
    "jason-1": (datetime(2001, 4, 15), datetime(2013, 7, 31)),
    "jason-2": (datetime(2008, 6, 20), datetime(2019, 12, 31)),
    "jason-3": (datetime(2016, 1, 17), None),          # operational
    "sentinel-3a": (datetime(2016, 2, 16), None),      # operational
    "sentinel-3b": (datetime(2018, 4, 25), None),      # operational
    "sentinel-6a": (datetime(2020, 11, 21), None),     # operational
    "cryosat-2": (datetime(2010, 4, 8), None),         # operational
    "saral": (datetime(2013, 2, 25), None),            # operational (drifting)
    "hy-2a": (datetime(2011, 8, 16), None),            # end of ops unconfirmed
    "swot": (datetime(2022, 12, 16), None),            # operational
}

# ILRS satellite id (YYXXXAA, as carried by MERIT/CRD/legacy records) ->
# repository sat_id.  Derived from the legacy raw archive itself (the ids
# every file family actually carries per satellite directory, extracted
# 2026-08-28); not copied from an external list.
ILRS_TARGET_TO_SAT: Dict[str, str] = {
    "9205201": "topex-poseidon",
    "0105501": "jason-1",
    "0803201": "jason-2",
    "1600201": "jason-3",
    "1601101": "sentinel-3a",
    "1803901": "sentinel-3b",
    "2008601": "sentinel-6a",
    "1001301": "cryosat-2",
    "1300901": "saral",
    "1104301": "hy-2a",
    "2217301": "swot",
}


def mission_window_for_sat(sat_id: str) -> Tuple[Optional[datetime], Optional[datetime]] | None:
    """Mission window for a repository sat_id, or ``None`` when unknown."""
    return SAT_MISSION_WINDOWS.get(str(sat_id).strip().lower())


def mission_window_for_target(target_id: str) -> Tuple[Optional[datetime], Optional[datetime]] | None:
    """Mission window for an ILRS 7-digit satellite id, or ``None``.

    Accepts both the zero-filled (``"0105501"``) and the blank-stripped
    (``"105501"``) spellings found in the archive.
    """
    key = str(target_id).strip()
    if not key.isdigit():
        return None
    sat_id = ILRS_TARGET_TO_SAT.get(key.lstrip("0").zfill(7))
    return mission_window_for_sat(sat_id) if sat_id else None

_CRD_HEADER_PREFIX = re.compile(r"^(?:h[1-9]|c[0-9])(?:\s|$)", re.IGNORECASE)


class UnsupportedSLRFormatError(FileNotFoundError):
    """Raised when an SLR file's content matches no supported format.

    Subclasses :class:`FileNotFoundError` so generic "cannot use this file"
    handling keeps working, while remaining distinguishable for callers that
    need to report unrecognized content explicitly.
    """


def read_slr_text(filepath: Union[str, Path]) -> Tuple[str, bool]:
    """Read an SLR file as decoded text with undecodable bytes replaced.

    A handful of archived files (measured: 7 -- six cryosat-2 2012 and one
    hy-2a 2013 ``.npt`` with CRD content) carry binary junk inside CRD
    ``60`` compatibility-record free text (18 non-ASCII bytes total).
    Strict UTF-8 crashed the entire file on read; every emitted record
    field of the supported formats is pure fixed-width ASCII, so decoding
    with ``errors="replace"`` is value-identical for the parsed data (the
    junk lives in pass-through records only).

    Returns ``(text, non_utf8_sanitized)`` where the flag is True iff at
    least one U+FFFD replacement character is present, i.e. bytes were
    actually dropped -- callers tag the parse/QC ledger with
    ``non_utf8_sanitized_copy`` in that case (no scratch copies needed).
    """
    with open(filepath, "r", errors="replace") as handle:
        text = handle.read()
    return text, "\ufffd" in text


def _first_nonempty(lines: List[str]) -> str:
    """First non-empty line, with only the trailing newline removed.

    Leading blanks are preserved: MERIT-II records are fixed-width and may
    blank-fill the satellite-id columns (e.g. ``' 105501'``), so stripping
    would shift every column by one.
    """
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.strip():
            return line
    return ""


def _slice(line: str, span: Tuple[int, int]) -> str:
    """1-based inclusive column slice of a fixed-width record."""
    start, end = span
    return line[start - 1:end]


def _looks_like_merit(line: str) -> bool:
    """Structural signature of a MERIT-II data record (no physics guessing)."""
    if len(line) < _MERIT_TOF[1]:
        return False
    satellite_id = _slice(line, _MERIT_SAT_ID).lstrip(" ")
    if not (satellite_id.isdigit() and len(satellite_id) >= 6):
        return False
    year = _slice(line, _MERIT_YEAR).strip()
    if not (year.isdigit() and 1 <= len(year) <= 2):
        return False
    doy = _slice(line, _MERIT_DOY).strip()
    if not (doy.isdigit() and 1 <= int(doy) <= 366):
        return False
    tod = _slice(line, _MERIT_TOD).strip()
    if not (tod.isdigit() and int(tod) < 86400 * 10_000_000):
        return False
    station = _slice(line, _MERIT_STATION).strip()
    if not (station.isdigit() and len(station) == 4):
        return False
    return _slice(line, _MERIT_TOF).strip().isdigit()


def detect_slr_format(first_lines: List[str]) -> str:
    """Classify an SLR file from its first lines by CONTENT.

    Args:
        first_lines: leading lines of the file (a handful is enough; blank
            lines are skipped).

    Returns:
        One of ``"crd"``, ``"legacy_npt"``, ``"merit_monthly"``,
        ``"html_error_page"`` (a saved provider error page, not SLR data),
        ``"unknown"``.
    """
    head = _first_nonempty(first_lines)
    if not head:
        return "unknown"
    head_left = head.lstrip()
    if head_left[:9].upper() == "<!DOCTYPE" or head_left[:5].upper() == "<HTML":
        return "html_error_page"
    if _CRD_HEADER_PREFIX.match(head_left):
        return "crd"
    if head_left == "99999":
        return "legacy_npt"
    if _looks_like_merit(head):
        return "merit_monthly"
    return "unknown"


def parse_merit_monthly_file(
    filepath: Union[str, Path],
    *,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
) -> Tuple[Dict, List[Dict]]:
    """Parse a MERIT-II / ILRS full-rate V2 ("frv2") merged SLR file.

    Format basis: the official ILRS specification archived at
    ``docs/standards/merit2-frv2-format.md`` (source:
    https://ilrs.gsfc.nasa.gov/data_and_products/formats/frv2_format.html,
    EUROLAS Data Center product page https://edc.dgfi.tum.de/en/oc/merit2/).
    The layout was additionally verified against the legacy raw archive
    (satellite id / DOY / station / range envelope cross-checks documented
    in the archived spec) and cross-validated against same-period CRD
    products by ``scripts/experiments/slr_crossformat_check.py``.

    Two-digit years are resolved by the mission window when provided
    (``window_start``/``window_end``), else anchored on the launch year
    encoded in the ILRS satellite id.  Records that cannot be placed inside
    the window (or structurally malformed lines) are REJECTED with a
    counted reason in the returned metadata -- never silently dropped and
    never force-fitted.

    Args:
        filepath: path to the MERIT-II file (fixed-width 130-char records).
        window_start: optional mission-window start (inclusive).
        window_end: optional mission-window end (inclusive).

    Returns:
        Tuple of (metadata dict, list of records).  Metadata keys include
        ``format``, ``target_id``, ``records_parsed``, ``records_rejected``
        and ``reject_reasons`` (reason -> count).
    """
    metadata: Dict = {
        "format": "merit_ii",
        "target_id": "",
        "target_name": "",
        "station_id": "",
        "records_parsed": 0,
        "records_rejected": 0,
        "reject_reasons": {},
        "window": (
            window_start.isoformat() if window_start else None,
            window_end.isoformat() if window_end else None,
        ),
        # True iff an explicit mission window gated the two-digit-year
        # resolution; False means the launch-year anchor (the wider
        # [launch, launch+40] span) was used instead.  Callers surface
        # this as the counted ``window_unavailable`` QC marker.
        "window_applied": window_start is not None or window_end is not None,
        # True iff undecodable bytes were replaced while reading the file
        # (see read_slr_text); parsed fields stay value-identical.
        "non_utf8_sanitized_copy": False,
    }
    records: List[Dict] = []
    stations: List[str] = []

    def _reject(reason: str) -> None:
        metadata["records_rejected"] += 1
        metadata["reject_reasons"][reason] = metadata["reject_reasons"].get(reason, 0) + 1

    text, non_utf8 = read_slr_text(filepath)
    metadata["non_utf8_sanitized_copy"] = non_utf8
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line:
            continue  # truly empty separator line
        # whitespace-only junk lines fall through and are counted as
        # malformed -- never silently skipped
        if len(line) < _MERIT_TOF[1]:
            _reject("malformed_record")
            continue

        satellite_id = _slice(line, _MERIT_SAT_ID).lstrip(" ")
        year_field = _slice(line, _MERIT_YEAR).strip()
        doy_field = _slice(line, _MERIT_DOY)
        tod_field = _slice(line, _MERIT_TOD).strip()
        station = _slice(line, _MERIT_STATION).strip()
        tof_field = _slice(line, _MERIT_TOF).strip()

        if not (satellite_id.isdigit() and year_field.isdigit()
                and doy_field.strip().isdigit() and tod_field.isdigit()
                and station.isdigit() and tof_field.isdigit()):
            _reject("malformed_record")
            continue

        satellite_id = satellite_id.zfill(7)
        if not metadata["target_id"]:
            metadata["target_id"] = satellite_id
        if station not in stations:
            stations.append(station)

        year_of_century = int(year_field)
        doy = int(doy_field)
        tod_units = int(tod_field)  # 0.1 microsecond granularity
        if not (1 <= doy <= 366) or tod_units >= 86400 * 10_000_000:
            _reject("malformed_record")
            continue

        time_of_flight_s = int(tof_field) * 1e-12  # ps -> s (two-way)
        epoch_candidates = [
            datetime(1900 + year_of_century, 1, 1) + timedelta(
                days=doy - 1, seconds=tod_units / 1e7),
            datetime(2000 + year_of_century, 1, 1) + timedelta(
                days=doy - 1, seconds=tod_units / 1e7),
        ]
        if window_start is not None or window_end is not None:
            in_window = [epoch for epoch in epoch_candidates
                         if (window_start is None or epoch >= window_start)
                         and (window_end is None or epoch <= window_end)]
            if len(in_window) != 1:
                _reject("epoch_outside_mission_window")
                continue
            epoch = in_window[0]
        else:
            # Anchor on the launch year encoded in the satellite id; the
            # plausibility test runs on the ROLLED epoch (doy 366 in a
            # non-leap year lands on Jan 1 of the next year -- a real
            # archive case: jason1.200501 line 1, a 2006-01-01 pass).
            launch_year2 = satellite_id[:2]
            if not launch_year2.isdigit():
                _reject("epoch_outside_mission_window")
                continue
            launch_year = (1900 + int(launch_year2)
                          if int(launch_year2) >= _SPUTNIK_BOUNDARY_YEAR2
                          else 2000 + int(launch_year2))
            plausible = [epoch for epoch in epoch_candidates
                         if launch_year <= epoch.year <= launch_year + 40]
            if len(plausible) != 1:
                _reject("epoch_outside_mission_window")
                continue
            epoch = plausible[0]

        sigma_field = _slice(line, _MERIT_SIGMA).strip()
        sigma_m = (int(sigma_field) * 1e-12 / 2) * SPEED_OF_LIGHT_M_PER_S if sigma_field.isdigit() else None
        returns_field = _slice(line, _MERIT_NUM_RETURNS).strip()
        num_returns = int(returns_field) if returns_field.isdigit() else None
        np_indicator = _slice(line, _MERIT_NP_INDICATOR).strip()
        window_length = _MERIT_NP_WINDOW_SECONDS.get(np_indicator) if np_indicator != "0" else None
        wavelength_field = _slice(line, _MERIT_WAVELENGTH).strip()

        records.append({
            "record_type": "normal_point" if np_indicator and np_indicator != "0" else "full_rate",
            "epoch": epoch,
            "time_of_flight_s": time_of_flight_s,
            "range_m": (time_of_flight_s / 2) * SPEED_OF_LIGHT_M_PER_S,
            "sigma_m": sigma_m,
            "num_returns": num_returns,
            "window_length": window_length,
            "station_id": station,
            "target_id": satellite_id,
            "target_name": "",
            "azimuth_deg": int(_slice(line, _MERIT_AZIMUTH).strip() or 0) * 1e-4
            if _slice(line, _MERIT_AZIMUTH).strip().isdigit() else None,
            "elevation_deg": int(_slice(line, _MERIT_ELEVATION).strip() or 0) * 1e-4
            if _slice(line, _MERIT_ELEVATION).strip().isdigit() else None,
            "wavelength_nm": int(wavelength_field) * 0.1 if wavelength_field.isdigit() else None,
        })

    metadata["records_parsed"] = len(records)
    metadata["stations"] = stations
    if stations:
        metadata["station_id"] = ",".join(stations)
    return metadata, records


__all__ = [
    "SPEED_OF_LIGHT_M_PER_S",
    "ILRS_TARGET_TO_SAT",
    "SAT_MISSION_WINDOWS",
    "UnsupportedSLRFormatError",
    "detect_slr_format",
    "mission_window_for_sat",
    "mission_window_for_target",
    "parse_merit_monthly_file",
    "read_slr_text",
]
