"""Authoritative time-base utilities (leap seconds + SP3 time metadata).

Two rules from REVIEW_FINDINGS.md section 6.2.1 are implemented here and
must not be re-implemented anywhere else in this repository:

1. Leap seconds (TAI-UTC) come exclusively from the IERS/BIPM cumulative
   table as implemented by ERFA - the same library astropy's own TAI/UTC
   scale conversions use.  Hand-maintained leap tables are forbidden
   (fix C6: the retired hand-maintained table carried three wrong
   entries).
2. A product's time scale is read from the product's own metadata.  For
   SP3, the first ``%c`` header line exposes the 3-character position
   time system at columns 10-12 (``line[9:12]``); filename-prefix
   heuristics are forbidden (fix D11).

Offline determinism: astropy IERS auto-download is disabled at import so
scale conversions never touch the network.

Example invocation (from the repository root)::

    python3 -c "import sys; sys.path.insert(0, 'scripts/data_pipeline'); \
        from processors.timebase import sp3_time_scale_shift_seconds; \
        print(sp3_time_scale_shift_seconds('TAI', '2026-01-01T00:00:00'))"
    -37
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable

import erfa
from astropy.time import Time
from astropy.utils.iers import conf as iers_conf

# Offline determinism: ERFA leap data is bundled with astropy/erfa, but any
# astropy IERS table access (e.g. UT1 conversions) must not hit the network.
iers_conf.auto_download = False

# TAI - GPS = 19 s, fixed by the 1980-01-06 GPS epoch definition (no leap
# seconds in GPS time itself).
TAI_MINUS_GPS_SECONDS = 19.0

# SP3 producers that left the %c time-system field unfilled emit the spec
# placeholder; such a product declares nothing and must be rejected.
_SP3_TIME_SYSTEM_PLACEHOLDER_CHARS = set("c")


class UnsupportedTimeSystemError(ValueError):
    """SP3 time system is not one of the supported TAI/GPS/UTC values."""


def tai_minus_utc_seconds(epoch_utc: object) -> float:
    """TAI-UTC offset in seconds in effect at the given UTC epoch.

    Authoritative source: ``erfa.dat`` (IERS/BIPM cumulative leap-second
    table shipped inside ERFA, identical to what astropy uses internally
    for TAI<->UTC scale conversion).  Offsets are exact whole seconds for
    every epoch from 1972 onward (this dataset starts in 1992); before
    1972 the table carries the fractional TAI-UTC drift, returned as-is.

    ``epoch_utc`` accepts anything ``astropy.time.Time`` parses with
    ``scale="utc"`` (ISO strings, naive/aware datetimes, tz-aware pandas
    Timestamps).
    """
    moment = Time(epoch_utc, scale="utc")
    utc_date = moment.datetime
    return float(erfa.dat(utc_date.year, utc_date.month, utc_date.day, 0.0))


def sp3_time_system(header_lines: Iterable[str]) -> str:
    """Time system declared in the SP3 ``%c`` header line.

    Per the SP3-c/d specification the first ``%c`` line carries the
    3-character position time system at columns 10-12
    (``line[9:12]``; verified on the ssa/grg/ext series = ``TAI`` and
    the gsc series = ``GPS``).  ``header_lines`` is the raw header line
    list exposed as ``header_lines`` by
    ``processors.pod_processor.parse_sp3_header``.

    Raises ``ValueError`` when no ``%c`` line exists or when the field
    still holds the spec placeholder (``ccc``), i.e. the producer never
    declared a time system.
    """
    for line in header_lines:
        if line.startswith("%c"):
            field = line[9:12].strip()
            if not field or set(field) <= _SP3_TIME_SYSTEM_PLACEHOLDER_CHARS:
                raise ValueError(f"sp3_percent_c_time_system_undeclared:{line!r}")
            return field
    raise ValueError("sp3_header_missing_percent_c_time_system_line")


def sp3_time_scale_shift_seconds(time_system: str, epoch_utc: object) -> int:
    """Seconds to ADD to raw SP3 epochs so they read in UTC.

    TAI-tagged epochs: UTC = TAI - (TAI-UTC)  -> shift = -(TAI-UTC)
    GPS-tagged epochs: UTC = GPS - (TAI-UTC-19) -> shift = -(TAI-UTC-19)
    UTC-tagged epochs: shift = 0

    Any other time system (including an undeclared ``ccc`` placeholder)
    raises ``UnsupportedTimeSystemError`` (a ``ValueError`` subclass) so
    callers fail loudly instead of silently mistagging time.
    """
    tai_minus_utc = tai_minus_utc_seconds(epoch_utc)
    if time_system == "TAI":
        return int(-tai_minus_utc)
    if time_system == "GPS":
        return int(-(tai_minus_utc - TAI_MINUS_GPS_SECONDS))
    if time_system == "UTC":
        return 0
    raise UnsupportedTimeSystemError(f"sp3_time_system_not_supported:{time_system!r}")


def tai_datetime_to_utc(tai: _dt.datetime) -> _dt.datetime:
    """Convert a TAI-tagged naive datetime to UTC via the same ERFA table.

    Used when a product carries only a TAI time tag (e.g. EOF OSVs without
    a ``<UTC>`` element, REVIEW_FINDINGS 6.3.3): the tag must be converted
    with the authoritative IERS/BIPM leap-second table, never used raw.
    astropy's ``Time(scale="tai").utc`` uses exactly the ERFA ``dat`` table
    this module designates as the single authority, so the conversion and
    :func:`tai_minus_utc_seconds` can never disagree.
    """
    moment = Time(tai.replace(tzinfo=None), scale="tai", format="datetime")
    return moment.utc.datetime
