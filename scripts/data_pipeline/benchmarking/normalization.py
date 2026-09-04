"""Normalization utilities for POD, SLR, TLE, and ephemeris sources."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Iterable, List
import xml.etree.ElementTree as ET

import pandas as pd

from benchmarking.cache import cache_path, load_cache, save_cache
from processors.jason3_ogdr_processor import process_jason3_ogdr_file, validate_jason3_ogdr_file
from processors.pod_processor import process_pod_file
from processors.slr_formats import detect_slr_format
from processors.slr_processor import process_slr_file


POD_COLUMNS = ["sat_id", "epoch", "x", "y", "z", "vx", "vy", "vz", "sigma_x", "sigma_y", "sigma_z"]
SLR_COLUMNS = ["sat_id", "epoch", "range_m", "sigma_m", "residual_m", "station_id", "record_type"]
TLE_ALIASES = {
    "Epoch": "epoch",
    "Timestamp": "epoch",
    "Satellite_name": "sat_id",
    "Satellite_Name": "sat_id",
    "Mean_Motion": "mean_motion",
    "mean_motion_rad_per_min": "mean_motion",
    "Inclination": "inclination",
    "inclination_rad": "inclination",
    "Eccentricity": "eccentricity",
    "BSTAR_Drag_Term": "bstar",
}
EPHEMERIS_ALIASES = {
    "Timestamp": "epoch",
    "timestamp": "epoch",
    "Satellite_Name": "sat_id",
    "Satellite_name": "sat_id",
    "X": "x",
    "x_m": "x",
    "Y": "y",
    "y_m": "y",
    "Z": "z",
    "z_m": "z",
    "VX": "vx",
    "vx_mps": "vx",
    "VY": "vy",
    "vy_mps": "vy",
    "VZ": "vz",
    "vz_mps": "vz",
    "Vx": "vx",
    "Vy": "vy",
    "Vz": "vz",
}
POD_ALIASES = {
    **EPHEMERIS_ALIASES,
    "sigma_x_m": "sigma_x",
    "sigma_y_m": "sigma_y",
    "sigma_z_m": "sigma_z",
}

DE2RA = 0.0174532925199433
TWOPI = 6.283185307179586
XMNPDA = 1440.0
TEMP = TWOPI / (XMNPDA * XMNPDA)


def parse_utc_timestamp(value: object) -> pd.Timestamp:
    """Parse a timestamp as UTC without silently shifting naive source values."""
    if value in (None, "", pd.NaT):
        return pd.NaT
    if isinstance(value, str):
        value = value.strip()
        if value.endswith(" UTC"):
            value = value[:-4]
    return pd.to_datetime(value, errors="coerce", utc=True)


def parse_utc_timestamp_series(values: pd.Series) -> pd.Series:
    """Vectorized UTC timestamp parsing for large operational source tables."""
    cleaned = values.astype(str).str.strip().str.replace(r"\s+UTC$", "", regex=True)
    return pd.to_datetime(cleaned, errors="coerce", utc=True)


def to_utc_iso(value: object) -> str:
    """Format a timestamp as compact ISO-8601 UTC with a trailing Z."""
    timestamp = parse_utc_timestamp(value)
    if pd.isna(timestamp):
        return ""
    if timestamp.microsecond:
        return timestamp.isoformat().replace("+00:00", "Z")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_utc_naive(value: object) -> pd.Timestamp:
    """Parse a timestamp as UTC and return a timezone-free internal Timestamp."""
    timestamp = parse_utc_timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert(None)


def to_utc_naive_series(values: pd.Series) -> pd.Series:
    """Vectorized UTC parsing that returns timezone-free internal timestamps."""
    return parse_utc_timestamp_series(values).dt.tz_localize(None)


def to_utc_iso_series(values: pd.Series) -> pd.Series:
    """Vectorized UTC ISO formatting with compact trailing-Z strings."""
    raw = values.fillna("").astype(str).str.strip()
    result = pd.Series([""] * len(raw), index=raw.index, dtype=object)
    empty = raw.isin(["", "NaT", "nan", "None"])
    compact = raw.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
    simple = raw.str.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?: UTC)?$")
    simple_only = simple & ~compact
    result.loc[compact] = raw.loc[compact]
    if simple_only.any():
        formatted = raw.loc[simple_only].str.replace(r"\s+UTC$", "", regex=True).str.replace(" ", "T", regex=False)
        result.loc[simple_only] = formatted + "Z"
    remainder = ~(empty | compact | simple_only)
    if remainder.any():
        timestamps = parse_utc_timestamp_series(raw.loc[remainder])
        formatted = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ").astype(object)
        has_microseconds = timestamps.dt.microsecond.fillna(0).ne(0)
        if has_microseconds.any():
            formatted = formatted.copy()
            formatted.loc[has_microseconds] = timestamps.loc[has_microseconds].map(to_utc_iso)
        result.loc[remainder] = formatted.fillna("")
    return result


def _tle_checksum(line: str) -> int:
    total = 0
    for index in range(68):
        char = line[index]
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return total % 10


def _tle_sci(value: str) -> float:
    if value[1] == " ":
        return 0.0
    sign = "-" if value[0] == "-" else ""
    mantissa = "." + value[1:6]
    exponent = value[6:]
    return float(f"{sign}{mantissa}e{exponent}")


def _tle_epoch_to_datetime(year_fragment: int, day_of_year: float) -> dt.datetime:
    year = 2000 + year_fragment if year_fragment < 57 else 1900 + year_fragment
    return dt.datetime(year, 1, 1) + dt.timedelta(days=day_of_year - 1)


# Canonical TLE line parser output columns.  ``line1``/``line2`` are kept for
# SGP4-propagating callers (state_estimation); ``parse_raw_tle_file`` strips
# them so the historical six-column release schema is unchanged.
TLE_PAIR_COLUMNS = [
    "sat_id",
    "epoch",
    "mean_motion",
    "inclination",
    "eccentricity",
    "bstar",
    "line1",
    "line2",
]


def parse_tle_line_pairs(lines: Iterable[object], sat_id: str | None = None) -> pd.DataFrame:
    """Single canonical TLE parser for raw line-pair inputs (REVIEW_FINDINGS 6.3.1).

    This is the ONLY TLE line parser in the repository; the retired parallel
    implementations (``analyzers/state_estimation.py`` line-history loop and
    ``alignment/audit_reference_window_alignment.py`` epoch inference) now
    delegate here.  Spec basis: fixed-width TLE layout with mod-10 checksum
    on both lines (Vallado, Crawford, Hujsak & Kelso 2006, archived as
    ``docs/standards/vallado-2006-revisiting-spacetrack-report-3.pdf``).

    Validation per pair (every failure is COUNTED, never silently dropped):

    1. line1 starts with ``1`` / line2 starts with ``2`` (pairing);
    2. both lines are >= 69 characters;
    3. checksum matches on line1 and line2 independently;
    4. the epoch field (columns 19-32 of line1) parses;
    5. line1/line2 NORAD satellite numbers (columns 3-7) agree
       (REVIEW_FINDINGS 6.2.4 "TLE: line1/line2 NORAD 一致 + 两行 checksum").

    Three-line named TLEs (``0 <name>`` / free-text name line preceding a
    line1/line2 pair) are explicitly supported: the name line is skipped and
    counted as ``name_lines``; any other stray line is rejected under
    ``unrecognized_line``.

    Returns a DataFrame with columns ``TLE_PAIR_COLUMNS``; the parse ledger
    (``lines_input``, ``name_lines``, ``pairs_accepted``, ``pairs_rejected``,
    ``reject_reasons`` reason -> count) rides on
    ``df.attrs["tle_parse_stats"]`` (the DataFrame return type is part of
    the contract, so the stats travel as attrs rather than a second value).
    """
    clean = [str(line).strip() for line in lines if str(line).strip()]
    stats: dict = {
        "lines_input": len(clean),
        "name_lines": 0,
        "pairs_accepted": 0,
        "pairs_rejected": 0,
        "reject_reasons": {},
    }

    def _reject(reason: str) -> None:
        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1

    records: list[dict] = []
    index = 0
    while index < len(clean):
        line1 = clean[index]
        if (
            line1.startswith("1")
            and index + 1 < len(clean)
            and clean[index + 1].startswith("2")
        ):
            line2 = clean[index + 1]
            index += 2
            rejected = False
            if len(line1) < 69 or len(line2) < 69:
                _reject("short_line")
                rejected = True
            if not rejected and (
                _tle_checksum(line1) != int(line1[68])
                or _tle_checksum(line2) != int(line2[68])
            ):
                if _tle_checksum(line1) != int(line1[68]):
                    _reject("checksum_line1")
                if _tle_checksum(line2) != int(line2[68]):
                    _reject("checksum_line2")
                rejected = True
            norad1 = line1[2:7].strip()
            if not rejected and norad1 != line2[2:7].strip():
                _reject("norad_mismatch")
                rejected = True
            if not rejected:
                try:
                    epoch = _tle_epoch_to_datetime(int(line1[18:20]), float(line1[20:32]))
                except ValueError:
                    _reject("bad_epoch")
                    rejected = True
            if rejected:
                stats["pairs_rejected"] += 1
                continue
            records.append(
                {
                    "sat_id": sat_id or norad1,
                    "epoch": epoch,
                    "mean_motion": float(line2[52:63].strip()) * TEMP * XMNPDA,
                    "inclination": float(line2[8:16]) * DE2RA,
                    "eccentricity": float("0." + line2[26:33].strip()),
                    "bstar": _tle_sci(line1[53:61]) if line1[53:61].strip() else 0.0,
                    "line1": line1,
                    "line2": line2,
                }
            )
            stats["pairs_accepted"] += 1
        elif (
            not line1.startswith(("1", "2"))
            and index + 1 < len(clean)
            and clean[index + 1].startswith("1")
        ):
            # 3LE name line ("0 JASON-3" or free text): supported, skipped.
            stats["name_lines"] += 1
            index += 1
        else:
            _reject("unrecognized_line")
            index += 1

    frame = pd.DataFrame(records, columns=TLE_PAIR_COLUMNS)
    frame.attrs["tle_parse_stats"] = stats
    return frame


def parse_raw_tle_file(path: Path, sat_id: str | None = None) -> pd.DataFrame:
    """Parse a raw line-pair TLE file into a normalized DataFrame.

    Delegates to the canonical :func:`parse_tle_line_pairs` and strips the
    ``line1``/``line2`` columns so the historical six-column output schema
    (used by the release export) is unchanged; the parse stats remain
    available on ``df.attrs``.
    """
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    frame = parse_tle_line_pairs(lines, sat_id=sat_id)
    if frame.empty:
        return pd.DataFrame()
    return frame.loc[:, TLE_PAIR_COLUMNS[:6]]


def _finalize_epochs(df: pd.DataFrame, column: str = "epoch") -> pd.DataFrame:
    if column in df.columns:
        df[column] = to_utc_naive_series(df[column])
    return df


def _is_parseable_pod_file(path: Path) -> bool:
    """Cheap validation to skip raw POD files that are clearly broken."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".eof":
            ET.parse(path)
            return True
        if suffix in {".nc", ".nc4"}:
            return validate_jason3_ogdr_file(str(path)) is None
    except Exception:
        return False
    return True


def standardize_pod_dataframe(df: pd.DataFrame, sat_id: str | None = None) -> pd.DataFrame:
    """Normalize POD data to the benchmark schema."""
    df = df.copy().rename(columns=POD_ALIASES)
    df = _finalize_epochs(df, "epoch")
    if sat_id is not None:
        df["sat_id"] = sat_id
    for column in POD_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[POD_COLUMNS].sort_values("epoch").reset_index(drop=True)


def standardize_slr_dataframe(df: pd.DataFrame, sat_id: str | None = None) -> pd.DataFrame:
    """Normalize SLR data to the benchmark schema."""
    df = df.copy().rename(columns={"Epoch": "epoch", "Timestamp": "epoch", "Satellite_Name": "sat_id", "Satellite_name": "sat_id"})
    df = _finalize_epochs(df, "epoch")
    if sat_id is not None:
        df["sat_id"] = sat_id
    elif "sat_id" not in df.columns:
        if "target_name" in df.columns:
            df["sat_id"] = df["target_name"].astype(str).str.lower()
        elif "target_id" in df.columns:
            df["sat_id"] = df["target_id"].astype(str)
        else:
            df["sat_id"] = pd.NA
    if "residual_m" not in df.columns:
        df["residual_m"] = pd.NA
    for column in SLR_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[SLR_COLUMNS].sort_values("epoch").reset_index(drop=True)


def standardize_tle_dataframe(df: pd.DataFrame, sat_id: str | None = None) -> pd.DataFrame:
    """Normalize TLE tables with mixed column conventions."""
    df = df.copy().rename(columns=TLE_ALIASES)
    df = _finalize_epochs(df, "epoch")
    if sat_id is not None:
        df["sat_id"] = sat_id
    required = ["sat_id", "epoch", "mean_motion", "inclination", "eccentricity", "bstar"]
    for column in required:
        if column not in df.columns:
            df[column] = pd.NA
    return df[required].sort_values("epoch").reset_index(drop=True)


def standardize_ephemeris_dataframe(df: pd.DataFrame, sat_id: str | None = None) -> pd.DataFrame:
    """Normalize processed ephemeris CSV data."""
    df = df.copy().rename(columns=EPHEMERIS_ALIASES)
    df = _finalize_epochs(df, "epoch")
    if sat_id is not None:
        df["sat_id"] = sat_id
    required = ["sat_id", "epoch", "x", "y", "z", "vx", "vy", "vz"]
    for column in required:
        if column not in df.columns:
            df[column] = pd.NA
    return df[required].sort_values("epoch").reset_index(drop=True)


def _combine_frames(frames: List[pd.DataFrame], columns: List[str], dedup_subset: List[str] | None = None) -> pd.DataFrame:
    """Concatenate normalized frames and drop redundant rows when safe."""
    if not frames:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(frames, ignore_index=True)
    if dedup_subset:
        combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
    else:
        combined = combined.drop_duplicates(keep="last")
    return combined.sort_values("epoch").reset_index(drop=True)


def _pod_nc_stride_rows() -> int:
    """Return the row stride applied to OGDR netCDF inputs (1 Hz -> N s).

    ``POD_NC_STRIDE_SECONDS`` opts into decimation for the Jason-3 OGDR
    holdings (13k+ files at ~1 Hz), which otherwise exceed memory when the
    full normalized frame is assembled. 0 or unset keeps every row.
    """
    raw_value = os.environ.get("POD_NC_STRIDE_SECONDS", "0")
    try:
        return max(0, int(float(raw_value)))
    except ValueError:
        return 0


def _apply_nc_stride(df: pd.DataFrame, stride_rows: int) -> pd.DataFrame:
    if stride_rows <= 1 or len(df) == 0:
        return df
    return df.iloc[::stride_rows].reset_index(drop=True)


def load_pod_inputs(paths: Iterable[Path], sat_id: str | None = None) -> pd.DataFrame:
    """Load and normalize POD inputs from raw files or processed CSVs."""
    nc_stride_rows = _pod_nc_stride_rows()
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.is_dir():
            candidates = (
                sorted(path.glob("*.EOF")) +
                sorted(path.glob("*.eof")) +
                sorted(path.glob("*.sp3")) +
                sorted(path.glob("*.SP3")) +
                sorted(path.glob("*.nc")) +
                sorted(path.glob("*.nc4"))
            )
            for candidate in candidates:
                try:
                    candidate_cache = cache_path("pod", candidate, sat_id=sat_id)
                    cached = load_cache(candidate_cache, candidate)
                    if cached is not None:
                        frames.append(_apply_nc_stride(cached, nc_stride_rows) if candidate.suffix.lower() in {".nc", ".nc4"} else cached)
                        continue
                    if candidate.suffix.lower() in {".eof", ".nc", ".nc4"} and not _is_parseable_pod_file(candidate):
                        # Counted skip (never silent): the validation error
                        # names the product, e.g. the clear
                        # UnsupportedProductError for Jason-1/2 GDR granules
                        # (REVIEW_FINDINGS 6.3.4).
                        reason = (
                            validate_jason3_ogdr_file(str(candidate))
                            if candidate.suffix.lower() in {".nc", ".nc4"}
                            else "xml_unparseable"
                        )
                        print(f"load_pod_inputs: skipping unparseable POD file ({reason}): {candidate.name}")
                        continue
                    if candidate.suffix.lower() in {".nc", ".nc4"}:
                        df = process_jason3_ogdr_file(str(candidate), sat_id=sat_id or "jason-3")
                    else:
                        df = process_pod_file(str(candidate))
                    if len(df) > 0:
                        normalized = standardize_pod_dataframe(df, sat_id=sat_id)
                        save_cache(normalized, candidate_cache)
                        if candidate.suffix.lower() in {".nc", ".nc4"}:
                            normalized = _apply_nc_stride(normalized, nc_stride_rows)
                        frames.append(normalized)
                except Exception as exc:
                    print(f"Error processing {candidate}: {exc}")
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            if len(df) > 0:
                frames.append(standardize_pod_dataframe(df, sat_id=sat_id))
        elif path.suffix.lower() in {".nc", ".nc4"}:
            try:
                df = process_jason3_ogdr_file(str(path), sat_id=sat_id or "jason-3")
                if len(df) > 0:
                    frames.append(_apply_nc_stride(standardize_pod_dataframe(df, sat_id=sat_id), nc_stride_rows))
            except Exception as exc:
                print(f"Error processing {path}: {exc}")
        else:
            try:
                df = process_pod_file(str(path))
                if len(df) > 0:
                    frames.append(standardize_pod_dataframe(df, sat_id=sat_id))
            except Exception as exc:
                print(f"Error processing {path}: {exc}")
    return _combine_frames(frames, POD_COLUMNS, dedup_subset=["sat_id", "epoch"])


def _detect_child_slr_format(path: Path) -> str:
    """Classify a directory child by CONTENT, never by name or extension.

    Returns ``detect_slr_format``'s verdict, or ``"unknown"`` for anything
    that cannot be read as text (subdirectories, binaries, empty files).
    """
    if not path.is_file():
        return "unknown"
    try:
        with open(path, "r", errors="replace") as handle:
            first_lines = [handle.readline() for _ in range(10)]
    except OSError:
        return "unknown"
    return detect_slr_format(first_lines)


def load_slr_inputs(paths: Iterable[Path], sat_id: str | None = None) -> pd.DataFrame:
    """Load and normalize SLR inputs from raw files or processed CSVs.

    Files inside directories are discovered by CONTENT (root-cause fix A4):
    the legacy raw archive stores MERIT-II merged monthly files with no
    extension at all and CRD content under ``.frd``/``.npt`` names, so the
    old extension whitelist (``.npt``/``.np``/``.np2``/``.crd``) left
    ~1,200 date-suffixed files plus 21 ``.frd`` files invisible to the
    pipeline.  Every child is now classified with
    ``processors.slr_formats.detect_slr_format``; a child whose content
    matches no supported SLR format is reported by name and skipped --
    it is not SLR data, and it is never force-parsed by a guess parser.
    A directly-passed non-CSV file with unknown content still raises
    ``UnsupportedSLRFormatError`` (file-level failure propagates).
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.glob("*")):
                _fmt = _detect_child_slr_format(child)
                if _fmt in ("unknown", "html_error_page"):
                    print(f"load_slr_inputs: skipping non-SLR file ({_fmt}): {child.name}")
                    continue
                child_cache = cache_path("slr", child, sat_id=sat_id)
                cached = load_cache(child_cache, child)
                if cached is not None:
                    frames.append(cached)
                    continue
                normalized = standardize_slr_dataframe(process_slr_file(str(child)), sat_id=sat_id)
                save_cache(normalized, child_cache)
                frames.append(normalized)
        elif path.suffix.lower() == ".csv":
            frames.append(standardize_slr_dataframe(pd.read_csv(path), sat_id=sat_id))
        else:
            frames.append(standardize_slr_dataframe(process_slr_file(str(path)), sat_id=sat_id))
    return _combine_frames(frames, SLR_COLUMNS)


def load_csv_inputs(paths: Iterable[Path], source_type: str, sat_id: str | None = None) -> pd.DataFrame:
    """Load and normalize TLE or ephemeris CSV inputs."""
    frames: List[pd.DataFrame] = []
    normalizer = standardize_tle_dataframe if source_type == "tle" else standardize_ephemeris_dataframe
    for path in paths:
        if path.is_dir():
            if source_type == "tle":
                files = (
                    sorted(path.glob("*.csv"))
                    + sorted(path.glob("*.csv.gz"))
                    + sorted(path.glob("*.tle"))
                    + sorted(path.glob("*.txt"))
                )
            else:
                files = sorted(path.glob("*.csv")) + sorted(path.glob("*.csv.gz"))
        else:
            files = [path]
        for file_path in files:
            if source_type == "tle" and file_path.suffix.lower() in {".tle", ".txt"}:
                file_cache = cache_path("tle", file_path, sat_id=sat_id)
                cached = load_cache(file_cache, file_path)
                if cached is not None:
                    frames.append(cached)
                    continue
                normalized = normalizer(parse_raw_tle_file(file_path, sat_id=sat_id), sat_id=sat_id)
                save_cache(normalized, file_cache)
                frames.append(normalized)
            else:
                frames.append(normalizer(pd.read_csv(file_path), sat_id=sat_id))
    if source_type == "tle":
        columns = ["sat_id", "epoch", "mean_motion", "inclination", "eccentricity", "bstar"]
        dedup_subset = ["sat_id", "epoch"]
    else:
        columns = ["sat_id", "epoch", "x", "y", "z", "vx", "vy", "vz"]
        dedup_subset = ["sat_id", "epoch"]
    return _combine_frames(frames, columns, dedup_subset=dedup_subset)
