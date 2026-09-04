"""Normalize the Starlink operational subset and exact 107-hour overlap slice."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

from benchmarking.normalization import parse_utc_timestamp, parse_utc_timestamp_series, to_utc_iso


def _counted_to_numeric(series: object) -> tuple[object, int]:
    """``pd.to_numeric(errors="coerce")`` with the coercion failures COUNTED.

    REVIEW_FINDINGS 6.3.8 ("to_numeric failures counted"): a non-empty
    source string that fails numeric conversion must never vanish into a
    silent NaN -- the count rides on the returned frame's attrs and is
    surfaced by the readers.  Empty strings are legitimate missing values
    and are not counted.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric is None:
        return None, 0
    raw = series.astype(str).str.strip()
    failures = int((numeric.isna() & raw.notna() & (raw != "") & (raw != "None")).sum())
    return numeric, failures


DEFAULT_OUTPUT_DIR = Path("data/interim/normalized_sources/operational_constellation/starlink")
PRIMARY_EPHEMERIS_COLUMNS = [
    "sat_id",
    "satellite_name",
    "source",
    "epoch",
    "frame",
    "x_m",
    "y_m",
    "z_m",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "source_file",
    "processing_level",
]
AUXILIARY_EPHEMERIS_COLUMNS = [
    "sat_id",
    "satellite_name",
    "epoch",
    "auxiliary_field",
    "auxiliary_value",
    "unit",
    "role",
    "source_file",
]
TLE_COLUMNS = [
    "sat_id",
    "satellite_name",
    "source",
    "epoch",
    "mean_anomaly_rad",
    "raan_rad",
    "argument_of_perigee_rad",
    "eccentricity",
    "inclination_rad",
    "mean_motion_rad_per_min",
    "bstar",
    "source_file",
    "processing_level",
]
OUTPUT_FILES = {
    "ephemeris_state": "ephemeris_state.csv.gz",
    "ephemeris_auxiliary": "ephemeris_auxiliary.csv.gz",
    "tle_elements": "tle_elements.csv.gz",
    "source_inventory": "source_inventory.csv",
    "alignment_summary": "alignment_summary.csv",
    "quality_flags": "quality_flags.csv",
}


@dataclass(frozen=True)
class OperationalWindow:
    start: str
    end: str

    @property
    def start_ts(self) -> pd.Timestamp:
        return parse_utc_timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return parse_utc_timestamp(self.end)

    @property
    def duration_hours(self) -> float:
        return float((self.end_ts - self.start_ts).total_seconds() / 3600.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Starlink operational overlap subset")
    parser.add_argument(
        "--starlink-raw-root",
        default="data/raw/operational_constellation/starlink",
        help="Root containing canonical Starlink TLE archive and ephemeris data",
    )
    parser.add_argument(
        "--previous-data",
        dest="starlink_raw_root",
        help="Deprecated alias for --starlink-raw-root",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Ignored normalized output directory")
    parser.add_argument("--start", default="2024-11-26T06:00:00Z", help="Operational window start")
    parser.add_argument("--end", default="2024-11-30T17:00:00Z", help="Operational window end")
    parser.add_argument("--chunksize", type=int, default=250_000, help="Rows per ephemeris chunk")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional deterministic eligible satellite sample")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print output plan without writing")
    return parser.parse_args()


def _csv_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.csv") if path.is_file() and not path.name.startswith("._"))


def _starlink_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "Satellite_Name" in df.columns:
        mask = df["Satellite_Name"].fillna("").astype(str).str.upper().str.contains("STARLINK", regex=False)
        return df[mask].copy()
    if "Satellite_name" in df.columns:
        mask = df["Satellite_name"].fillna("").astype(str).str.upper().str.contains("STARLINK", regex=False)
        return df[mask].copy()
    return df.copy()


def _clip_by_epoch(df: pd.DataFrame, window: OperationalWindow) -> pd.DataFrame:
    if df.empty:
        return df
    epochs = parse_utc_timestamp_series(df["epoch"])
    mask = (epochs >= window.start_ts) & (epochs <= window.end_ts)
    clipped = df[mask].copy()
    clipped["epoch"] = _format_utc_series(epochs[mask])
    return clipped


def _format_utc_series(values: pd.Series) -> pd.Series:
    formatted = pd.Series(values.dt.strftime("%Y-%m-%dT%H:%M:%SZ"), index=values.index, dtype=object)
    has_microseconds = values.dt.microsecond != 0
    if has_microseconds.any():
        formatted.loc[has_microseconds] = values.loc[has_microseconds].map(to_utc_iso)
    return formatted


def _normalize_sat_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _read_tle(starlink_raw_root: Path, window: OperationalWindow) -> pd.DataFrame:
    root = starlink_raw_root / "tle_archive" / "overlaped_data"
    frames: List[pd.DataFrame] = []
    for path in _csv_files(root):
        df = pd.read_csv(path, dtype=str)
        if df.empty:
            continue
        df = _starlink_filter(df)
        if df.empty:
            continue
        epoch_field = "Epoch" if "Epoch" in df.columns else "Timestamp"
        numeric_failures = 0
        def _numeric(column: object) -> object:
            nonlocal numeric_failures
            values, failures = _counted_to_numeric(column)
            numeric_failures += failures
            return values
        normalized = pd.DataFrame(
            {
                "sat_id": _normalize_sat_id(df["Satellite_Number"]),
                "satellite_name": df.get("Satellite_Name", pd.Series([""] * len(df))).astype(str),
                "source": "tle",
                "epoch": _format_utc_series(parse_utc_timestamp_series(df[epoch_field])),
                "mean_anomaly_rad": _numeric(df.get("Mean_Anomaly")),
                "raan_rad": _numeric(df.get("Right_Ascension_of_Node")),
                "argument_of_perigee_rad": _numeric(df.get("Argument_of_Perigee")),
                "eccentricity": _numeric(df.get("Eccentricity")),
                "inclination_rad": _numeric(df.get("Inclination")),
                "mean_motion_rad_per_min": _numeric(df.get("Mean_Motion")),
                "bstar": _numeric(df.get("BSTAR_Drag_Term")),
                "source_file": path.name,
                "processing_level": "normalized_operational",
            }
        )
        normalized.attrs["to_numeric_failures"] = numeric_failures
        frames.append(_clip_by_epoch(normalized, window))
    if not frames:
        return pd.DataFrame(columns=TLE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    # REVIEW_FINDINGS 6.3.8 / B8: publication time NOT recorded in source;
    # "keep the newest prediction" is unimplementable.  Identical-epoch
    # duplicates removed keep='first' by filename order (files are
    # processed in sorted order above).
    combined = combined.drop_duplicates(subset=["sat_id", "epoch"]).sort_values(["sat_id", "epoch"]).reset_index(drop=True)
    total_failures = int(sum(frame.attrs.get("to_numeric_failures", 0) for frame in frames))
    if total_failures:
        print(f"Starlink TLE elements: {total_failures} to_numeric coercion failure(s) counted (kept as NaN)")
    combined.attrs["to_numeric_failures"] = total_failures
    return combined


def _normalize_ephemeris_chunk(
    chunk: pd.DataFrame,
    path: Path,
    window: OperationalWindow,
    selected_sat_ids: Set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk = _starlink_filter(chunk)
    if chunk.empty:
        return (
            pd.DataFrame(columns=PRIMARY_EPHEMERIS_COLUMNS),
            pd.DataFrame(columns=AUXILIARY_EPHEMERIS_COLUMNS),
        )
    sat_ids = _normalize_sat_id(chunk["Satellite_Number"])
    epochs = parse_utc_timestamp_series(chunk["Timestamp"])
    mask = (epochs >= window.start_ts) & (epochs <= window.end_ts)
    if selected_sat_ids is not None:
        mask &= sat_ids.isin(selected_sat_ids)
    chunk = chunk[mask].copy()
    if chunk.empty:
        return (
            pd.DataFrame(columns=PRIMARY_EPHEMERIS_COLUMNS),
            pd.DataFrame(columns=AUXILIARY_EPHEMERIS_COLUMNS),
        )
    sat_ids = sat_ids[mask]
    epochs = epochs[mask]
    satellite_names = chunk.get("Satellite_Name", pd.Series([""] * len(chunk))).astype(str)
    formatted_epochs = _format_utc_series(epochs)
    numeric_failures = 0
    def _numeric(column: object) -> object:
        nonlocal numeric_failures
        values, failures = _counted_to_numeric(column)
        numeric_failures += failures
        return values
    normalized = pd.DataFrame(
        {
            "sat_id": sat_ids,
            "satellite_name": satellite_names,
            "source": "ephemeris",
            "epoch": formatted_epochs,
            "frame": chunk.get("Coordinate", pd.Series([""] * len(chunk))).astype(str),
            "x_m": _numeric(chunk.get("X")) * 1000.0,
            "y_m": _numeric(chunk.get("Y")) * 1000.0,
            "z_m": _numeric(chunk.get("Z")) * 1000.0,
            "vx_mps": _numeric(chunk.get("Vx", chunk.get("VX"))) * 1000.0,
            "vy_mps": _numeric(chunk.get("Vy", chunk.get("VY"))) * 1000.0,
            "vz_mps": _numeric(chunk.get("Vz", chunk.get("VZ"))) * 1000.0,
            "source_file": path.name,
            "processing_level": "normalized_operational",
        }
    )
    normalized.attrs["to_numeric_failures"] = numeric_failures
    auxiliary_fields = [column for column in chunk.columns if column.startswith("High_Order_")]
    if auxiliary_fields:
        auxiliary_wide = pd.DataFrame(
            {
                "sat_id": sat_ids,
                "satellite_name": satellite_names,
                "epoch": formatted_epochs,
                "source_file": path.name,
            }
        )
        for field in auxiliary_fields:
            auxiliary_wide[field.lower()] = pd.to_numeric(chunk[field], errors="coerce")
        auxiliary = auxiliary_wide.melt(
            id_vars=["sat_id", "satellite_name", "epoch", "source_file"],
            value_vars=[field.lower() for field in auxiliary_fields],
            var_name="auxiliary_field",
            value_name="auxiliary_value",
        )
        auxiliary = auxiliary.dropna(subset=["auxiliary_value"]).reset_index(drop=True)
        auxiliary["unit"] = "source_provided_auxiliary"
        auxiliary["role"] = "auxiliary_preserved"
        auxiliary = auxiliary[AUXILIARY_EPHEMERIS_COLUMNS]
    else:
        auxiliary = pd.DataFrame(columns=AUXILIARY_EPHEMERIS_COLUMNS)
    return normalized, auxiliary


def _sat_id_from_ephemeris_filename(path: Path) -> str:
    parts = path.stem.split("_")
    for part in parts:
        if part.isdigit():
            return part
    return ""


def _read_ephemeris(
    starlink_raw_root: Path,
    window: OperationalWindow,
    chunksize: int,
    selected_sat_ids: Set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = starlink_raw_root / "ephemeris"
    frames: List[pd.DataFrame] = []
    auxiliary_frames: List[pd.DataFrame] = []
    for path in _csv_files(root):
        file_sat_id = _sat_id_from_ephemeris_filename(path)
        if selected_sat_ids is not None and file_sat_id and file_sat_id not in selected_sat_ids:
            continue
        try:
            for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str):
                normalized, auxiliary = _normalize_ephemeris_chunk(
                    chunk,
                    path,
                    window,
                    selected_sat_ids=selected_sat_ids,
                )
                if not normalized.empty:
                    frames.append(normalized)
                if not auxiliary.empty:
                    auxiliary_frames.append(auxiliary)
        except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
    if not frames:
        return (
            pd.DataFrame(columns=PRIMARY_EPHEMERIS_COLUMNS),
            pd.DataFrame(columns=AUXILIARY_EPHEMERIS_COLUMNS),
        )
    combined = pd.concat(frames, ignore_index=True)
    # REVIEW_FINDINGS 6.3.8 / B8: publication time NOT recorded in source;
    # "keep the newest prediction" is unimplementable.  Identical-epoch
    # duplicates removed keep='first' by filename order (files are
    # processed in sorted order above).
    combined = combined.drop_duplicates(subset=["sat_id", "epoch"]).sort_values(["sat_id", "epoch"]).reset_index(drop=True)
    total_failures = int(sum(frame.attrs.get("to_numeric_failures", 0) for frame in frames))
    if total_failures:
        print(f"Starlink ephemeris: {total_failures} to_numeric coercion failure(s) counted (kept as NaN)")
    combined.attrs["to_numeric_failures"] = total_failures
    if auxiliary_frames:
        auxiliary = pd.concat(auxiliary_frames, ignore_index=True)
        auxiliary = auxiliary.drop_duplicates(
            subset=["sat_id", "epoch", "auxiliary_field"],
        ).sort_values(["sat_id", "epoch", "auxiliary_field"]).reset_index(drop=True)
    else:
        auxiliary = pd.DataFrame(columns=AUXILIARY_EPHEMERIS_COLUMNS)
    return combined, auxiliary


def _read_last_data_line(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""
        while position > 0:
            read_size = min(4096, position)
            position -= read_size
            handle.seek(position)
            buffer = handle.read(read_size) + buffer
            lines = [line for line in buffer.splitlines() if line.strip()]
            if len(lines) >= 2:
                return lines[-1].decode("utf-8")
    return ""


def _read_ephemeris_file_bounds(path: Path) -> Dict[str, str] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            header_line = handle.readline().strip()
            first_line = handle.readline().strip()
        last_line = _read_last_data_line(path)
        if not header_line or not first_line or not last_line:
            return None
        header = next(csv.reader([header_line]))
        first = dict(zip(header, next(csv.reader([first_line]))))
        last = dict(zip(header, next(csv.reader([last_line]))))
    except (OSError, UnicodeDecodeError, StopIteration, csv.Error):
        return None

    sat_id = str(first.get("Satellite_Number", "")).strip()
    satellite_name = str(first.get("Satellite_Name", "")).strip()
    if "STARLINK" not in satellite_name.upper():
        return None
    first_epoch = parse_utc_timestamp(first.get("Timestamp"))
    last_epoch = parse_utc_timestamp(last.get("Timestamp"))
    start = min(first_epoch, last_epoch)
    end = max(first_epoch, last_epoch)
    return {
        "sat_id": sat_id,
        "satellite_name": satellite_name,
        "ephemeris_start_epoch": to_utc_iso(start),
        "ephemeris_end_epoch": to_utc_iso(end),
        "ephemeris_row_count": 0,
        "source_file": path.name,
    }


def _ephemeris_file_summary(starlink_raw_root: Path) -> pd.DataFrame:
    rows = []
    for path in _csv_files(starlink_raw_root / "ephemeris"):
        summary = _read_ephemeris_file_bounds(path)
        if summary is not None:
            rows.append(summary)
    return pd.DataFrame(rows)


def _build_alignment_from_ephemeris_summary(
    ephemeris_summary: pd.DataFrame,
    tle: pd.DataFrame,
    window: OperationalWindow,
) -> pd.DataFrame:
    tle_counts = tle.groupby("sat_id").size().to_dict() if not tle.empty else {}
    sat_ids = sorted(set(ephemeris_summary.get("sat_id", pd.Series(dtype=str)).astype(str)).union(tle.get("sat_id", pd.Series(dtype=str)).astype(str)))
    rows = []
    for sat_id in sat_ids:
        eph_sat = ephemeris_summary[ephemeris_summary["sat_id"].astype(str) == sat_id]
        start = eph_sat["ephemeris_start_epoch"].min() if not eph_sat.empty else ""
        end = eph_sat["ephemeris_end_epoch"].max() if not eph_sat.empty else ""
        tle_count = int(tle_counts.get(sat_id, 0))
        has_full_window = bool(start and end and parse_utc_timestamp(start) <= window.start_ts and parse_utc_timestamp(end) >= window.end_ts)
        rows.append(
            {
                "sat_id": sat_id,
                "ephemeris_start_epoch": to_utc_iso(window.start_ts) if has_full_window else start,
                "ephemeris_end_epoch": to_utc_iso(window.end_ts) if has_full_window else end,
                "ephemeris_row_count": int(eph_sat["ephemeris_row_count"].sum()) if not eph_sat.empty else 0,
                "tle_row_count": tle_count,
                "has_full_ephemeris_window": has_full_window,
                "has_tle_in_window": tle_count >= 1,
                "tle_rows_ge5": tle_count >= 5,
                "eligible": bool(has_full_window and tle_count >= 1),
            }
        )
    return pd.DataFrame(rows)


def _build_alignment(ephemeris: pd.DataFrame, tle: pd.DataFrame, window: OperationalWindow) -> pd.DataFrame:
    sat_ids = sorted(set(ephemeris.get("sat_id", pd.Series(dtype=str)).astype(str)).union(tle.get("sat_id", pd.Series(dtype=str)).astype(str)))
    tle_counts = tle.groupby("sat_id").size().to_dict() if not tle.empty else {}
    rows = []
    for sat_id in sat_ids:
        eph_sat = ephemeris[ephemeris["sat_id"].astype(str) == sat_id]
        tle_count = int(tle_counts.get(sat_id, 0))
        start = eph_sat["epoch"].min() if not eph_sat.empty else ""
        end = eph_sat["epoch"].max() if not eph_sat.empty else ""
        has_full_window = start == to_utc_iso(window.start_ts) and end == to_utc_iso(window.end_ts)
        eligible = bool(has_full_window and tle_count >= 1)
        rows.append(
            {
                "sat_id": sat_id,
                "ephemeris_start_epoch": start,
                "ephemeris_end_epoch": end,
                "ephemeris_row_count": int(len(eph_sat)),
                "tle_row_count": tle_count,
                "has_full_ephemeris_window": has_full_window,
                "has_tle_in_window": tle_count >= 1,
                "tle_rows_ge5": tle_count >= 5,
                "eligible": eligible,
            }
        )
    return pd.DataFrame(rows)


def _quality_flags(alignment: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for row in alignment.itertuples(index=False):
        if not row.has_full_ephemeris_window:
            rows.append({"sat_id": row.sat_id, "quality_flag": "ephemeris_window_incomplete"})
        if not row.has_tle_in_window:
            rows.append({"sat_id": row.sat_id, "quality_flag": "missing_tle_in_window"})
        rows.append({"sat_id": row.sat_id, "quality_flag": "tle_rows_ge5" if row.tle_rows_ge5 else "tle_rows_lt5"})
        if row.eligible:
            rows.append({"sat_id": row.sat_id, "quality_flag": "eligible_operational_subset"})
    return pd.DataFrame(rows, columns=["sat_id", "quality_flag"])


def _source_inventory(ephemeris: pd.DataFrame, tle: pd.DataFrame, window: OperationalWindow) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "ephemeris",
                "row_count": int(len(ephemeris)),
                "satellite_count": int(ephemeris["sat_id"].nunique()) if not ephemeris.empty else 0,
                "window_start": to_utc_iso(window.start_ts),
                "window_end": to_utc_iso(window.end_ts),
                "truth_status": "operational_not_flight_truth",
            },
            {
                "source": "tle",
                "row_count": int(len(tle)),
                "satellite_count": int(tle["sat_id"].nunique()) if not tle.empty else 0,
                "window_start": to_utc_iso(window.start_ts),
                "window_end": to_utc_iso(window.end_ts),
                "truth_status": "operational_not_flight_truth",
            },
        ]
    )


def _filter_eligible(
    ephemeris: pd.DataFrame,
    auxiliary: pd.DataFrame,
    tle: pd.DataFrame,
    alignment: pd.DataFrame,
    sample_size: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = stable_satellite_sample(alignment, sample_size=sample_size)
    return (
        ephemeris[ephemeris["sat_id"].astype(str).isin(selected)].reset_index(drop=True),
        auxiliary[auxiliary["sat_id"].astype(str).isin(selected)].reset_index(drop=True),
        tle[tle["sat_id"].astype(str).isin(selected)].reset_index(drop=True),
    )


def _mark_selection_and_counts(alignment: pd.DataFrame, selected_sat_ids: Set[str], ephemeris: pd.DataFrame) -> pd.DataFrame:
    alignment = alignment.copy()
    alignment["is_selected"] = alignment["sat_id"].astype(str).isin(selected_sat_ids)
    if not ephemeris.empty:
        counts = ephemeris.groupby(ephemeris["sat_id"].astype(str)).size().to_dict()
        alignment["ephemeris_row_count"] = alignment.apply(
            lambda row: int(counts.get(str(row["sat_id"]), row["ephemeris_row_count"])),
            axis=1,
        )
    return alignment


def stable_satellite_sample(alignment: pd.DataFrame, sample_size: int | None = None) -> List[str]:
    eligible = [str(value) for value in alignment.loc[alignment["eligible"], "sat_id"].tolist()]
    if sample_size is None or sample_size >= len(eligible):
        return sorted(eligible)
    ranked = sorted(eligible, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    return sorted(ranked[:sample_size])


def write_starlink_operational_subset(
    starlink_raw_root: Path | str,
    output_dir: Path | str,
    window: OperationalWindow,
    chunksize: int = 250_000,
    sample_size: int | None = None,
) -> List[Path]:
    starlink_raw_root = Path(starlink_raw_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tle = _read_tle(starlink_raw_root, window)
    if sample_size is not None:
        ephemeris_summary = _ephemeris_file_summary(starlink_raw_root)
        alignment = _build_alignment_from_ephemeris_summary(ephemeris_summary, tle, window)
        selected_sat_ids = set(stable_satellite_sample(alignment, sample_size=sample_size))
        ephemeris, auxiliary = _read_ephemeris(
            starlink_raw_root,
            window,
            chunksize=chunksize,
            selected_sat_ids=selected_sat_ids,
        )
    else:
        ephemeris, auxiliary = _read_ephemeris(starlink_raw_root, window, chunksize=chunksize)
        alignment = _build_alignment(ephemeris, tle, window)
        selected_sat_ids = set(stable_satellite_sample(alignment, sample_size=sample_size))
    alignment = _mark_selection_and_counts(alignment, selected_sat_ids, ephemeris)
    flags = _quality_flags(alignment)
    ephemeris_out, auxiliary_out, tle_out = _filter_eligible(ephemeris, auxiliary, tle, alignment, sample_size)
    inventory = _source_inventory(ephemeris_out, tle_out, window)

    outputs = {
        "ephemeris_state": output_dir / OUTPUT_FILES["ephemeris_state"],
        "ephemeris_auxiliary": output_dir / OUTPUT_FILES["ephemeris_auxiliary"],
        "tle_elements": output_dir / OUTPUT_FILES["tle_elements"],
        "source_inventory": output_dir / OUTPUT_FILES["source_inventory"],
        "alignment_summary": output_dir / OUTPUT_FILES["alignment_summary"],
        "quality_flags": output_dir / OUTPUT_FILES["quality_flags"],
    }
    ephemeris_out.to_csv(outputs["ephemeris_state"], index=False, compression="gzip")
    auxiliary_out.to_csv(outputs["ephemeris_auxiliary"], index=False, compression="gzip")
    tle_out.to_csv(outputs["tle_elements"], index=False, compression="gzip")
    inventory.to_csv(outputs["source_inventory"], index=False)
    alignment.to_csv(outputs["alignment_summary"], index=False)
    flags.to_csv(outputs["quality_flags"], index=False)
    return list(outputs.values())


def main() -> None:
    args = parse_args()
    window = OperationalWindow(args.start, args.end)
    if args.dry_run:
        print(f"Would normalize Starlink operational subset from {args.starlink_raw_root}")
        print(f"Window: {to_utc_iso(window.start_ts)} through {to_utc_iso(window.end_ts)} ({window.duration_hours:.1f}h)")
        print(f"Output: {args.output_dir}")
        return
    written = write_starlink_operational_subset(
        args.starlink_raw_root,
        args.output_dir,
        window,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
