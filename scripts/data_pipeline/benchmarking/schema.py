"""Schema helpers for benchmark events and evidence tables."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List

import pandas as pd

from benchmarking.normalization import to_utc_iso, to_utc_iso_series, to_utc_naive


EVENT_LABELS = {"event", "no_event", "ignore"}
TRACKS = {"validated", "operational", "open_world"}
EVENT_FAMILIES = {
    "orbit_raise",
    "orbit_lower",
    "station_keeping",
    "plane_change",
    "unknown_maneuver",
    "none",
}
ORBITAL_EFFECTS = {
    "semi_major_axis",
    "inclination",
    "raan",
    "mean_longitude",
    "multi_axis",
    "none",
}

LABEL_TABLE_COLUMNS = [
    "event_id",
    "sat_id",
    "track",
    "event_label",
    "t_start_lo",
    "t_start_hi",
    "t_end_lo",
    "t_end_hi",
    "event_family",
    "orbital_effect",
    "evidence_sources",
    "candidate_id",
    "evidence_ids",
    "quality_flags",
    "ignore_reason",
    "gate_status_json",
    "review_status",
    "notes",
]

CANDIDATE_TABLE_COLUMNS = [
    "candidate_id",
    "sat_id",
    "t_start_lo",
    "t_start_hi",
    "t_end_lo",
    "t_end_hi",
    "evidence_sources",
    "evidence_count",
    "score_sum",
    "score_max",
]

EVIDENCE_TABLE_COLUMNS = [
    "evidence_id",
    "candidate_id",
    "sat_id",
    "source",
    "start_time",
    "end_time",
    "score",
    "supports_event",
    "signal_class",
    "metrics_json",
    "details",
]

GATE_ATTRITION_TABLE_COLUMNS = [
    "sat_id",
    "track",
    "pod_candidate_count",
    "tle_consistent_count",
    "tle_conflict_count",
    "slr_supportive_count",
    "slr_insufficient_count",
    "quality_pass_count",
    "event_count",
    "no_event_count",
    "ignore_count",
]

MANEUVER_ANNOTATION_TABLE_COLUMNS = [
    "annotation_id",
    "sat_id",
    "source",
    "source_provider",
    "source_url",
    "reference",
    "event_type",
    "event_time_utc",
    "event_time_role",
    "reported_operation_start_utc",
    "reported_operation_end_utc",
    "window_start_utc",
    "window_end_utc",
    "time_uncertainty_seconds",
    "impulse_count",
    "mission_record_code",
    "annotation_status",
    "truth_status",
    "raw_record",
    "notes",
]

MANEUVER_EVENT_WINDOW_TABLE_COLUMNS = [
    "annotation_id",
    "sat_id",
    "window_start_utc",
    "window_end_utc",
    "event_type",
    "source_url",
    "event_label",
    "confidence_tier",
]

SOURCE_INVENTORY_TABLE_COLUMNS = [
    "sat_id",
    "track",
    "source",
    "row_count",
    "columns",
    "time_start",
    "time_end",
    "sat_ids",
]

SOURCE_SNAPSHOT_TABLE_SCHEMAS: Dict[str, List[str]] = {
    "source_snapshot_pod": [
        "sat_id",
        "epoch",
        "x_m",
        "y_m",
        "z_m",
        "vx_mps",
        "vy_mps",
        "vz_mps",
        "sigma_x_m",
        "sigma_y_m",
        "sigma_z_m",
    ],
    "source_snapshot_slr": [
        "sat_id",
        "epoch",
        "range_m",
        "sigma_m",
        "residual_m",
        "station_id",
        "record_type",
    ],
    "source_snapshot_tle": [
        "sat_id",
        "epoch",
        "mean_motion_rad_per_min",
        "inclination_rad",
        "eccentricity",
        "bstar",
    ],
    "source_snapshot_ephemeris": [
        "sat_id",
        "epoch",
        "x_m",
        "y_m",
        "z_m",
        "vx_mps",
        "vy_mps",
        "vz_mps",
    ],
}

OPERATIONAL_TABLE_SCHEMAS: Dict[str, List[str]] = {
    "operational_ephemeris_state": [
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
    ],
    "operational_ephemeris_auxiliary": [
        "sat_id",
        "satellite_name",
        "epoch",
        "auxiliary_field",
        "auxiliary_value",
        "unit",
        "role",
        "source_file",
    ],
    "operational_tle_elements": [
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
    ],
}

RELEASE_TABLE_SCHEMAS: Dict[str, List[str]] = {
    "events": LABEL_TABLE_COLUMNS,
    "stable_windows": LABEL_TABLE_COLUMNS,
    "ignore_windows": LABEL_TABLE_COLUMNS,
    "review_queue": LABEL_TABLE_COLUMNS,
    "label_audit": LABEL_TABLE_COLUMNS,
    "candidates": CANDIDATE_TABLE_COLUMNS,
    "evidence": EVIDENCE_TABLE_COLUMNS,
    "gate_attrition": GATE_ATTRITION_TABLE_COLUMNS,
    "maneuver_annotations": MANEUVER_ANNOTATION_TABLE_COLUMNS,
    "maneuver_event_windows": MANEUVER_EVENT_WINDOW_TABLE_COLUMNS,
    "source_inventory": SOURCE_INVENTORY_TABLE_COLUMNS,
}

DATA_DICTIONARY_TABLE_SCHEMAS: Dict[str, List[str]] = {
    **RELEASE_TABLE_SCHEMAS,
    **SOURCE_SNAPSHOT_TABLE_SCHEMAS,
    **OPERATIONAL_TABLE_SCHEMAS,
}

RELEASE_TIME_COLUMNS: Dict[str, List[str]] = {
    "events": ["t_start_lo", "t_start_hi", "t_end_lo", "t_end_hi"],
    "stable_windows": ["t_start_lo", "t_start_hi", "t_end_lo", "t_end_hi"],
    "ignore_windows": ["t_start_lo", "t_start_hi", "t_end_lo", "t_end_hi"],
    "review_queue": ["t_start_lo", "t_start_hi", "t_end_lo", "t_end_hi"],
    "label_audit": ["t_start_lo", "t_start_hi", "t_end_lo", "t_end_hi"],
    "candidates": ["t_start_lo", "t_start_hi", "t_end_lo", "t_end_hi"],
    "evidence": ["start_time", "end_time"],
    "maneuver_annotations": [
        "event_time_utc",
        "reported_operation_start_utc",
        "reported_operation_end_utc",
        "window_start_utc",
        "window_end_utc",
    ],
    "maneuver_event_windows": ["window_start_utc", "window_end_utc"],
    "source_inventory": ["time_start", "time_end"],
}

SOURCE_SNAPSHOT_RENAMES: Dict[str, Dict[str, str]] = {
    "pod": {
        "x": "x_m",
        "y": "y_m",
        "z": "z_m",
        "vx": "vx_mps",
        "vy": "vy_mps",
        "vz": "vz_mps",
        "sigma_x": "sigma_x_m",
        "sigma_y": "sigma_y_m",
        "sigma_z": "sigma_z_m",
    },
    "ephemeris": {
        "x": "x_m",
        "y": "y_m",
        "z": "z_m",
        "vx": "vx_mps",
        "vy": "vy_mps",
        "vz": "vz_mps",
    },
    "tle": {
        "mean_motion": "mean_motion_rad_per_min",
        "inclination": "inclination_rad",
    },
    "slr": {},
}

FIELD_DESCRIPTIONS = {
    "event_id": "Stable public window identifier.",
    "sat_id": "Repository satellite identifier.",
    "track": "Release track, such as validated, operational, or open_world.",
    "event_label": "Strict public label restricted to event, no_event, or ignore.",
    "t_start_lo": "Lower bound on the window start time, serialized as ISO-8601 UTC.",
    "t_start_hi": "Upper bound on the window start time, serialized as ISO-8601 UTC.",
    "t_end_lo": "Lower bound on the window end time, serialized as ISO-8601 UTC.",
    "t_end_hi": "Upper bound on the window end time, serialized as ISO-8601 UTC.",
    "event_family": "Coarse event family for event rows, or none.",
    "orbital_effect": "Primary orbital effect for event rows, or none.",
    "evidence_sources": "Comma-separated sources supporting or informing the row.",
    "candidate_id": "Internal candidate identifier, blank for mined no-event rows.",
    "evidence_ids": "Comma-separated evidence row identifiers.",
    "quality_flags": "Comma-separated machine-readable quality flags.",
    "ignore_reason": "Machine-readable reason for ignore rows.",
    "gate_status_json": "JSON object with source and quality gate statuses.",
    "review_status": "Review state assigned by the label pipeline.",
    "notes": "Short audit note generated by the pipeline.",
    "annotation_id": "Stable mission-reported maneuver annotation identifier.",
    "source_provider": "Organization or service providing the source record.",
    "source_url": "Exact URL for the third-party source record or file.",
    "reference": "Citation or source reference string.",
    "event_type": "Mission-reported event type.",
    "event_time_utc": "Representative event timestamp, serialized as ISO-8601 UTC.",
    "event_time_role": "Role of the representative timestamp, such as first_impulse_time.",
    "reported_operation_start_utc": "Mission-reported operation start timestamp.",
    "reported_operation_end_utc": "Mission-reported operation end timestamp.",
    "window_start_utc": "Margin-expanded training or evaluation window start timestamp.",
    "window_end_utc": "Margin-expanded training or evaluation window end timestamp.",
    "time_uncertainty_seconds": "Total uncertainty or expansion span assigned to the event window.",
    "impulse_count": "Mission-reported number of impulses in the maneuver record.",
    "mission_record_code": "Mission or file identifier code from the source record (not a maneuver-type taxonomy).",
    "annotation_status": "Annotation provenance status.",
    "truth_status": "Claim boundary for the annotation truth source.",
    "raw_record": "Raw source line preserved for audit.",
    "confidence_tier": "Mission annotation confidence tier.",
    "evidence_count": "Number of evidence rows fused into a candidate.",
    "score_sum": "Sum of positive evidence scores for a candidate.",
    "score_max": "Maximum evidence score for a candidate.",
    "evidence_id": "Stable source evidence identifier.",
    "source": "Evidence or inventory source name.",
    "start_time": "Evidence interval start time, serialized as ISO-8601 UTC.",
    "end_time": "Evidence interval end time, serialized as ISO-8601 UTC.",
    "score": "Evidence confidence or support score.",
    "supports_event": "Boolean indicating whether the evidence supports an event.",
    "signal_class": "Evidence signal class assigned by the extractor.",
    "metrics_json": "JSON object with source-specific evidence metrics.",
    "details": "Short source-specific evidence detail code.",
    "pod_candidate_count": "Rows entering the POD candidate gate.",
    "tle_consistent_count": "Rows with TLE gate pass status.",
    "tle_conflict_count": "Rows with TLE conflict status.",
    "slr_supportive_count": "Rows with SLR gate pass status.",
    "slr_insufficient_count": "Rows with insufficient SLR coverage status.",
    "quality_pass_count": "Rows promoted to event or no_event.",
    "event_count": "Strict event row count.",
    "no_event_count": "Strict no_event row count.",
    "ignore_count": "Ignored row count.",
    "row_count": "Number of normalized source rows.",
    "columns": "Serialized normalized source columns.",
    "time_start": "First normalized source epoch, serialized as ISO-8601 UTC.",
    "time_end": "Last normalized source epoch, serialized as ISO-8601 UTC.",
    "sat_ids": "Serialized satellite identifiers present in the source table.",
    "epoch": "Source sample timestamp, serialized as ISO-8601 UTC.",
    "x_m": "Cartesian x position in meters.",
    "y_m": "Cartesian y position in meters.",
    "z_m": "Cartesian z position in meters.",
    "vx_mps": "Cartesian x velocity in meters per second.",
    "vy_mps": "Cartesian y velocity in meters per second.",
    "vz_mps": "Cartesian z velocity in meters per second.",
    "sigma_x_m": "Source-provided x-position uncertainty in meters.",
    "sigma_y_m": "Source-provided y-position uncertainty in meters.",
    "sigma_z_m": "Source-provided z-position uncertainty in meters.",
    "range_m": "SLR range observation in meters.",
    "sigma_m": "SLR source-provided uncertainty in meters.",
    "residual_m": "SLR residual in meters.",
    "station_id": "SLR station identifier.",
    "record_type": "SLR source record type.",
    "mean_motion_rad_per_min": "TLE mean motion in radians per minute.",
    "inclination_rad": "TLE inclination in radians.",
    "eccentricity": "TLE eccentricity.",
    "bstar": "TLE BSTAR drag term.",
    "satellite_name": "Provider satellite display name.",
    "frame": "Source-provided coordinate frame label.",
    "source_file": "Input file name used to create the normalized row.",
    "processing_level": "Normalization or processing level assigned by the pipeline.",
    "auxiliary_field": "Name of a source-specific auxiliary field preserved outside the primary state schema.",
    "auxiliary_value": "Source-provided high-order or auxiliary ephemeris value preserved for audit.",
    "unit": "Unit label for an auxiliary or metadata value.",
    "role": "Release role assigned to an auxiliary or metadata value.",
    "mean_anomaly_rad": "TLE mean anomaly in radians.",
    "raan_rad": "TLE right ascension of ascending node in radians.",
    "argument_of_perigee_rad": "TLE argument of perigee in radians.",
}

FIELD_UNITS = {
    "x_m": "m",
    "y_m": "m",
    "z_m": "m",
    "vx_mps": "m/s",
    "vy_mps": "m/s",
    "vz_mps": "m/s",
    "sigma_x_m": "m",
    "sigma_y_m": "m",
    "sigma_z_m": "m",
    "range_m": "m",
    "sigma_m": "m",
    "residual_m": "m",
    "mean_motion_rad_per_min": "rad/min",
    "inclination_rad": "rad",
    "mean_anomaly_rad": "rad",
    "raan_rad": "rad",
    "argument_of_perigee_rad": "rad",
}

FIELD_DATA_TYPES = {
    "event_id": "string",
    "sat_id": "string",
    "track": "string",
    "event_label": "string",
    "event_family": "string",
    "orbital_effect": "string",
    "evidence_sources": "string",
    "candidate_id": "string",
    "evidence_ids": "string",
    "quality_flags": "string",
    "ignore_reason": "string",
    "gate_status_json": "json",
    "review_status": "string",
    "notes": "string",
    "source": "string",
    "signal_class": "string",
    "metrics_json": "json",
    "details": "string",
    "columns": "json",
    "sat_ids": "json",
    "satellite_name": "string",
    "frame": "string",
    "source_file": "string",
    "processing_level": "string",
    "auxiliary_field": "string",
    "unit": "string",
    "role": "string",
    "supports_event": "boolean",
    "time_uncertainty_seconds": "integer",
    "impulse_count": "integer",
    "eccentricity": "float",
    "bstar": "float",
    "station_id": "string",
    "record_type": "string",
}

TIME_FIELD_NAMES = {
    "epoch",
    "t_start_lo",
    "t_start_hi",
    "t_end_lo",
    "t_end_hi",
    "start_time",
    "end_time",
    "time_start",
    "time_end",
    "event_time_utc",
    "reported_operation_start_utc",
    "reported_operation_end_utc",
    "window_start_utc",
    "window_end_utc",
}


def release_data_dictionary() -> pd.DataFrame:
    """Return the frozen release data dictionary with units and time format."""
    rows = []
    for table_name, fields in DATA_DICTIONARY_TABLE_SCHEMAS.items():
        for field_name in fields:
            rows.append(
                {
                    "table_name": table_name,
                    "field_name": field_name,
                    "required": True,
                    "data_type": _field_data_type(field_name),
                    "unit": FIELD_UNITS.get(field_name, ""),
                    "time_format": "ISO-8601 UTC Z" if field_name in TIME_FIELD_NAMES else "",
                    "description": FIELD_DESCRIPTIONS.get(field_name, ""),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["table_name", "field_name", "required", "data_type", "unit", "time_format", "description"],
    )


def _field_data_type(field_name: str) -> str:
    if field_name in TIME_FIELD_NAMES:
        return "datetime"
    if field_name.endswith("_count"):
        return "integer"
    if field_name in {
        "score",
        "score_sum",
        "score_max",
        "x_m",
        "y_m",
        "z_m",
        "vx_mps",
        "vy_mps",
        "vz_mps",
        "sigma_x_m",
        "sigma_y_m",
        "sigma_z_m",
        "range_m",
        "sigma_m",
        "residual_m",
        "mean_motion_rad_per_min",
        "inclination_rad",
        "mean_anomaly_rad",
        "raan_rad",
        "argument_of_perigee_rad",
        "auxiliary_value",
    }:
        return "float"
    return FIELD_DATA_TYPES.get(field_name, "string")


def stable_id(prefix: str, *parts: object) -> str:
    """Create a deterministic short identifier."""
    payload = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def to_timestamp(value: object) -> pd.Timestamp:
    """Convert a value to a UTC-normalized internal pandas Timestamp."""
    return to_utc_naive(value)


def serialize_metrics(metrics: Dict) -> str:
    """Serialize evidence metrics as JSON."""
    return json.dumps(metrics, ensure_ascii=True, sort_keys=True)


def normalize_time_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with selected timestamp columns serialized as ISO-8601 UTC Z."""
    if df is None or df.empty:
        return df
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = to_utc_iso_series(result[column])
    return result


def release_table_for_export(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a release table before writing public CSV artifacts."""
    result = normalize_time_columns(df, RELEASE_TIME_COLUMNS.get(table_name, []))
    if table_name == "source_inventory":
        result = result.copy()
        if "columns" in result.columns and "source" in result.columns:
            result["columns"] = [
                _source_inventory_columns_for_export(source, columns)
                for source, columns in zip(result["source"], result["columns"])
            ]
    return result


def source_snapshot_for_export(source_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataset-facing source snapshot with UTC times and unit-suffixed fields."""
    table_name = f"source_snapshot_{source_name}"
    columns = SOURCE_SNAPSHOT_TABLE_SCHEMAS.get(table_name)
    if columns is None:
        raise ValueError(f"Unsupported source snapshot type: {source_name}")
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    result = df.copy()
    rename_map = {
        source: target
        for source, target in SOURCE_SNAPSHOT_RENAMES.get(source_name, {}).items()
        if source in result.columns and target not in result.columns
    }
    result = result.rename(columns=rename_map)
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    result = normalize_time_columns(result, ["epoch"])
    return result[columns]


def _source_inventory_columns_for_export(source_name: object, columns_value: object) -> str:
    source = str(source_name)
    table_name = f"source_snapshot_{source}"
    if table_name in SOURCE_SNAPSHOT_TABLE_SCHEMAS:
        return json.dumps(SOURCE_SNAPSHOT_TABLE_SCHEMAS[table_name], ensure_ascii=True)
    if isinstance(columns_value, str):
        return columns_value
    return json.dumps(list(columns_value or []), ensure_ascii=True)


def event_row(
    sat_id: str,
    track: str,
    event_label: str,
    t_start_lo: object,
    t_start_hi: object,
    t_end_lo: object,
    t_end_hi: object,
    event_family: str,
    orbital_effect: str,
    evidence_sources: Iterable[str],
    review_status: str,
    notes: str = "",
    candidate_id: str = "",
    evidence_ids: Iterable[str] | None = None,
    quality_flags: Iterable[str] | None = None,
    ignore_reason: str = "",
    gate_status: Dict | None = None,
) -> Dict:
    """Build a benchmark event row."""
    if event_label not in EVENT_LABELS:
        raise ValueError(f"Unsupported event label: {event_label}")
    if track not in TRACKS:
        raise ValueError(f"Unsupported track: {track}")
    if event_family not in EVENT_FAMILIES:
        raise ValueError(f"Unsupported event family: {event_family}")
    if orbital_effect not in ORBITAL_EFFECTS:
        raise ValueError(f"Unsupported orbital effect: {orbital_effect}")

    t_start_lo = to_timestamp(t_start_lo)
    t_start_hi = to_timestamp(t_start_hi)
    t_end_lo = to_timestamp(t_end_lo)
    t_end_hi = to_timestamp(t_end_hi)
    evidence_sources = sorted(set(evidence_sources))
    evidence_ids = sorted(set(evidence_ids or []))
    quality_flags = sorted(set(quality_flags or []))
    gate_status = gate_status or {}
    event_id = stable_id(
        "evt",
        sat_id,
        track,
        event_label,
        to_utc_iso(t_start_lo),
        to_utc_iso(t_end_hi),
        ",".join(evidence_sources),
    )
    return {
        "event_id": event_id,
        "sat_id": sat_id,
        "track": track,
        "event_label": event_label,
        "t_start_lo": to_utc_iso(t_start_lo),
        "t_start_hi": to_utc_iso(t_start_hi),
        "t_end_lo": to_utc_iso(t_end_lo),
        "t_end_hi": to_utc_iso(t_end_hi),
        "event_family": event_family,
        "orbital_effect": orbital_effect,
        "evidence_sources": ",".join(evidence_sources),
        "candidate_id": candidate_id,
        "evidence_ids": ",".join(evidence_ids),
        "quality_flags": ",".join(quality_flags),
        "ignore_reason": ignore_reason,
        "gate_status_json": serialize_metrics(gate_status),
        "review_status": review_status,
        "notes": notes,
    }


def evidence_row(
    sat_id: str,
    source: str,
    start_time: object,
    end_time: object,
    score: float,
    supports_event: bool,
    signal_class: str,
    metrics: Dict,
    details: str = "",
    candidate_id: str | None = None,
) -> Dict:
    """Build a source evidence row."""
    start_time = to_timestamp(start_time)
    end_time = to_timestamp(end_time)
    evidence_id = stable_id(
        "evd",
        sat_id,
        source,
        to_utc_iso(start_time),
        to_utc_iso(end_time),
        signal_class,
    )
    return {
        "evidence_id": evidence_id,
        "candidate_id": candidate_id,
        "sat_id": sat_id,
        "source": source,
        "start_time": to_utc_iso(start_time),
        "end_time": to_utc_iso(end_time),
        "score": float(score),
        "supports_event": bool(supports_event),
        "signal_class": signal_class,
        "metrics_json": serialize_metrics(metrics),
        "details": details,
    }
