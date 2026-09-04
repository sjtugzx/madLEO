"""Normalize IDS/DORIS maneuver-history files into annotation event windows.

Supported record arities in ``parse_ids_maneuver_line``:

- 9 tokens: operation interval only; the representative event time is the
  reported operation end (``event_time_role="reported_operation_end_time"``).
- 11 tokens: operation interval + ``mission_record_code`` + ``impulse_count``
  with no reported impulse epochs; the representative event time is again the
  reported operation end.
- >= 16 tokens: full record with a first-impulse epoch
  (``event_time_role="first_impulse_time"``).

Any other non-comment line raises ``ValueError``; the file reader propagates
the failure so no record is silently dropped.

``time_uncertainty_seconds`` is the strict additive interval bound on the
representative event time (REVIEW_FINDINGS §8.1), defined once in
``_time_uncertainty_seconds``:

- first-impulse records: string-precision half-width of the reported impulse
  seconds token, ``0.5 * 10**-decimals`` (``"26.812"`` -> ``0.0005`` s,
  integer-second ``"26"`` -> ``0.5`` s);
- operation-end records (9- and 11-token): operation interval width plus the
  ``30.0`` s half-width of the ``HH MM`` string precision,
  ``width_seconds + 30.0`` (zero-width records get ``30.0``, not ``0``);
- any other role: ``NaN`` and the row notes gain ``timing_uncertainty_unknown``.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.schema import (
    MANEUVER_ANNOTATION_TABLE_COLUMNS,
    MANEUVER_EVENT_WINDOW_TABLE_COLUMNS,
    stable_id,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "targets_maneuver_annotations.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"
SOURCE_NAME = "ids_doris_maneuver_history"
SOURCE_PROVIDER = "International DORIS Service"
TRUTH_STATUS = "mission_reported_not_algorithmic_detection"


def ids_year_doy_to_datetime(year: int, day_of_year: int, hour: int, minute: int, second: float = 0.0) -> datetime:
    """Convert IDS year/day-of-year fields into a timezone-aware UTC datetime."""
    if day_of_year < 1:
        raise ValueError(f"Invalid day-of-year {day_of_year} for {year}")
    whole_seconds = int(second)
    microseconds = int(round((second - whole_seconds) * 1_000_000))
    if microseconds == 1_000_000:
        whole_seconds += 1
        microseconds = 0
    value = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day_of_year - 1,
        hours=hour,
        minutes=minute,
        seconds=whole_seconds,
        microseconds=microseconds,
    )
    if value.year != year:
        raise ValueError(f"Invalid day-of-year {day_of_year} for {year}")
    return value


def _to_utc_iso(value: datetime) -> str:
    if value.microsecond:
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def ids_year_doy_to_utc(year: int, day_of_year: int, hour: int, minute: int, second: float = 0.0) -> str:
    """Convert IDS year/day-of-year fields into compact ISO-8601 UTC."""
    return _to_utc_iso(ids_year_doy_to_datetime(year, day_of_year, hour, minute, second))


def _time_uncertainty_seconds(event_time_role: str, tokens: list[str]) -> float:
    """Strict additive interval bound on the representative event time.

    Rules (single definition point for the released ``time_uncertainty_seconds``
    column of maneuver annotations; see the module docstring):

    - ``first_impulse_time``: string-precision half-width of the reported
      impulse seconds token ``tokens[15]``, i.e. ``0.5 * 10**-decimals`` where
      ``decimals`` counts fractional digits of the token (``"26.812"`` ->
      ``0.0005``; integer-second ``"26"`` -> ``0.5``).
    - ``reported_operation_end_time``: operation interval width
      (``tokens[5:9]`` end minus ``tokens[1:5]`` start) plus the ``30.0`` s
      half-width of the ``HH MM`` string precision, i.e.
      ``width_seconds + 30.0``.
    - any other role: ``NaN`` (the caller flags ``timing_uncertainty_unknown``).
    """
    if event_time_role == "first_impulse_time":
        seconds_token = tokens[15]
        decimals = len(seconds_token.partition(".")[2]) if "." in seconds_token else 0
        return 0.5 * 10.0 ** (-decimals)
    if event_time_role == "reported_operation_end_time":
        start = ids_year_doy_to_datetime(int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4]))
        end = ids_year_doy_to_datetime(int(tokens[5]), int(tokens[6]), int(tokens[7]), int(tokens[8]))
        return float((end - start).total_seconds()) + 30.0
    return float("nan")


def _build_annotation_row(
    sat_id: str,
    source_url: str,
    reference: str,
    parts: list[str],
    raw_record: str,
    event_time: datetime,
    event_time_role: str,
    reported_start: datetime,
    reported_end: datetime,
    window_start: datetime,
    window_end: datetime,
    impulse_count: int,
    mission_record_code: str,
    notes: str,
) -> dict:
    """Assemble the normalized annotation row and its stable identifier."""
    # 9-token operation-only records carry an empty mission_record_code but keep
    # the historical "operation_only" sentinel in the stable_id payload so
    # already-shipped annotation_id values remain byte-identical.
    annotation_id = stable_id(
        "ann",
        sat_id,
        SOURCE_NAME,
        _to_utc_iso(event_time),
        _to_utc_iso(reported_start),
        _to_utc_iso(reported_end),
        mission_record_code or "operation_only",
        impulse_count,
    )
    time_uncertainty = _time_uncertainty_seconds(event_time_role, parts)
    if math.isnan(time_uncertainty):
        notes = f"{notes} timing_uncertainty_unknown".strip()
    return {
        "annotation_id": annotation_id,
        "sat_id": sat_id,
        "source": SOURCE_NAME,
        "source_provider": SOURCE_PROVIDER,
        "source_url": source_url,
        "reference": reference,
        "event_type": "maneuver",
        "event_time_utc": _to_utc_iso(event_time),
        "event_time_role": event_time_role,
        "reported_operation_start_utc": _to_utc_iso(reported_start),
        "reported_operation_end_utc": _to_utc_iso(reported_end),
        "window_start_utc": _to_utc_iso(window_start),
        "window_end_utc": _to_utc_iso(window_end),
        "time_uncertainty_seconds": time_uncertainty,
        "impulse_count": impulse_count,
        "mission_record_code": mission_record_code,
        "annotation_status": "external_mission_reported",
        "truth_status": TRUTH_STATUS,
        "raw_record": raw_record,
        "notes": notes,
    }


def parse_ids_maneuver_line(
    line: str,
    sat_id: str,
    source_url: str,
    reference: str = "IDS/DORIS maneuver history",
    pre_event_margin_hours: int = 6,
    post_event_margin_hours: int = 24,
) -> dict:
    """Parse one IDS/DORIS maneuver-history row into the normalized annotation schema.

    Supports 9-token (operation interval only), 11-token (operation interval +
    mission record code + impulse count, no impulse epochs), and >=16-token
    (first-impulse epoch) records; see the module docstring. Any other arity
    raises ``ValueError`` so no record is silently dropped.
    """
    raw_record = line.strip()
    parts = raw_record.split()
    if len(parts) in (9, 11):
        reported_start = ids_year_doy_to_datetime(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        reported_end = ids_year_doy_to_datetime(int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8]))
        event_time = reported_end
        window_start = event_time - timedelta(hours=pre_event_margin_hours)
        window_end = event_time + timedelta(hours=post_event_margin_hours)
        if len(parts) == 11:
            mission_record_code = parts[9]
            impulse_count = int(parts[10])
            notes = (
                "Parsed from IDS/DORIS maneuver history; operation interval carries mission "
                "record code and impulse count without reported impulse epochs, so the event "
                "time uses the reported operation end."
            )
        else:
            mission_record_code = ""
            impulse_count = 0
            notes = "Parsed from IDS/DORIS operation-only maneuver history; event time uses reported operation end."
        return _build_annotation_row(
            sat_id,
            source_url,
            reference,
            parts,
            raw_record,
            event_time,
            "reported_operation_end_time",
            reported_start,
            reported_end,
            window_start,
            window_end,
            impulse_count,
            mission_record_code,
            notes,
        )
    if len(parts) < 16:
        raise ValueError("IDS maneuver line is too short")

    reported_start = ids_year_doy_to_datetime(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
    reported_end = ids_year_doy_to_datetime(int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8]))
    mission_record_code = parts[9]
    impulse_count = int(parts[10])
    event_time = ids_year_doy_to_datetime(int(parts[11]), int(parts[12]), int(parts[13]), int(parts[14]), float(parts[15]))
    window_start = event_time - timedelta(hours=pre_event_margin_hours)
    window_end = event_time + timedelta(hours=post_event_margin_hours)
    return _build_annotation_row(
        sat_id,
        source_url,
        reference,
        parts,
        raw_record,
        event_time,
        "first_impulse_time",
        reported_start,
        reported_end,
        window_start,
        window_end,
        impulse_count,
        mission_record_code,
        "Parsed from IDS/DORIS maneuver history; window is margin-expanded for evidence alignment.",
    )


def _event_windows_from_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    if annotations.empty:
        return pd.DataFrame(columns=MANEUVER_EVENT_WINDOW_TABLE_COLUMNS)
    return annotations[
        [
            "annotation_id",
            "sat_id",
            "window_start_utc",
            "window_end_utc",
            "event_type",
            "source_url",
        ]
    ].assign(event_label="event", confidence_tier="C")[MANEUVER_EVENT_WINDOW_TABLE_COLUMNS]


def normalize_ids_maneuver_file(
    path: str | Path,
    sat_id: str,
    source_url: str,
    reference: str = "IDS/DORIS maneuver history",
    pre_event_margin_hours: int = 6,
    post_event_margin_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize one local IDS/DORIS maneuver-history file.

    Every non-comment line must parse; any parse failure raises so the whole
    file fails loudly (zero silent drops).
    """
    source_path = Path(path)
    rows = []
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(
            parse_ids_maneuver_line(
                line,
                sat_id=sat_id,
                source_url=source_url,
                reference=reference,
                pre_event_margin_hours=pre_event_margin_hours,
                post_event_margin_hours=post_event_margin_hours,
            )
        )

    annotations = pd.DataFrame(rows, columns=MANEUVER_ANNOTATION_TABLE_COLUMNS)
    if not annotations.empty:
        annotations = (
            annotations.drop_duplicates(subset=["annotation_id"], keep="first")
            .sort_values(["event_time_utc", "annotation_id"])
            .reset_index(drop=True)
        )
    return annotations, _event_windows_from_annotations(annotations)


def write_normalized_maneuver_outputs(
    annotations: pd.DataFrame,
    event_windows: pd.DataFrame,
    output_root: str | Path,
    sat_id: str,
) -> dict[str, Path]:
    """Write normalized maneuver annotation and event-window CSV outputs."""
    target_dir = ensure_directory(Path(output_root) / sat_id)
    annotation_path = target_dir / "maneuver_annotations.csv"
    event_window_path = target_dir / "event_windows.csv"
    annotations.to_csv(annotation_path, index=False)
    event_windows.to_csv(event_window_path, index=False)
    return {"annotations": annotation_path, "event_windows": event_window_path}


def load_annotation_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config_path = resolve_repo_path(str(path)) if not isinstance(path, Path) else path
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize IDS/DORIS maneuver-history files")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to maneuver annotation target config")
    parser.add_argument("--sat-id", action="append", default=None, help="Optional satellite id filter")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Normalized reference-validation output root")
    parser.add_argument("--pre-event-margin-hours", type=int, default=6)
    parser.add_argument("--post-event-margin-hours", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_annotation_config(args.config)
    include = set(args.sat_id or [])
    for target in config.get("targets", []):
        if include and target["sat_id"] not in include:
            continue
        raw_path = resolve_repo_path(target["raw_annotation_path"])
        if raw_path is None or not raw_path.exists():
            print(f"{target['sat_id']} status=missing_annotation path={target['raw_annotation_path']}")
            continue
        annotations, event_windows = None, None
        try:
            annotations, event_windows = normalize_ids_maneuver_file(
                raw_path,
                sat_id=target["sat_id"],
                source_url=target["maneuver_history_url"],
                pre_event_margin_hours=args.pre_event_margin_hours,
                post_event_margin_hours=args.post_event_margin_hours,
            )
        except Exception as exc:
            # Legacy Batch C formats (e.g. early SPOT records) may not parse;
            # log and keep going so supported targets still normalize.
            print(f"{target['sat_id']} status=parse_failed error={exc}")
            continue
        paths = write_normalized_maneuver_outputs(
            annotations,
            event_windows,
            output_root=args.output_root,
            sat_id=target["sat_id"],
        )
        print(
            f"{target['sat_id']} annotations={len(annotations)} "
            f"annotation_path={paths['annotations']} event_window_path={paths['event_windows']}"
        )


if __name__ == "__main__":
    main()
