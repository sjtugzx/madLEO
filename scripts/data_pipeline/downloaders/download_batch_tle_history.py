"""Planner and executor for Batch A TLE history acquisition windows.

Prints a dry-run plan by default; ``--execute-spacetrack-fallback`` runs the
authenticated Space-Track GP history fallback and writes download metadata.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from alignment.normalize_maneuver_annotations import parse_ids_maneuver_line


DEFAULT_CONFIG = REPO_ROOT / "configs" / "targets_maneuver_annotations.json"
DEFAULT_METADATA_ROOT = REPO_ROOT / "data" / "validation" / "download_status" / "benchmark"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _to_utc_iso(value: datetime) -> str:
    if value.microsecond:
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_annotation_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load the mission-reported maneuver annotation source config."""
    config_path = resolve_repo_path(str(path)) if not isinstance(path, Path) else path
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_metadata_root(metadata_root: str | Path) -> Path:
    path = Path(metadata_root)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def iter_targets_by_batch(
    config: dict,
    batches: Iterable[str] = ("A",),
    include: Iterable[str] | None = None,
) -> list[dict]:
    """Return targets from selected batches, optionally filtered by satellite id."""
    batch_set = set(batches)
    include_set = set(include or [])
    targets = [target for target in config.get("targets", []) if target.get("batch") in batch_set]
    if include_set:
        targets = [target for target in targets if target.get("sat_id") in include_set]
    return targets


def iter_batch_a_targets(config: dict, include: Iterable[str] | None = None) -> list[dict]:
    """Return Batch A targets, optionally filtered by satellite id."""
    return iter_targets_by_batch(config, batches=["A"], include=include)


def _build_spacetrack_client():
    from benchmarking.spacetrack import SpaceTrackClient

    return SpaceTrackClient.from_env()


def load_ids_annotation_events(path: str | Path, sat_id: str, source_url: str = "") -> list[dict]:
    """Parse local IDS/DORIS maneuver-history rows into event dictionaries."""
    annotation_path = Path(path)
    events: list[dict] = []
    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            events.append(parse_ids_maneuver_line(line, sat_id=sat_id, source_url=source_url))
        except ValueError as exc:
            if "too short" in str(exc) and line.split()[-2:] == ["007", "0"]:
                continue
            continue
    return sorted(events, key=lambda event: event["event_time_utc"])


def compute_event_window(
    events: list[dict],
    margin_days: int = 30,
    clip: tuple[str, str] | None = None,
) -> dict:
    """Compute a TLE acquisition window from mission-reported event times."""
    if not events:
        raise ValueError("At least one parsed IDS event is required")

    event_times = sorted(_parse_utc(event["event_time_utc"]) for event in events)
    unclipped_start = event_times[0] - timedelta(days=margin_days)
    unclipped_end = event_times[-1] + timedelta(days=margin_days)
    start = unclipped_start
    end = unclipped_end
    window_source = "ids_doris_maneuver_history"

    if clip is not None:
        clip_start, clip_end = (_parse_utc(clip[0]), _parse_utc(clip[1]))
        start = max(start, clip_start)
        end = min(end, clip_end)
        window_source = "ids_doris_maneuver_history_clipped_to_primary_window"

    return {
        "window_start_utc": _to_utc_iso(start),
        "window_end_utc": _to_utc_iso(end),
        "unclipped_window_start_utc": _to_utc_iso(unclipped_start),
        "unclipped_window_end_utc": _to_utc_iso(unclipped_end),
        "window_source": window_source,
    }


def build_tle_acquisition_plan(
    target: dict,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    clip: tuple[str, str] | None = None,
) -> dict:
    """Build one dry-run TLE acquisition plan row for a target."""
    annotation_path = resolve_repo_path(target["raw_annotation_path"])
    metadata_dir = _resolve_metadata_root(metadata_root) / target["sat_id"] / "tle"
    base = {
        "sat_id": target["sat_id"],
        "norad": target.get("norad", ""),
        "raw_annotation_path": str(annotation_path),
        "tle_output_dir": target["tle_raw_path"],
        "metadata_dir": str(metadata_dir),
        "preferred_source_status": "public_archive_pending_or_unavailable",
        "fallback_source": "space_track_gp_history_authenticated",
        "requires_credentials": True,
    }

    if not base["norad"]:
        return {
            **base,
            "status": "missing_norad",
            "window_start_utc": "",
            "window_end_utc": "",
            "window_source": "",
            "next_action": "verify target NORAD id before TLE acquisition",
        }

    if annotation_path is None or not annotation_path.exists():
        return {
            **base,
            "status": "missing_annotation",
            "window_start_utc": "",
            "window_end_utc": "",
            "window_source": "",
            "next_action": "run download_ids_maneuver_histories.py --execute",
        }

    events = load_ids_annotation_events(
        annotation_path,
        sat_id=target["sat_id"],
        source_url=target.get("maneuver_history_url", ""),
    )
    if not events:
        return {
            **base,
            "status": "no_parseable_events",
            "window_start_utc": "",
            "window_end_utc": "",
            "window_source": "",
            "next_action": "inspect IDS maneuver-history format",
        }

    return {
        **base,
        "status": "planned",
        **compute_event_window(events, margin_days=30, clip=clip),
        "next_action": "acquire TLE history from public archive or authenticated fallback",
    }


def execute_tle_acquisition_plan(plan: dict, client=None) -> dict:
    """Execute or explicitly block one TLE acquisition plan with metadata."""
    metadata_dir = ensure_directory(Path(plan["metadata_dir"]))
    base_metadata = {
        "sat_id": plan["sat_id"],
        "norad": plan.get("norad", ""),
        "window_start_utc": plan.get("window_start_utc", ""),
        "window_end_utc": plan.get("window_end_utc", ""),
        "tle_output_dir": plan.get("tle_output_dir", ""),
        "fallback_source": "space_track_gp_history_authenticated",
        "output_path": "",
    }
    if plan.get("status") != "planned":
        metadata = {**base_metadata, "status": plan.get("status", "not_planned")}
    else:
        try:
            active_client = client or _build_spacetrack_client()
            active_client.login()
            start_date = _parse_utc(plan["window_start_utc"]).date().isoformat()
            end_date = _parse_utc(plan["window_end_utc"]).date().isoformat()
            output_path = active_client.download_tle_history(
                norad=str(plan["norad"]),
                start=start_date,
                end=end_date,
                output_dir=resolve_repo_path(plan["tle_output_dir"]),
            )
            metadata = {**base_metadata, "status": "downloaded", "output_path": str(output_path)}
        except RuntimeError as exc:
            if "SPACETRACK_ID and SPACETRACK_PASSWORD" not in str(exc):
                raise
            metadata = {**base_metadata, "status": "credentials_missing"}
    (metadata_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def render_tle_plan(plan: dict) -> str:
    """Render one TLE acquisition plan as a stable dry-run line."""
    requires_credentials = str(bool(plan.get("requires_credentials", False))).lower()
    return (
        f"{plan['sat_id']} status={plan['status']} "
        f"window={plan.get('window_start_utc', '')}..{plan.get('window_end_utc', '')} "
        f"tle_output={plan.get('tle_output_dir', '')} fallback={plan.get('fallback_source', '')} "
        f"requires_credentials={requires_credentials}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan Batch A TLE history acquisition windows")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to maneuver annotation target config")
    parser.add_argument("--metadata-root", default=str(DEFAULT_METADATA_ROOT), help="Download metadata root directory")
    parser.add_argument("--batch", action="append", default=None, help="Batch letter to include; repeatable")
    parser.add_argument("--include", nargs="*", default=None, help="Optional sat_id filters")
    parser.add_argument("--clip-primary-window", action="store_true", help="Clip event-derived windows to config window")
    parser.add_argument("--dry-run", action="store_true", help="Print acquisition plan without network access")
    parser.add_argument(
        "--execute-spacetrack-fallback",
        action="store_true",
        help="Execute authenticated Space-Track GP history fallback and write ignored raw TLE files",
    )
    args = parser.parse_args()
    if args.dry_run and args.execute_spacetrack_fallback:
        parser.error("--dry-run cannot be combined with --execute-spacetrack-fallback")
    return args


def main() -> None:
    args = parse_args()
    config = load_annotation_config(args.config)
    release_window = config.get("release_window", {})
    clip = None
    if args.clip_primary_window:
        clip = (release_window["primary_start"], release_window["primary_end"])
    for target in iter_targets_by_batch(config, batches=args.batch or ["A"], include=args.include):
        plan = build_tle_acquisition_plan(target, metadata_root=args.metadata_root, clip=clip)
        if args.execute_spacetrack_fallback:
            metadata = execute_tle_acquisition_plan(plan)
            print(f"{metadata['sat_id']} status={metadata['status']} output={metadata['output_path']}")
        else:
            print(render_tle_plan(plan))


if __name__ == "__main__":
    main()
