"""Planner and executor for IDS/DORIS maneuver-history downloads.

Prints a dry-run plan by default; ``--execute`` downloads the public
maneuver-history files and writes download metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable
from datetime import datetime, timezone

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from downloaders.net_guard import guarded_request


DEFAULT_CONFIG = REPO_ROOT / "configs" / "targets_maneuver_annotations.json"
DEFAULT_METADATA_ROOT = REPO_ROOT / "data" / "validation" / "download_status" / "benchmark"


def load_annotation_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load the mission-reported maneuver annotation source config."""
    config_path = resolve_repo_path(str(path)) if not isinstance(path, Path) else path
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


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


def build_ids_download_plan(target: dict, metadata_root: str | Path = DEFAULT_METADATA_ROOT) -> dict:
    """Build one dry-run IDS/DORIS maneuver-history download plan row."""
    metadata_dir = Path(metadata_root) / target["sat_id"] / "annotations"
    return {
        "sat_id": target["sat_id"],
        "display_name": target["display_name"],
        "provider": "International DORIS Service",
        "source_url": target["maneuver_history_url"],
        "destination_path": target["raw_annotation_path"],
        "metadata_dir": str(metadata_dir),
        "license_or_terms_status": target["license_or_terms_status"],
        "redistribution_status": target["redistribution_status"],
        "would_write": False,
    }


def _fetch_url(url: str) -> bytes:
    response = guarded_request(
        "GET",
        url,
        headers={"User-Agent": "H2G2-Orbit IDS/DORIS source planner"},
        timeout=60,
    )
    response.raise_for_status()
    return response.content


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def execute_ids_download(plan: dict, fetcher=_fetch_url) -> dict:
    """Download one public IDS/DORIS maneuver-history file and write metadata."""
    payload = fetcher(plan["source_url"])
    destination = Path(plan["destination_path"])
    metadata_dir = Path(plan["metadata_dir"])
    ensure_directory(destination.parent)
    ensure_directory(metadata_dir)
    destination.write_bytes(payload)
    metadata = {
        "sat_id": plan["sat_id"],
        "source_url": plan["source_url"],
        "destination_path": str(destination),
        "bytes_written": len(payload),
        "line_count": len(payload.decode("utf-8", errors="replace").splitlines()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "downloaded_at_utc": _utc_now_iso(),
        "status": "downloaded",
    }
    (metadata_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def render_ids_plan(plan: dict) -> str:
    """Render one IDS download plan as a stable dry-run line."""
    would_write = str(bool(plan.get("would_write", False))).lower()
    return (
        f"{plan['sat_id']} source={plan['source_url']} "
        f"dest={plan['destination_path']} metadata={plan['metadata_dir']} "
        f"provider={plan['provider']} would_write={would_write}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan IDS/DORIS maneuver-history downloads")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to maneuver annotation target config")
    parser.add_argument("--metadata-root", default=str(DEFAULT_METADATA_ROOT), help="Download metadata root directory")
    parser.add_argument("--batch", action="append", default=None, help="Batch letter to include; repeatable")
    parser.add_argument("--include", nargs="*", default=None, help="Optional sat_id filters")
    parser.add_argument("--dry-run", action="store_true", help="Print download plan without writing raw data")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for future network downloads; not used by tests and never writes credentials",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_annotation_config(args.config)
    for target in iter_targets_by_batch(config, batches=args.batch or ["A"], include=args.include):
        plan = build_ids_download_plan(target, metadata_root=args.metadata_root)
        if args.execute:
            plan["would_write"] = True
            metadata = execute_ids_download(plan)
            print(f"{metadata['sat_id']} status={metadata['status']} bytes={metadata['bytes_written']}")
        else:
            print(render_ids_plan(plan))


if __name__ == "__main__":
    main()
