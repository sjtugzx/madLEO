"""Manifest loading and path resolution helpers for benchmark pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "benchmark_targets.json"


def load_manifest(manifest_path: str | None = None) -> Dict:
    """Load the benchmark manifest JSON file."""
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_repo_path(path_str: str | None) -> Path | None:
    """Resolve a repo-relative path to an absolute Path."""
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def iter_enabled_sources(target: Dict) -> Iterator[Tuple[str, Dict]]:
    """Yield enabled sources from a target configuration."""
    for source_name, config in target.get("sources", {}).items():
        if config.get("enabled", False):
            yield source_name, config


def target_date_range(target: Dict) -> Tuple[str, str]:
    """Return date range strings from the target config."""
    date_range = target.get("date_range", {})
    return date_range.get("start", ""), date_range.get("end", "")


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path

