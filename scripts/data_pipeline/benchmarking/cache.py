"""Simple parquet cache helpers for normalized source tables."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory


CACHE_ROOT = REPO_ROOT / "data" / "interim" / "cache" / "benchmark"


def cache_path(source_type: str, raw_path: str | Path, sat_id: str | None = None) -> Path:
    """Build a deterministic cache path for a raw source file."""
    raw_path = Path(raw_path)
    digest = hashlib.sha1(str(raw_path.resolve()).encode("utf-8")).hexdigest()[:16]
    sat_key = sat_id or "unknown"
    return ensure_directory(CACHE_ROOT / source_type / sat_key) / f"{digest}.parquet"


def load_cache(path: Path, raw_path: str | Path) -> Optional[pd.DataFrame]:
    """Load a parquet cache if it is newer than the raw file."""
    raw_path = Path(raw_path)
    if not path.exists():
        return None
    if path.stat().st_mtime < raw_path.stat().st_mtime:
        return None
    return pd.read_parquet(path)


def save_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist a normalized DataFrame to parquet."""
    ensure_directory(path.parent)
    df.to_parquet(path, index=False)
