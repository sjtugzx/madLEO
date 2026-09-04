"""Canonicalize existing release CSV artifacts to the dataset schema contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from benchmarking.schema import RELEASE_TABLE_SCHEMAS, release_data_dictionary, release_table_for_export


# C19: the retired `leo-orbital-dynamics` tree no longer exists; the default
# follows the current mad-leo release workspace layout.
DEFAULT_RELEASE_ROOT = Path("data/release/mad-leo/reference_validation_subset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize release CSV timestamp and schema formatting")
    parser.add_argument(
        "--release-root",
        default=str(DEFAULT_RELEASE_ROOT),
        help="Release root containing top-level and per-target CSV artifacts",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Optional release table names to canonicalize; defaults to all release CSV tables",
    )
    return parser.parse_args()


def _release_csv_paths(release_root: Path, table_name: str) -> Iterable[Path]:
    filename = f"{table_name}.csv"
    top_level = release_root / filename
    if top_level.exists():
        yield top_level
    for child in sorted(release_root.iterdir() if release_root.exists() else []):
        if child.is_dir():
            path = child / filename
            if path.exists():
                yield path


def canonicalize_release_schema(
    release_root: Path = DEFAULT_RELEASE_ROOT,
    table_names: Iterable[str] | None = None,
) -> list[Path]:
    """Rewrite existing release CSVs with canonical UTC timestamps and dictionary metadata."""
    written: list[Path] = []
    selected = list(table_names or RELEASE_TABLE_SCHEMAS)
    unsupported = sorted(set(selected) - set(RELEASE_TABLE_SCHEMAS))
    if unsupported:
        raise ValueError(f"Unsupported release table names: {unsupported}")
    for table_name in selected:
        for path in _release_csv_paths(release_root, table_name):
            _canonicalize_csv(path, table_name)
            written.append(path)

    metadata_dir = release_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = metadata_dir / "data_dictionary.csv"
    release_data_dictionary().to_csv(dictionary_path, index=False)
    written.append(dictionary_path)
    return written


def _canonicalize_csv(path: Path, table_name: str, chunksize: int = 100_000) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    wrote = False
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            release_table_for_export(table_name, chunk).to_csv(tmp_path, index=False, mode="a", header=not wrote)
            wrote = True
    except pd.errors.EmptyDataError:
        pd.DataFrame(columns=RELEASE_TABLE_SCHEMAS[table_name]).to_csv(tmp_path, index=False)
        wrote = True

    if not wrote:
        pd.DataFrame(columns=RELEASE_TABLE_SCHEMAS[table_name]).to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    canonicalize_release_schema(Path(args.release_root), table_names=args.tables)


if __name__ == "__main__":
    main()
