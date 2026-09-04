"""Export normalized source CSV snapshots for selected benchmark targets.

Snapshots are written through the release schema canonicalizer
(`source_snapshot_for_export`) so that this export and `build.py` produce the
identical dataset-facing schema on the same paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking.config import ensure_directory, iter_enabled_sources, load_manifest, resolve_repo_path
from benchmarking.normalization import load_csv_inputs, load_pod_inputs, load_slr_inputs
from benchmarking.schema import source_snapshot_for_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export normalized source CSV snapshots for selected targets")
    parser.add_argument("--manifest", default=None, help="Path to benchmark manifest JSON")
    parser.add_argument(
        "--release-root",
        default="data/release/mad-leo/reference_validation_subset",
        help="Benchmark release root where per-target sources directories will be written",
    )
    parser.add_argument("--include", nargs="*", default=None, help="Optional subset of sat_id values")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional subset of source names, e.g. pod slr tle",
    )
    return parser.parse_args()


def load_source(source_name: str, sat_id: str, output_dir: Path):
    paths = [output_dir]
    if source_name == "pod":
        return load_pod_inputs(paths, sat_id=sat_id)
    if source_name == "slr":
        return load_slr_inputs(paths, sat_id=sat_id)
    if source_name == "tle":
        return load_csv_inputs(paths, source_type="tle", sat_id=sat_id)
    if source_name == "ephemeris":
        return load_csv_inputs(paths, source_type="ephemeris", sat_id=sat_id)
    raise ValueError(f"Unsupported source export: {source_name}")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    include = set(args.include or [])
    source_filter = set(args.sources or [])
    release_root = resolve_repo_path(args.release_root)
    if release_root is None:
        raise ValueError("release_root could not be resolved")

    for target in manifest.get("targets", []):
        sat_id = target["sat_id"]
        if include and sat_id not in include:
            continue

        target_sources_dir = ensure_directory(release_root / sat_id / "sources")

        for source_name, source_config in iter_enabled_sources(target):
            if source_filter and source_name not in source_filter:
                continue
            raw_output_dir = resolve_repo_path(source_config.get("output_dir"))
            if raw_output_dir is None or not raw_output_dir.exists():
                print(f"Skipping {sat_id} {source_name}: raw output directory missing")
                continue

            df = load_source(source_name, sat_id=sat_id, output_dir=raw_output_dir)
            if df is None or df.empty:
                print(f"Skipping {sat_id} {source_name}: normalized dataframe is empty")
                continue

            destination = target_sources_dir / f"{source_name}.csv"
            source_snapshot_for_export(source_name, df).to_csv(destination, index=False)
            print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
