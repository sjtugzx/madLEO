"""Download orchestration for benchmark source data."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from benchmarking.config import ensure_directory, iter_enabled_sources, load_manifest, resolve_repo_path, target_date_range
from benchmarking.metadata import build_source_download_metadata, metadata_root, target_metadata_dir, write_json, write_jsonl
from benchmarking.spacetrack import SpaceTrackClient
from downloaders.cdse_downloader import download_cdse_sentinel3_orbits, list_cdse_sentinel3_orbits
from downloaders.gfz_downloader import download_gfz_orbit, list_gfz_files
from downloaders.ilrs_downloader import download_ilrs_data, list_edc_files
from downloaders.podaac_downloader import download_podaac_jason3_ogdr_gps, list_podaac_jason3_granules
from downloaders.sentinel_downloader import download_sentinel_orbits, list_aws_orbits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw source data for the maneuver benchmark")
    parser.add_argument("--manifest", default=None, help="Path to benchmark manifest JSON")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for downloads per source")
    parser.add_argument("--include", nargs="*", default=None, help="Optional subset of target sat_id values")
    parser.add_argument("--sources", nargs="*", default=None, help="Optional subset of source names, e.g. pod tle slr")
    parser.add_argument(
        "--metadata-root",
        default="data/validation/download_status/benchmark",
        help="Output root for download metadata",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retry count for each source download step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    include = set(args.include or [])
    source_filter = set(args.sources or [])
    spacetrack_client = None
    metadata_dir = ensure_directory(metadata_root(args.metadata_root))

    for target in manifest.get("targets", []):
        sat_id = target["sat_id"]
        if include and sat_id not in include:
            continue

        start_str, end_str = target_date_range(target)
        start_date = datetime.fromisoformat(start_str)
        end_date = datetime.fromisoformat(end_str)
        target_meta_dir = target_metadata_dir(metadata_dir, sat_id)
        write_json(target_meta_dir / "target_config.json", target)
        target_status: Dict[str, Dict] = {}

        for source_name, source_config in iter_enabled_sources(target):
            if source_filter and source_name not in source_filter:
                continue
            output_dir = ensure_directory(resolve_repo_path(source_config.get("output_dir")))
            provider = source_config.get("provider", "")
            source_meta_dir = target_metadata_dir(metadata_dir, sat_id, source_name)
            write_json(source_meta_dir / "source_config.json", source_config)
            listing: List[Dict] = []
            downloaded_files: List[Path] = []
            extra: Dict = {}
            success = False
            skipped = False
            last_error = None
            for attempt in range(1, args.retries + 1):
                try:
                    listing = []
                    downloaded_files = []
                    extra = {}
                    if source_name == "pod" and provider == "sentinel":
                        listing = list_aws_orbits(
                            source_config["satellite"],
                            start_date=start_date,
                            end_date=end_date,
                            verbose=False,
                        )
                        downloaded = download_sentinel_orbits(
                            source_config["satellite"],
                            start_date,
                            end_date,
                            str(output_dir),
                            max_files=args.max_files,
                            verbose=True,
                        )
                        downloaded_files = [Path(path) for path in downloaded]
                    elif source_name == "pod" and provider == "cdse_sentinel3":
                        listing = list_cdse_sentinel3_orbits(
                            source_config["satellite"],
                            start_date=start_date,
                            end_date=end_date,
                            product_type=source_config.get("product_type", "AUX_POEORB"),
                            verbose=False,
                        )
                        downloaded = download_cdse_sentinel3_orbits(
                            source_config["satellite"],
                            start_date,
                            end_date,
                            str(output_dir),
                            product_type=source_config.get("product_type", "AUX_POEORB"),
                            max_files=args.max_files,
                            verbose=True,
                        )
                        downloaded_files = [Path(path) for path in downloaded]
                    elif source_name == "pod" and provider == "gfz":
                        listing = list_gfz_files(
                            start_date,
                            end_date,
                            product_type=source_config.get("product_type", "rapid"),
                            verbose=False,
                        )
                        downloaded = download_gfz_orbit(
                            start_date,
                            end_date,
                            str(output_dir),
                            product_type=source_config.get("product_type", "rapid"),
                            max_files=args.max_files,
                            verbose=True,
                        )
                        downloaded_files = [Path(path) for path in downloaded]
                    elif source_name == "pod" and provider == "podaac_jason3":
                        listing = list_podaac_jason3_granules(
                            start_date=start_date,
                            end_date=end_date,
                            verbose=False,
                        )
                        downloaded = download_podaac_jason3_ogdr_gps(
                            start_date=start_date,
                            end_date=end_date,
                            output_dir=str(output_dir),
                            max_files=args.max_files,
                            verbose=True,
                        )
                        downloaded_files = [Path(path) for path in downloaded]
                    elif source_name == "slr":
                        listing = [{"remote_path": path} for path in list_edc_files(
                            source_config.get("satellite", sat_id),
                            data_type=source_config.get("data_type", "npt"),
                            start_date=start_date,
                            end_date=end_date,
                            verbose=False,
                        )]
                        downloaded = download_ilrs_data(
                            source_config.get("satellite", sat_id),
                            start_date,
                            end_date,
                            str(output_dir),
                            data_type=source_config.get("data_type", "npt"),
                            max_files=args.max_files,
                            verbose=True,
                        )
                        downloaded_files = [Path(path) for path in downloaded]
                    elif source_name in {"tle", "ephemeris"}:
                        if spacetrack_client is None:
                            spacetrack_client = SpaceTrackClient.from_env()
                            spacetrack_client.login()
                        if source_name == "tle":
                            norad = source_config.get("norad")
                            if not norad:
                                print(f"Skipping {sat_id} TLE download: missing NORAD in manifest")
                                target_status[source_name] = {"status": "skipped", "reason": "missing_norad"}
                                skipped = True
                                break
                            extra["satcat"] = spacetrack_client.get_satcat_metadata(norad)
                            downloaded_files = [spacetrack_client.download_tle_history(norad, start_str, end_str, output_dir)]
                        else:
                            name_contains = source_config.get("name_contains") or [sat_id]
                            files = spacetrack_client.list_public_ephemeris(name_contains=name_contains)
                            listing = files
                            downloaded_files = spacetrack_client.download_public_files(files, output_dir, limit=args.max_files)
                            extra["filters"] = {"name_contains": name_contains}
                    else:
                        print(f"Skipping unsupported source {source_name} for {sat_id}")
                        target_status[source_name] = {"status": "skipped", "reason": f"unsupported_provider:{provider}"}
                        skipped = True
                        break
                    success = True
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"{sat_id} {source_name} attempt {attempt}/{args.retries} failed: {exc}")
                    if attempt == args.retries:
                        break
            if skipped:
                continue
            if not success:
                exc = last_error
                target_status[source_name] = {
                    "status": "failed",
                    "provider": provider,
                    "error": str(exc),
                }
                write_json(
                    source_meta_dir / "download_error.json",
                    {
                        "sat_id": sat_id,
                        "source": source_name,
                        "provider": provider,
                        "error": str(exc),
                    },
                )
                print(f"{sat_id} {source_name} download failed: {exc}")
                continue

            if args.max_files is not None and listing:
                listing = listing[:args.max_files]
            source_metadata = build_source_download_metadata(
                sat_id=sat_id,
                source_name=source_name,
                provider=provider,
                date_range={"start": start_str, "end": end_str},
                source_config=source_config,
                listing=listing,
                downloaded_files=downloaded_files,
                extra=extra,
            )
            write_json(source_meta_dir / "download_metadata.json", source_metadata)
            write_json(source_meta_dir / "listing.json", list(listing))
            write_jsonl(source_meta_dir / "downloaded_files.jsonl", source_metadata["downloaded_files"])
            target_status[source_name] = {
                "status": "downloaded",
                "provider": provider,
                "download_count": source_metadata["download_count"],
                "listing_count": source_metadata["listing_summary"]["count"],
            }

        write_json(target_meta_dir / "download_status.json", target_status)


if __name__ == "__main__":
    main()
