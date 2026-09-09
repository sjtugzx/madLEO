"""Assemble the MAD-LEO public release tree.

Builds the two-subset release layout (mission_reported/ annotations /
evidence + operational / docs + manifest) from local artifacts. Evidence
snapshots use Parquet (zstd); annotations stay CSV for accessibility.
Raw provider files are NOT copied — the archive repository carries them.

The statistical technical-validation tables are NOT part of the dataset:
the validation stage exports them to ``experiments/validation/`` and the
Starlink Q4 analysis tables to ``experiments/starlink/`` (both tracked with
the code), independent of ``--output-dir``.

Layout:
    mad-leo/
    ├── README.md, manifest.json
    ├── mission_reported/
    │   ├── annotations/     (CSV)
    │   └── evidence/        (Parquet: tle, orbit, slr per target)
    ├── operational/starlink/ (Parquet full-constellation 107h slice)
    └── docs/                (schema, protocol, limitations)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Flat-package bootstrap (same as the process.py dispatcher): make the
# data_pipeline packages importable when this module is run as a script.
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path  # noqa: E402
from benchmarking.experiment_params import EARTH_RADIUS_KM  # noqa: E402
from processors.physical_qc import (  # noqa: E402
    EARTH_MU_M3_PER_S2,
    PhysicalQCViolation,
    assert_orbit_records,
    ecef_velocity_to_inertial,
)
from processors.slr_formats import mission_window_for_sat  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables"
INTERIM = REPO_ROOT / "data" / "interim" / "normalized_sources"
RAW = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "release" / "mad-leo"
# Technical-validation artifacts are published with the code, not the
# dataset: the validation stage and the Starlink Q4 tables export here.
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

TARGETS = [
    "sentinel-3a", "sentinel-3b", "jason-3", "sentinel-6a", "cryosat-2", "saral",
    "jason-1", "jason-2", "topex-poseidon", "hy-2a", "swot",
]

ANNOTATION_FILES = {
    "maneuver_annotations.csv": "combined per-target maneuver_annotations.csv",
    "event_windows.csv": "combined annotated_event_windows.csv (tiers)",
    "stable_windows.csv": "results/tables/stable_windows.csv",
    "stable_windows_matched.csv": "results/tables/stable_windows_matched.csv",
}

VALIDATION_TABLES = [
    "reference_all_window_alignment_status.csv",
    "reference_annotation_alignment_summary.csv",
    "maneuver_annotation_summary.csv",
    "maneuver_evidence_alignment_summary.csv",
    "maneuver_confidence_tier_summary.csv",
    "maneuver_event_response_validation.csv",
    "maneuver_event_response_summary.csv",
    "stable_window_response_validation.csv",
    "maneuver_event_state_estimates.csv",
    "maneuver_event_fused_states.csv",
    "harmonization_coverage.csv",
    "harmonization_validation_summary.csv",
    "harmonization_sigma_split_validation.csv",
    "harmonization_confidence_discrimination.csv",
    "sgp4_propagation_residuals.csv",
    "sgp4_propagation_residuals_clean.csv",
    "orbit_interpolation_selfcheck.csv",
    "source_bias_calibration.csv",
    "annotation_response_floor.csv",
    "annotation_uncertainty_strata.csv",
    "sign_agreement_by_magnitude.csv",
    "tier_response_distribution.csv",
    "tier_sign_agreement.csv",
    "tier_sigma_coverage.csv",
    "matched_specificity.csv",
    "fusion_value_summary.csv",
    "fusion_value_per_window.csv",
    "external_benchmark_crossvalidation.csv",
    "external_benchmark_crossvalidation_per_event.csv",
    "tle_element_distribution_summary.csv",
    "orbit_state_distribution_summary.csv",
    "slr_distribution_summary.csv",
    "tier_distribution_equivalence.csv",
    "stable_windows.csv",
    "stable_windows_summary.csv",
    "stable_windows_matched.csv",
    "headline_statistics.csv",
    "gap_taxonomy_completeness.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def stage_annotations(output: Path) -> None:
    """Copy/combine the label-layer tables."""
    dest = ensure_directory(output / "mission_reported" / "annotations")
    annotation_frames = []
    window_frames = []
    for sat_id in TARGETS:
        base = INTERIM / "reference_validation" / sat_id
        ann_path = base / "maneuver_annotations.csv"
        win_path = base / "annotated_event_windows.csv"
        if ann_path.exists():
            annotation_frames.append(pd.read_csv(ann_path))
        if win_path.exists():
            window_frames.append(pd.read_csv(win_path))
    pd.concat(annotation_frames, ignore_index=True).to_csv(dest / "maneuver_annotations.csv", index=False)
    pd.concat(window_frames, ignore_index=True).to_csv(dest / "event_windows.csv", index=False)
    for name in ("stable_windows.csv", "stable_windows_matched.csv"):
        src = TABLES / name
        if src.exists():
            # The recommended split was removed from the public release;
            # it remains only in the local planning tables.
            frame = pd.read_csv(src).drop(columns=["split"], errors="ignore")
            frame.to_csv(dest / name, index=False)
    print("annotations stage done", flush=True)


def stage_validation(output: Path) -> None:
    """Copy the audit/experiment tables to ``experiments/validation/``.

    Missing inputs are reported loudly (C2): a silently incomplete
    ``experiments/validation/`` tree must never look like a complete export.
    These tables are analysis artifacts published with the code, not part of
    the dataset release tree (``output``).
    """
    dest = ensure_directory(EXPERIMENTS_ROOT / "validation")
    copied = 0
    missing: list[str] = []
    for name in VALIDATION_TABLES:
        src = TABLES / name
        if src.exists():
            if name in {"stable_windows.csv", "stable_windows_matched.csv"}:
                frame = pd.read_csv(src).drop(columns=["split"], errors="ignore")
                frame.to_csv(dest / name, index=False)
            else:
                shutil.copy(src, dest / name)
            copied += 1
        else:
            missing.append(name)
    print(f"validation stage done ({copied} tables)", flush=True)
    if missing:
        print(
            f"WARNING: {len(missing)} validation tables missing from {TABLES} "
            f"(experiments/validation/ will be INCOMPLETE): {', '.join(missing)}",
            flush=True,
        )


def stage_evidence_tle(output: Path) -> None:
    """Full-mission normalized TLE per target."""
    from benchmarking.normalization import parse_raw_tle_file

    dest = ensure_directory(output / "mission_reported" / "evidence" / "tle")
    for sat_id in TARGETS:
        tle_dir = RAW / "reference_validation" / sat_id / "tle"
        if not tle_dir.is_dir():
            continue
        frames = []
        for path in sorted(tle_dir.iterdir()):
            if path.suffix.lower() not in {".tle", ".txt"} or path.name.startswith("."):
                continue
            frame = parse_raw_tle_file(path, sat_id=sat_id)
            if not frame.empty:
                frames.append(frame)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged["epoch"] = pd.to_datetime(merged["epoch"], utc=True)
            merged = merged.drop_duplicates(subset=["epoch"]).sort_values("epoch")
            # Dataset-facing schema contract: angular fields carry _rad
            # suffixes, mean motion is mean_motion_rad_per_min.
            merged = merged.rename(
                columns={
                    "mean_motion": "mean_motion_rad_per_min",
                    "inclination": "inclination_rad",
                    "bstar": "bstar_per_earth_radius",
                }
            )
            _write_parquet(merged, dest / f"{sat_id}.parquet")
            print(f"  tle {sat_id}: {len(merged)} epochs", flush=True)


# Dataset-facing schema contract: positions in meters carry _m suffixes,
# velocities in m/s carry _mps suffixes. SP3/NetCDF parsers emit compact
# aliases, so the snapshot writer renames them before writing.
_ORBIT_CANONICAL_COLUMNS = {
    "x": "x_m",
    "y": "y_m",
    "z": "z_m",
    "vx": "vx_mps",
    "vy": "vy_mps",
    "vz": "vz_mps",
    "sigma_x": "sigma_x_m",
    "sigma_y": "sigma_y_m",
    "sigma_z": "sigma_z_m",
}
# Inverse rename applied to the published (shipped) parquet rows so the
# retention seed enters the merge in the same compact column space the
# parsers emit, and physical_qc.assert_orbit_records can run on the merged
# frame before the canonical rename.
_ORBIT_COMPACT_COLUMNS = {v: k for k, v in _ORBIT_CANONICAL_COLUMNS.items()}

# P3 evidence-floor inputs (WP7).  The published dataset tree is read-only:
# its orbit parquets are the RETENTION SEED (rows already released are never
# lost); bulk precise-orbit products and the A1 backfill manifest extend the
# floor, and same-epoch rows are resolved by the dedup rules below.
PUBLISHED_ROOT = REPO_ROOT / "dataset"
BACKFILL_MANIFEST = REPO_ROOT / "data" / "interim" / "backfill_manifest.csv"
DEFAULT_STAGING_ROOT = REPO_ROOT / "results" / "staging_evidence"

# Every orbit product family in the release is Earth-fixed (ITRF SP3, ECEF
# EOF/OGDR/POR/VOR), so the merged frame is one rotating frame and the
# vis-viva assertions use the omega x r correction throughout.
_ORBIT_MERGED_ROTATING = True

# Dedup rule names recorded in the staging QC ledger.
DEDUP_POR_BEATS_VOR = "por_beats_vor"
DEDUP_POE_BEATS_MOE = "poeorb_beats_moeorb"
DEDUP_LATER_ARC_START = "later_arc_start"
DEDUP_PARSED_OVER_PUBLISHED = "parsed_over_published"
DEDUP_TIEBREAK = "same_arc_tiebreak"
DEDUP_RULES = (
    DEDUP_POR_BEATS_VOR,
    DEDUP_POE_BEATS_MOE,
    DEDUP_LATER_ARC_START,
    DEDUP_PARSED_OVER_PUBLISHED,
    DEDUP_TIEBREAK,
)

_ORIGIN_FILE = "file"
_ORIGIN_PUBLISHED = "published"


def orbit_file_family_rank(source_name: str) -> int:
    """Product-family rank for same-epoch resolution (lower wins).

    Binding rule (a): SWOT precise-orbit POR beats VOR -- the paper's
    source declaration is POR-only, and measured POR x VOR disagreement is
    ~3.4 km.  Same-family extension for the Sentinel EOF products: POEORB
    (restituted/precise) beats MOEORB (medium), mirroring
    ``analyzers.event_response._source_priority``.
    """
    name = source_name.upper()
    if name.startswith("SWOT_VOR"):
        return 1
    if "MOEORB" in name:
        return 1
    return 0


def _sp3_arc_year(year_of_century: int, sat_id: str | None) -> int:
    """Century for a two-digit SP3 ``b``-token year (YYDDD).

    Resolution order (REVIEW_FINDINGS 6.2.1 rule 4: mission-window pivot,
    never a global Y2K guess -- topex-poseidon arcs carry YY in 92..99 and
    the previous 20YY mapping put 1990s arcs at 2092-2099, outranking every
    2000s arc under the later-arc-start dedup rule):

    1. the satellite's declared mission window
       (``processors.slr_formats.SAT_MISSION_WINDOWS``): pick the century
       whose resolved year falls inside the window (unambiguous -- no
       mission in this dataset spans a century boundary);
    2. no/ambiguous window: the repository-wide static pivot
       (YY >= 57 -> 19YY), the same convention as
       ``benchmarking.normalization._tle_epoch_to_datetime`` and
       ``audit_reference_window_alignment._parse_sp3_year_doy``.
    """
    if sat_id:
        window = mission_window_for_sat(sat_id)
        if window is not None:
            window_start, window_end = window
            candidates = [1900 + year_of_century, 2000 + year_of_century]
            in_window = [
                year
                for year in candidates
                if (window_start is None or year >= window_start.year)
                and (window_end is None or year <= window_end.year)
            ]
            if len(in_window) == 1:
                return in_window[0]
    return 1900 + year_of_century if year_of_century >= 57 else 2000 + year_of_century


def orbit_file_arc_start(source_name: str, sat_id: str | None = None) -> int:
    """Arc-start day number for same-family resolution (later start wins).

    Binding rule (b): among same-family files the arc with the LATER start
    wins -- a later arc determined the shared epoch more recently.  Tokens,
    in order: the SP3 ``b`` arc field (``.b18348.``; YYDDD, two-digit years
    resolved via the satellite's mission window, see ``_sp3_arc_year``),
    the SWOT/OGDR NetCDF ``_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS`` start stamp,
    and the Sentinel EOF ``VYYYYMMDDTHHMMSS`` validity start.  All tokens
    normalize to whole days since 2000-01-01 so the comparison stays
    monotone even across product formats sharing an epoch (jason-3 SP3 x
    OGDR).  Files without a token rank lowest (-1) and fall through to the
    deterministic filename tie-break.
    """
    name = str(source_name).upper()
    epoch_day = datetime(2000, 1, 1)
    match = re.search(r"\.B(\d{5})\.", name)
    if match:
        token = match.group(1)
        year = _sp3_arc_year(int(token[:2]), sat_id)
        start = datetime(year, 1, 1) + timedelta(days=int(token[2:]) - 1)
        return (start - epoch_day).days
    match = re.search(r"_(\d{8}_\d{6})_(\d{8}_\d{6})", name)
    if match:
        start = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        return (start - epoch_day).days
    match = re.search(r"V(\d{8}T\d{6})", name)
    if match:
        token = match.group(1)
        start = datetime.strptime(f"{token[:8]}_{token[9:]}", "%Y%m%d_%H%M%S")
        return (start - epoch_day).days
    return -1


def _dedup_rule_for(dropped: pd.Series, kept: pd.Series) -> str:
    """Which dedup rule dropped this row, relative to its epoch's kept row."""
    if dropped["_origin"] != kept["_origin"]:
        return DEDUP_PARSED_OVER_PUBLISHED
    if dropped["_family"] != kept["_family"]:
        # Family rank 0 kept over rank 1 (rank order guarantees the
        # direction); name the rule after the product families involved.
        if str(kept["_source_name"]).upper().startswith("SWOT"):
            return DEDUP_POR_BEATS_VOR
        return DEDUP_POE_BEATS_MOE
    if dropped["_arc_start"] != kept["_arc_start"]:
        return DEDUP_LATER_ARC_START
    return DEDUP_TIEBREAK


def dedup_orbit_rows(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Resolve same-epoch rows of one satellite's merged orbit evidence.

    ``merged`` must already be sorted or unsorted (this function sorts) and
    carries the helper columns ``_origin`` ('file' | 'published'),
    ``_source_name``, ``_family``, ``_arc_start``.  Returns the deduplicated
    frame (epoch strictly increasing) and per-rule dropped-row counts.

    Precedence within an epoch group (first wins):

    1. rows re-parsed from provider files over published-seed rows (the
       seed is a retention floor, not an authority);
    2. product family: SWOT POR over VOR; EOF POEORB over MOEORB;
    3. later arc start (``b`` field / validity start);
    4. deterministic filename tie-break (same arc re-listed).
    """
    if merged.empty:
        return merged, {rule: 0 for rule in DEDUP_RULES}
    frame = merged.copy()
    frame["_arc_rank"] = -frame["_arc_start"].astype("int64")
    frame = frame.sort_values(
        ["epoch", "_origin_rank", "_family", "_arc_rank", "_source_name"],
        kind="stable",
    )
    dup_mask = frame.duplicated(subset=["epoch"], keep="first")
    counts = {rule: 0 for rule in DEDUP_RULES}
    dropped_frame = frame[dup_mask]
    kept_by_epoch = frame[~dup_mask].set_index("epoch")
    for _, row in dropped_frame.iterrows():
        kept = kept_by_epoch.loc[row["epoch"]]
        counts[_dedup_rule_for(row, kept)] += 1
    deduped = frame[~dup_mask].drop(columns=["_arc_rank"]).reset_index(drop=True)
    return deduped, counts


def _published_nominal_sma_m(seed: pd.DataFrame) -> float | None:
    """Median omega x r-corrected vis-viva sma of the published seed rows.

    This is the documented nominal for the per-satellite
    ``sma_median_off_nominal`` gate: all released orbit products are
    Earth-fixed, so the correction is applied before vis-viva.  The nominal
    is the shipped release's own median (arc-level noise mm-cm), keeping the
    gate sensitive to the hundreds-of-km frame/unit defect class while the
    A1 backfill rows (same mission, same products) stay well inside the
    25 km default tolerance.  Returns None when the seed carries no usable
    velocities (no nominal gate then).
    """
    if seed is None or seed.empty:
        return None
    for column in ("vx", "vy", "vz", "x", "y", "z"):
        if column not in seed.columns:
            return None
    x = pd.to_numeric(seed["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(seed["y"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(seed["z"], errors="coerce").to_numpy(dtype=float)
    vx = pd.to_numeric(seed["vx"], errors="coerce").to_numpy(dtype=float)
    vy = pd.to_numeric(seed["vy"], errors="coerce").to_numpy(dtype=float)
    vz = pd.to_numeric(seed["vz"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
    if not finite.any():
        return None
    # v_inertial = v_ecef + omega x r -- the single shared implementation
    # (physical_qc.ecef_velocity_to_inertial; REVIEW_FINDINGS 6.2.2 rule 2).
    inertial = ecef_velocity_to_inertial(
        np.column_stack([vx, vy, vz]), np.column_stack([x, y, z])
    )
    radius = np.sqrt(x * x + y * y + z * z)
    speed_sq = np.sum(inertial * inertial, axis=1)
    inv_a = 2.0 / radius - speed_sq / EARTH_MU_M3_PER_S2
    positive = finite & (inv_a > 0)
    if not positive.any():
        return None
    return float(np.median(1.0 / inv_a[positive]))


def _orbit_evidence_frames(sat_id: str, raw_root: Path) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Collect the P3 evidence-floor frames for one satellite.

    Sources (in dedup precedence order): provider files re-parsed through
    ``load_orbit_file`` -- bulk precise-orbit products under
    ``raw_root/<sat>/precise_orbit`` (the bulk-download layout) UNION the A1
    backfill manifest files (``data/interim/backfill_manifest.csv``, the
    migrated window-band evidence) -- then the published dataset parquet as
    the retention seed.  ``PhysicalQCViolation`` from any file re-raises
    (blocking assertion, never a silent drop); other parse errors are
    counted and returned in the ledger list instead of being swallowed.
    """
    from analyzers.event_response import load_orbit_file
    from alignment.audit_reference_window_alignment import infer_orbit_product_spans

    ledger: dict[str, Any] = {
        "sat_id": sat_id,
        "n_bulk_files": 0,
        "n_backfill_files": 0,
        "published_seed": False,
        "parse_failures": [],
    }
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []

    bulk = infer_orbit_product_spans([raw_root / sat_id / "pod", raw_root / sat_id / "precise_orbit"])
    bulk = [row for row in bulk if row.get("source_type") == "precise_orbit"]
    paths.extend(Path(row["path"]) for row in bulk)
    ledger["n_bulk_files"] = len(bulk)

    if BACKFILL_MANIFEST.exists():
        manifest = pd.read_csv(BACKFILL_MANIFEST)
        # The manifest is window-scoped; satellite membership comes from the
        # resolved-path layout <root>/<sat>/<subdir>/<file>.
        sat_mask = manifest["resolved_path"].astype(str).str.contains(f"/{sat_id}/", regex=False)
        sat_files = manifest.loc[sat_mask, "resolved_path"].drop_duplicates().tolist()
        backfill = [Path(p) for p in sat_files if Path(p) not in paths]
        paths.extend(backfill)
        ledger["n_backfill_files"] = len(backfill)

    for path in paths:
        try:
            frame, _rotating = load_orbit_file(path)
        except PhysicalQCViolation:
            # Ruling 14: a physics-identity violation is a blocking defect;
            # it must abort the stage, never silently drop the file.
            raise
        except Exception as exc:
            ledger["parse_failures"].append(
                {"sat_id": sat_id, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        if frame is None or frame.empty:
            continue
        frame = frame.assign(
            source_product=path.name,
            _origin=_ORIGIN_FILE,
            _origin_rank=0,
            _source_name=path.name,
            _family=orbit_file_family_rank(path.name),
            _arc_start=orbit_file_arc_start(path.name, sat_id=sat_id),
        )
        frames.append(frame)

    seed_path = PUBLISHED_ROOT / "mission_reported" / "evidence" / "orbit" / f"{sat_id}.parquet"
    if seed_path.exists():
        seed = pd.read_parquet(seed_path).rename(columns=_ORBIT_COMPACT_COLUMNS)
        if not seed.empty:
            seed = seed.assign(
                _origin=_ORIGIN_PUBLISHED,
                _origin_rank=1,
                _source_name=seed_path.name,
                _family=1,
                _arc_start=-1,
            )
            frames.append(seed)
            ledger["published_seed"] = True
    return frames, ledger


def stage_evidence_orbit(
    output: Path,
    raw_root: Path | None = None,
    staging_root: Path | None = None,
    sats: list[str] | None = None,
) -> None:
    """Normalized precise-orbit snapshots per target -- the P3 evidence floor.

    One rule for every satellite (the old three-satellite pod.csv special
    case is deleted): orbit evidence = parse(bulk precise-orbit products)
    UNION parse(A1 backfill manifest files for windows whose band needs
    them) UNION the published snapshot rows (retention seed).  Same-epoch
    rows are resolved by the dedup rules (SWOT POR over VOR, later arc
    start over earlier, re-parsed file over seed); kept/dropped counts land
    in the staging QC ledger CSV.  The merged frame must pass
    ``processors.physical_qc.assert_orbit_records`` (radius band,
    omega x r-corrected vis-viva, strictly increasing epochs, and the
    nominal-a gate against the published seed's median sma) before it is
    written.

    ``staging_root`` (CLI ``--staging-root``) redirects the parquet + ledger
    writes away from the release tree (e.g. ``results/staging_evidence``)
    so evidence-floor repair can be staged without touching ``dataset/``;
    unset, the stage writes into ``<output>/mission_reported/evidence/orbit``
    as before.
    ``raw_root`` overrides the bulk raw workspace root (default
    ``data/raw/reference_validation``).  ``sats`` restricts the run to a
    satellite subset (CLI ``--orbit-sats``) -- staging iteration aid for
    working around blocking per-satellite parse defects.
    """
    dest_root = Path(staging_root) if staging_root else output
    dest = ensure_directory(dest_root / "mission_reported" / "evidence" / "orbit")
    raw_root = Path(raw_root) if raw_root else RAW
    targets = list(sats) if sats else list(TARGETS)
    ledger_rows: list[dict[str, Any]] = []
    parse_failure_rows: list[dict[str, Any]] = []

    for sat_id in targets:
        frames, ledger = _orbit_evidence_frames(sat_id, raw_root)
        parse_failures = list(ledger.pop("parse_failures"))
        parse_failure_rows.extend(parse_failures)
        if not frames:
            ledger.update({
                "n_records": 0,
                "dup_epoch_rows_before": 0,
                "dup_epoch_rows_after": 0,
                "assertions": "no_orbit_product_ships",
                "sma_median_m": "",
            })
            ledger_rows.append(ledger)
            print(f"  orbit {sat_id}: no evidence sources (skipped)", flush=True)
            continue

        merged = pd.concat(frames, ignore_index=True)
        merged["epoch"] = pd.to_datetime(merged["epoch"], utc=True)
        merged["sat_id"] = sat_id
        dup_before = int(merged["epoch"].duplicated().sum())

        deduped, drop_counts = dedup_orbit_rows(merged)
        dup_after = int(deduped["epoch"].duplicated().sum())

        nominal = None
        seed_frames = [f for f in frames if f["_origin"].iloc[0] == _ORIGIN_PUBLISHED]
        if seed_frames:
            nominal = _published_nominal_sma_m(seed_frames[0])

        qc = assert_orbit_records(
            deduped,
            sat_id,
            source=f"{sat_id}.parquet",
            rotating_frame=_ORBIT_MERGED_ROTATING,
            expected_sma_m=nominal,
        )
        ledger.update({
            "n_records": int(len(deduped)),
            "dup_epoch_rows_before": dup_before,
            "dup_epoch_rows_after": dup_after,
            **{f"dropped_{rule}": drop_counts[rule] for rule in DEDUP_RULES},
            "expected_sma_m": nominal,
            "sma_median_m": qc["sma_median_m"],
            "epoch_start_utc": qc["epoch_start_utc"],
            "epoch_end_utc": qc["epoch_end_utc"],
            "assertions": ",".join(qc["assertions"]),
            "parse_failure_count": len(parse_failures),
        })
        ledger_rows.append(ledger)

        snapshot = deduped.drop(columns=[c for c in deduped.columns if c.startswith("_")])
        snapshot = snapshot.rename(columns=_ORBIT_CANONICAL_COLUMNS)
        _write_parquet(snapshot, dest / f"{sat_id}.parquet")
        dropped_total = sum(drop_counts.values())
        print(
            f"  orbit {sat_id}: {len(snapshot)} rows "
            f"(bulk={ledger['n_bulk_files']} backfill={ledger['n_backfill_files']} "
            f"seed={ledger['published_seed']}; dup {dup_before}->{dup_after}, "
            f"dropped={dropped_total}; parse_failures={len(parse_failures)})",
            flush=True,
        )

    ledger_columns = [
        "sat_id", "n_bulk_files", "n_backfill_files", "published_seed",
        "n_records", "dup_epoch_rows_before", "dup_epoch_rows_after",
        *[f"dropped_{rule}" for rule in DEDUP_RULES],
        "expected_sma_m", "sma_median_m", "epoch_start_utc", "epoch_end_utc",
        "assertions", "parse_failure_count",
    ]
    pd.DataFrame(ledger_rows, columns=ledger_columns).to_csv(dest_root / "orbit_qc_ledger.csv", index=False)
    failures = pd.DataFrame(
        parse_failure_rows, columns=["sat_id", "path", "error"]
    )
    failures.to_csv(dest_root / "orbit_parse_failures.csv", index=False)
    print(f"  orbit qc ledger: {dest_root / 'orbit_qc_ledger.csv'}", flush=True)


def stage_evidence_slr(output: Path) -> None:
    """Full-mission normalized SLR normal points per target.

    Delegates to the production staging implementation
    ``alignment.stage_evidence_slr_tle.stage_slr_for_sat`` (the legacy body
    here enumerated by extension whitelist and silently dropped files via
    ``except Exception: continue``): content-routed file enumeration
    (``_detect_child_slr_format``), the canonical ``slr_to_dataframe``
    parse with counted corrupt-row rejection, mission-window filtering,
    ``(station_id, epoch)`` dedupe, and the shared record triage (Class-A
    counted row-level rejections, never silent drops).  The per-file parse
    QC ledger (``slr_parse_qc.csv``) and the per-target summary ledger
    (``slr_qc_ledger.csv``) are written to the release root, mirroring the
    orbit stage's QC artifacts.  A hard per-file parse failure is
    SURFACED: reported per target and re-raised as one RuntimeError after
    the loop, so an incomplete SLR evidence tree can never ship quietly.
    """
    from alignment.stage_evidence_slr_tle import (
        QC_COLUMNS,
        StagingSLRError,
        stage_slr_for_sat,
    )

    dest = ensure_directory(output / "mission_reported" / "evidence" / "slr")
    raw_root = RAW / "reference_validation"
    ledgers: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for sat_id in TARGETS:
        try:
            ledger, rows = stage_slr_for_sat(sat_id, raw_root, dest)
        except StagingSLRError as exc:
            qc_rows.extend(exc.qc_rows)
            failures.append(f"slr:{sat_id}:{exc}")
            print(f"  slr {sat_id}: FAILED {exc}", flush=True)
            continue
        ledgers.append(ledger)
        qc_rows.extend(rows)
        print(
            f"  slr {sat_id}: {ledger.get('staged_rows', 0)} rows from "
            f"{ledger.get('files', 0)} files "
            f"(dup_dropped {ledger.get('dup_rows_dropped', 0)}; "
            f"rejected {ledger.get('records_rejected_total', 0)}; "
            f"tof_green={ledger.get('tof_assert_green')})",
            flush=True,
        )
    if qc_rows:
        pd.DataFrame(qc_rows, columns=QC_COLUMNS).to_csv(output / "slr_parse_qc.csv", index=False)
    if ledgers:
        pd.DataFrame(ledgers).to_csv(output / "slr_qc_ledger.csv", index=False)
    if failures:
        raise RuntimeError(f"evidence-slr stage incomplete: {failures}")


def stage_starlink(output: Path) -> None:
    """Full-constellation 107h Starlink operational slice (state columns)."""
    src_dir = RAW / "operational_constellation" / "starlink" / "ephemeris"
    dest = ensure_directory(output / "operational" / "starlink")
    frames = []
    for path in sorted(src_dir.glob("MEME_*.csv")):
        try:
            frame = pd.read_csv(path, usecols=["Satellite_Number", "Satellite_Name", "Timestamp", "X", "Y", "Z", "Vx", "Vy", "Vz"])
        except Exception:
            continue
        frames.append(frame)
    if not frames:
        print("  starlink: no files", flush=True)
        return
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.rename(
        columns={
            "Satellite_Number": "sat_id",
            "Satellite_Name": "satellite_name",
            "Timestamp": "epoch",
            "X": "x_m", "Y": "y_m", "Z": "z_m",
            "Vx": "vx_mps", "Vy": "vy_mps", "Vz": "vz_mps",
        }
    )
    merged["epoch"] = pd.to_datetime(merged["epoch"].str.replace(" UTC", ""), utc=True)
    for column in ("x_m", "y_m", "z_m"):
        merged[column] = merged[column].astype(float) * 1000.0
    for column in ("vx_mps", "vy_mps", "vz_mps"):
        merged[column] = merged[column].astype(float) * 1000.0
    merged = merged.sort_values(["satellite_name", "epoch"])
    # Claim boundary: the operational slice is exactly 2024-11-26T06:00:00Z
    # through 2024-11-30T17:00:00Z (107 h). Raw files carry a few hours of
    # margin on both sides; clip here so the released span matches the docs.
    slice_start = pd.Timestamp("2024-11-26T06:00:00Z")
    slice_end = pd.Timestamp("2024-11-30T17:00:00Z")
    merged = merged[(merged["epoch"] >= slice_start) & (merged["epoch"] <= slice_end)]
    # quality_flag: operator-published predictions of actively deorbiting
    # objects continue below the surface (final review: 5,861 of 43.4M states,
    # 3 objects).  Marked, never dropped -- consumers computing altitude or
    # radius statistics filter quality_flag == 'below_surface'.
    radius_m = np.sqrt(merged["x_m"] ** 2 + merged["y_m"] ** 2 + merged["z_m"] ** 2)
    merged["quality_flag"] = np.where(
        radius_m < EARTH_RADIUS_KM * 1000.0, "below_surface", ""
    )
    _write_parquet(merged, dest / "ephemeris_state.parquet")
    print(f"  starlink: {len(merged)} rows, {merged['satellite_name'].nunique()} satellites", flush=True)

    sample = merged[merged["satellite_name"].isin(sorted(merged["satellite_name"].unique())[:25])]
    _write_parquet(sample, dest / "ephemeris_state_sample25.parquet")

    tle = _load_starlink_tle(slice_start, slice_end)
    if not tle.empty:
        _write_parquet(tle, dest / "tle_elements.parquet")
        print(f"  starlink tle: {len(tle)} epochs, {tle['sat_id'].nunique()} satellites", flush=True)

    # QC audits (overlap_audit.csv, schema_audit.csv) and the Q4
    # distribution/consistency tables are analysis artifacts: they export to
    # experiments/starlink/ (tracked with the code), not into the dataset
    # release tree.  Regenerate the Q4 tables via
    # `python scripts/experiments/run_experiments.py starlink-distributions`.
    starlink_tables = ensure_directory(EXPERIMENTS_ROOT / "starlink")
    _write_starlink_audits(merged, tle, starlink_tables)
    for name in (
        "starlink_tle_distribution_summary.csv",
        "starlink_ephemeris_distribution.csv",
        "starlink_tle_ephemeris_consistency.csv",
        "starlink_tle_ephemeris_consistency_samples.csv",
        "starlink_frame_sensitivity.csv",
    ):
        src = TABLES / name
        if src.exists():
            shutil.copy(src, starlink_tables / name)
            print(f"  starlink: copied {name} -> experiments/starlink/", flush=True)
        else:
            print(f"  starlink: {name} missing (run run_experiments.py starlink-distributions)", flush=True)


def _iso_utc(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _load_starlink_tle(slice_start: pd.Timestamp, slice_end: pd.Timestamp) -> pd.DataFrame:
    """Normalized Starlink TLE records for the released 107 h window."""
    src_dir = RAW / "operational_constellation" / "starlink" / "tle_archive" / "overlaped_data"
    usecols = [
        "Satellite_Name", "Satellite_Number", "Epoch", "Mean_Anomaly",
        "Right_Ascension_of_Node", "Argument_of_Perigee", "Eccentricity",
        "Inclination", "Mean_Motion", "BSTAR_Drag_Term",
    ]
    frames = []
    for path in sorted(src_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path, usecols=usecols)
        except Exception:
            continue
        frames.append(frame)
    if not frames:
        print("  starlink tle: no archive files", flush=True)
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["Satellite_Name"].astype(str).str.startswith("STARLINK")]
    merged = merged.rename(
        columns={
            "Satellite_Number": "sat_id",
            "Satellite_Name": "satellite_name",
            "Epoch": "epoch",
            "Mean_Anomaly": "mean_anomaly_rad",
            "Right_Ascension_of_Node": "raan_rad",
            "Argument_of_Perigee": "argument_of_perigee_rad",
            "Eccentricity": "eccentricity",
            "Inclination": "inclination_rad",
            "Mean_Motion": "mean_motion_rad_per_min",
            "BSTAR_Drag_Term": "bstar_per_earth_radius",
        }
    )
    merged["epoch"] = pd.to_datetime(merged["epoch"], utc=True)
    merged["sat_id"] = pd.to_numeric(merged["sat_id"], errors="coerce")
    merged = merged.dropna(subset=["sat_id"])
    merged["sat_id"] = merged["sat_id"].astype("int64")
    for column in (
        "mean_anomaly_rad", "raan_rad", "argument_of_perigee_rad", "eccentricity",
        "inclination_rad", "mean_motion_rad_per_min", "bstar_per_earth_radius",
    ):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged[(merged["epoch"] >= slice_start) & (merged["epoch"] <= slice_end)]
    merged = merged.drop_duplicates(subset=["sat_id", "epoch"]).sort_values(["sat_id", "epoch"])
    return merged.reset_index(drop=True)


def _source_audit_rows(name: str, df: pd.DataFrame, state_columns: list[str]) -> dict[str, Any]:
    epochs = df["epoch"].sort_values()
    deltas = epochs.groupby(df["sat_id"]).diff().dropna().dt.total_seconds()
    cadence = float(deltas.median()) if len(deltas) else None
    gaps = deltas[deltas > 2 * cadence] if cadence else deltas.iloc[0:0]
    return {
        "source": name,
        "start_epoch": _iso_utc(epochs.iloc[0]),
        "end_epoch": _iso_utc(epochs.iloc[-1]),
        "span_hours": round(float((epochs.iloc[-1] - epochs.iloc[0]).total_seconds() / 3600.0), 4),
        "row_count": int(len(df)),
        "satellite_count": int(df["sat_id"].nunique()),
        "cadence_median_seconds": cadence,
        "gap_count": int(len(gaps)),
        "max_gap_hours": round(float(gaps.max() / 3600.0), 4) if len(gaps) else 0.0,
        "null_state_rows": int(df[state_columns].isna().any(axis=1).sum()),
    }


def _write_starlink_audits(ephemeris: pd.DataFrame, tle: pd.DataFrame, dest: Path) -> None:
    """Audit tables computed from the released slice itself (overlap + schema)."""
    overlap_rows = [
        _source_audit_rows("ephemeris", ephemeris, ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]),
    ]
    if not tle.empty:
        overlap_rows.append(
            _source_audit_rows(
                "tle",
                tle,
                ["mean_motion_rad_per_min", "inclination_rad", "eccentricity", "bstar_per_earth_radius"],
            )
        )
        start = max(ephemeris["epoch"].min(), tle["epoch"].min())
        end = min(ephemeris["epoch"].max(), tle["epoch"].max())
        overlap_rows.append(
            {
                "source": "ephemeris_tle_overlap",
                "start_epoch": _iso_utc(start),
                "end_epoch": _iso_utc(end),
                "span_hours": round(float((end - start).total_seconds() / 3600.0), 4),
                "row_count": "",
                "satellite_count": "",
                "cadence_median_seconds": "",
                "gap_count": "",
                "max_gap_hours": "",
                "null_state_rows": "",
            }
        )
    pd.DataFrame(overlap_rows).to_csv(dest / "overlap_audit.csv", index=False)

    schema_rows = []
    for table, df, required in (
        ("ephemeris_state.parquet", ephemeris, ["sat_id", "satellite_name", "epoch", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]),
        ("tle_elements.parquet", tle, ["sat_id", "satellite_name", "epoch", "mean_motion_rad_per_min", "inclination_rad", "eccentricity", "bstar_per_earth_radius"]),
    ):
        if df.empty:
            continue
        flagged = (
            int((df["quality_flag"] != "").sum()) if "quality_flag" in df.columns else 0
        )
        schema_rows.append(
            {
                "table": table,
                "row_count": int(len(df)),
                "satellite_count": int(df["sat_id"].nunique()),
                "required_fields_present": all(column in df.columns for column in required),
                "epochs_utc": bool(df["epoch"].dt.tz is not None),
                "duplicate_sat_epoch_rows": int(df.duplicated(subset=["sat_id", "epoch"]).sum()),
                "quality_flag_rows": flagged,
                "quality_flags": "ok" if flagged == 0 else f"ok ({flagged} rows flagged; see quality_flag column)",
            }
        )
    pd.DataFrame(schema_rows).to_csv(dest / "schema_audit.csv", index=False)
    print("  starlink audits: overlap_audit.csv, schema_audit.csv", flush=True)


def stage_manifest(output: Path) -> None:
    """Write README.md + manifest.json with per-file sha256 and row counts."""
    entries = []
    for path in sorted(output.rglob("*")):
        # Skip AppleDouble `._*` / .DS_Store sidecars that macOS writes onto
        # external exFAT volumes; they are Finder metadata, not release data.
        # manifest.json excludes only itself (a self-hash can never verify);
        # README files are release content and ARE listed.
        if not path.is_file() or path.name.startswith(".") or path.name == "manifest.json":
            continue
        record: dict[str, Any] = {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".csv":
            record["rows"] = int(sum(1 for _ in path.open()) - 1)
        elif path.suffix == ".parquet":
            # Row count from parquet metadata only; never load the payload.
            record["rows"] = int(pq.read_metadata(path).num_rows)
        entries.append(record)
    manifest = {
        "dataset": "MAD-LEO",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {len(entries)} files", flush=True)


STAGES = {
    "annotations": stage_annotations,
    "validation": stage_validation,
    "evidence-tle": stage_evidence_tle,
    "evidence-orbit": stage_evidence_orbit,
    "evidence-slr": stage_evidence_slr,
    "starlink": stage_starlink,
    "manifest": stage_manifest,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--staging-root", default=None,
                        help="Stage the orbit evidence floor (parquets + QC ledger) under this "
                             "root instead of the release tree, without touching dataset/ "
                             "(e.g. results/staging_evidence). Default off: release behavior.")
    parser.add_argument("--orbit-raw-root", default=None,
                        help="Bulk precise-orbit raw root override for the orbit stage "
                             "(default: data/raw/reference_validation)")
    parser.add_argument("--orbit-sats", default=None,
                        help="Comma-separated satellite subset for the orbit stage "
                             "(default: all targets)")
    args = parser.parse_args()
    output = ensure_directory(resolve_repo_path(args.output_dir))
    staging_root = resolve_repo_path(args.staging_root) if args.staging_root else None
    orbit_raw_root = resolve_repo_path(args.orbit_raw_root) if args.orbit_raw_root else None
    orbit_sats = [sat.strip() for sat in args.orbit_sats.split(",") if sat.strip()] if args.orbit_sats else None
    for name, func in STAGES.items():
        if args.stage in ("all", name):
            print(f"[stage] {name}", flush=True)
            if name == "evidence-orbit":
                func(output, raw_root=orbit_raw_root, staging_root=staging_root, sats=orbit_sats)
            else:
                func(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
