"""Pre-build the regenerated SLR and TLE evidence parquets into staging.

Friday's full regeneration (T11) must not run the heaviest raw parsing on
its critical path.  This module re-parses the raw provider archive
(``data/raw/reference_validation``) with the
production content-routed parsers and writes the per-satellite evidence
snapshots plus their QC artifacts into ``results/staging_evidence`` so the
release stage can pick them up without re-reading ~30M raw SLR lines.

Outputs (defaults, all under ``results/staging_evidence``):

- ``slr/<sat_id>.parquet``   zstd parquet, dataset/docs/metadata.md SLR
  evidence column set plus the triage ``source_zero_fields`` marker,
  deduped on ``(station_id, epoch)``.
- ``tle/<sat_id>.parquet``   zstd parquet, dataset/mission_reported/evidence/tle
  schema
  (``mean_motion_rad_per_min`` / ``inclination_rad`` /
  ``bstar_per_earth_radius`` rename), deduped by ``epoch``.
- ``slr_parse_qc.csv``       per-file parse ledger (L4 QC artifact):
  detected content family, records parsed/rejected with counted reject
  reasons, per-file physical-QC assertion hits, throughput.
- ``slr_tle_reconciliation.csv``  staged vs shipped row counts with the
  old-list/new-list file enumeration counts that explain the deltas.

Fidelity contract (what makes a staged parquet equivalent to what T11
would build):

- SLR enumeration is CONTENT-based (``benchmarking.normalization.
  _detect_child_slr_format``, the same helper ``load_slr_inputs`` uses) --
  the legacy extension whitelist left the extension-less date-suffixed
  MERIT-II files and the ``.frd`` CRD files invisible, which is the
  jason-1/jason-2 row-count explosion being repaired.
- Parsing goes through ``processors.slr_processor.slr_to_dataframe`` with
  the satellite's mission window
  (``processors.slr_formats.mission_window_for_sat``) -- the same
  production path ``analyzers.event_response.load_slr_records`` uses per
  file.
- Merge, dedupe on ``(station_id, epoch)``, sort, and the record triage
  (shared ``processors.physical_qc.assert_slr_records`` gate +
  ``source_zero_fields`` markers) mirror ``load_slr_records`` exactly;
  files are visited in sorted-name order, so first-occurrence-wins
  deduplication is deterministic.  A Class-A physics violation is a
  counted ROW-level rejection (Ruling-17): the offending rows are marked
  ``qc_rejected`` and counted (``tof_assert_green=False``,
  ``physical_qc_hits`` in the reconciliation table) but stay staged --
  only a gate that raises without identifiable rows (no report mode)
  blocks that satellite's parquet.
- TLE goes through the canonical ``load_tle_history`` loader and the
  same release rename as ``alignment.release_export.stage_evidence_tle``.

The module is deliberately NOT registered in the ``process.py``
dispatcher: it is a one-shot staging aid, run directly, e.g.::

    python3 scripts/data_pipeline/alignment/stage_evidence_slr_tle.py
    python3 scripts/data_pipeline/alignment/stage_evidence_slr_tle.py \
        --sats jason-1 --source slr --compaction-rows 4000000
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Flat-package bootstrap (same as the other alignment modules): make the
# data_pipeline packages importable when this module is run as a script.
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from analyzers.event_response import (  # noqa: E402
    _apply_slr_record_triage,
    load_tle_history,
)
from benchmarking.config import REPO_ROOT, ensure_directory  # noqa: E402
from benchmarking.normalization import _detect_child_slr_format  # noqa: E402
from processors.physical_qc import (  # noqa: E402
    PhysicalQCViolation,
    assert_slr_records,
    slr_qc_status,
)
from processors.slr_formats import mission_window_for_sat  # noqa: E402
from processors.slr_processor import slr_to_dataframe  # noqa: E402

# The 11 reference targets, same order/list as the release export.
_TARGETS = [
    "sentinel-3a", "sentinel-3b", "jason-3", "sentinel-6a", "cryosat-2", "saral",
    "jason-1", "jason-2", "topex-poseidon", "hy-2a", "swot",
]

# Raw provider archive backing the shipped dataset (read-only), materialized
# inside this repo at data/raw/reference_validation; override with
# $MADLEO_RAW_ARCHIVE.
DEFAULT_RAW_ROOT = Path(
    os.environ.get(
        "MADLEO_RAW_ARCHIVE",
        str(REPO_ROOT / "data" / "raw" / "reference_validation"),
    )
)
STAGING_ROOT = REPO_ROOT / "results" / "staging_evidence"
SHIPPED_ROOT = REPO_ROOT / "dataset"

# The retired extension whitelist (documented in load_slr_inputs' docstring)
# -- kept only to enumerate old-list vs new-list file counts for the
# reconciliation table; content detection decides what is parsed.
OLD_SLR_EXTENSIONS = {".npt", ".np", ".np2", ".crd"}

# dataset/docs/metadata.md evidence/slr column contract (union schema; the
# shipped per-satellite parquets vary slightly -- topex-poseidon lacks
# target_name/window_length -- the staged files use one consistent layout)
# plus the triage markers attached by _apply_slr_record_triage:
# source_zero_fields (Class B structural zeros) and qc_status (Class A
# row-level rejections -- marked, never dropped; see physical_qc.slr_qc_status).
SLR_EVIDENCE_COLUMNS = [
    "record_type", "epoch", "time_of_flight_s", "range_m", "sigma_m",
    "num_returns", "window_length", "station_id", "target_id", "target_name",
    "source_zero_fields", "qc_status",
]
SLR_NUMERIC_COLUMNS = {
    "time_of_flight_s", "range_m", "sigma_m", "num_returns", "window_length",
}

# dataset/mission_reported/evidence/tle column contract (post release rename).
TLE_EVIDENCE_COLUMNS = [
    "sat_id", "epoch", "mean_motion_rad_per_min", "inclination_rad",
    "eccentricity", "bstar_per_earth_radius",
]

# Compaction threshold: accumulated not-yet-deduplicated rows before an
# intermediate (station_id, epoch) dedupe pass.  Bounds peak memory to
# roughly final_size + compaction_rows for the ~9-11M-row satellites.
DEFAULT_COMPACTION_ROWS = 6_000_000


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write one zstd parquet (same writer settings as release_export)."""
    ensure_directory(path.parent)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def _join_reasons(reasons: Dict[str, int]) -> str:
    """Stable 'reason:count;...' rendering of a counted-rejection dict."""
    return ";".join(f"{name}:{count}" for name, count in sorted(reasons.items()))


def enumerate_slr_files(slr_dir: Path) -> Tuple[List[Path], Dict[str, int]]:
    """Content-based SLR enumeration of one satellite directory.

    Every child is classified by CONTENT via
    ``benchmarking.normalization._detect_child_slr_format`` (the
    ``load_slr_inputs`` helper); ``unknown`` children (subdirectories,
    binaries, non-SLR text) and ``html_error_page`` children (saved
    provider error pages) are excluded from the parse list -- same
    skip set as ``load_slr_inputs``.  Returns ``(parseable files in
    sorted-name order, format -> file count)``; the format census covers
    EVERY child (excluded ones included), so skipped content stays
    visible in the reconciliation table.
    """
    formats: Dict[str, int] = {}
    files: List[Path] = []
    if not slr_dir.is_dir():
        return files, formats
    for child in sorted(slr_dir.glob("*")):
        fmt = _detect_child_slr_format(child)
        formats[fmt] = formats.get(fmt, 0) + 1
        if fmt not in ("unknown", "html_error_page"):
            files.append(child)
    return files, formats


def _compact(frames: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Concat and dedupe accumulated per-file frames on (station_id, epoch).

    Same key and keep-first semantics as ``load_slr_records``; the file
    frames arrive in sorted-name order, so intermediate compaction cannot
    change which row wins a duplicate group.
    """
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    keys = [key for key in ("station_id", "epoch") if key in merged.columns]
    if keys:
        merged = merged.drop_duplicates(subset=keys)
    return merged


def _parse_slr_file(path: Path, window) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Parse one SLR file via the production path, with a decode fallback.

    The production loaders now read with ``errors="replace"``
    (``slr_formats.read_slr_text``): raw files whose CRD ``60``
    compatibility records carry binary junk in free-text fields (measured:
    18 non-ASCII bytes inside ``Software Versions: ...`` strings,
    cryosat-2 2012 files) parse directly, value-identically (every emitted
    record field is fixed-width pure ASCII), and the validation dict
    carries ``non_utf8_sanitized_copy`` -- surfaced into the QC note by
    the caller.  The decode-replace scratch-copy fallback below is kept
    as belt-and-braces for any future strict-decode path and no longer
    triggers on the current archive.
    """
    try:
        df, validation = slr_to_dataframe(str(path), mission_window=window)
        return df, validation, ""
    except UnicodeDecodeError:
        # Scratch copy lives OUTSIDE the read-only archive (system temp);
        # the sibling-name form keeps any extension-based logging honest.
        with tempfile.NamedTemporaryFile(
            "w", suffix=path.suffix or ".slr", prefix="mad-leo-utf8san-",
            delete=False, encoding="utf-8",
        ) as scratch:
            scratch.write(path.read_text(encoding="utf-8", errors="replace"))
            scratch_path = Path(scratch.name)
        try:
            df, validation = slr_to_dataframe(str(scratch_path), mission_window=window)
        finally:
            scratch_path.unlink(missing_ok=True)
        return df, validation, "non_utf8_sanitized_copy"


def _triage_with_counted_gate(
    merged: pd.DataFrame, sat_id: str, window,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run the production record triage; surface (never hide) gate hits.

    Ruling-17 counted row-level rejection: inside
    ``_apply_slr_record_triage`` a Class-A physics violation (TOF/range
    identity, range band, epoch presence, mission span) MARKS the
    offending rows ``qc_rejected`` (rows are never dropped) and counts
    them into ``attrs["slr_qc_rejections"]`` -- the production call no
    longer raises for real physics hits.  The counts are surfaced here
    as the qc_hits dict (assertion -> offending row count) for the
    staging ledger (``tof_assert_green`` / ``physical_qc_hits``).  The 3
    near-zero-range records in the topex-poseidon legacy NPT files never
    reach this layer at all: the parse layer
    (``processors.slr_processor.slr_to_dataframe``) rejects a one-way
    range below the physical slant-range floor as a corrupt parse
    (``range_below_band_corrupt``) -- the rows are DROPPED with a counted
    reason in the validation ledger (``corrupt_records_rejected`` /
    ``corrupt_reject_reasons``), so they are NOT present in the staged or
    shipped parquet.
    """
    out = _apply_slr_record_triage(merged, sat_id=sat_id, mission_span=window)
    return out, dict(out.attrs.get("slr_qc_rejections") or {})


def stage_slr_for_sat(
    sat_id: str,
    raw_root: Path,
    dest_dir: Path,
    *,
    compaction_rows: int = DEFAULT_COMPACTION_ROWS,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Stage one satellite's SLR evidence parquet; return (ledger, per-file QC).

    Ledger keys (reconciliation-facing): ``sat_id``, ``source='slr'``,
    ``files``/``files_old_list``/``files_new_list``/``formats``,
    ``staged_rows``, ``staged_normal_point``/``staged_full_rate``,
    ``dup_rows_dropped``, ``records_rejected_total``,
    ``outside_window_rows``, ``tof_assert_green``, marker counts,
    ``staged_epoch_min_utc``/``staged_epoch_max_utc``, ``window_start``/
    ``window_end``, ``duration_s``, ``rows_per_second``, ``note``.

    Raises only for unrecoverable per-file failures (an SLR file whose
    content matches no supported format is a hard file-level failure in
    the production path); a Class-A physics violation is a counted
    row-level rejection (rows marked ``qc_rejected``, counts in
    ``physical_qc_hits`` / ``tof_assert_green=False``) and a gate that
    raises without identifiable rows records ``tof_assert_green=False``
    with NO parquet written.
    """
    slr_dir = Path(raw_root) / sat_id / "slr"
    window = mission_window_for_sat(sat_id)
    started = time.perf_counter()

    # Physical-QC span gate routing: processors.physical_qc.assert_slr_records
    # now guards open bounds itself (SLR-chain fix round 2: None bounds are
    # skipped, pandas-2.2 "to_datetime(None) -> None / epochs > None"
    # TypeError fixed), so open windows COULD go straight to the gate.
    # The local one-sided routing below is kept as defense-in-depth: open
    # windows pass mission_span=None to the gate (identity + band
    # assertions still run) and the span check is done locally with
    # identical semantics.
    closed_window = (
        window is not None and window[0] is not None and window[1] is not None
    )
    span_for_gate = window if closed_window else None

    files, formats = enumerate_slr_files(slr_dir)
    old_list = [c for c in sorted(slr_dir.glob("*")) if c.suffix.lower() in OLD_SLR_EXTENSIONS] \
        if slr_dir.is_dir() else []

    frames: List[pd.DataFrame] = []
    pending_rows = 0
    parsed_rows = 0
    rejected_total = 0
    qc_rows: List[Dict[str, Any]] = []

    for child in files:
        fmt = _detect_child_slr_format(child)
        size = child.stat().st_size
        file_start = time.perf_counter()
        try:
            df, validation, decode_note = _parse_slr_file(child, window)
        except Exception as exc:  # hard file-level failure: surface, do not skip
            qc_rows.append({
                "sat_id": sat_id, "file": child.name, "format": fmt,
                "size_bytes": size, "records_parsed": 0, "records_rejected": 0,
                "reject_reasons": "", "window_unavailable": False,
                "assertion_hit": f"parse_error:{type(exc).__name__}",
                "parse_seconds": round(time.perf_counter() - file_start, 4),
                "rows_per_second": 0.0, "note": str(exc).splitlines()[0][:200],
            })
            raise StagingSLRError(sat_id, exc, qc_rows) from exc
        seconds = time.perf_counter() - file_start

        reasons: Dict[str, int] = {}
        reasons.update(validation.get("merit_reject_reasons") or {})
        reasons.update(validation.get("npt_reject_reasons") or {})
        reasons.update(validation.get("crd_reject_reasons") or {})
        reasons.update(validation.get("corrupt_reject_reasons") or {})
        rejected = int(validation.get("merit_records_rejected") or 0) \
            + int(validation.get("npt_records_rejected") or 0) \
            + int(validation.get("crd_records_rejected") or 0) \
            + int(validation.get("corrupt_records_rejected") or 0)
        rejected_total += rejected
        window_unavailable = bool(validation.get("window_unavailable", False))
        # Production loader now sanitizes non-UTF-8 bytes itself
        # (slr_formats.read_slr_text); the scratch-copy fallback in
        # _parse_slr_file stays as a belt-and-braces path only.
        decode_note = decode_note or (
            "non_utf8_sanitized_copy"
            if validation.get("non_utf8_sanitized_copy") else ""
        )

        # Per-file Class-A physics probe (counted hit; the merged-frame
        # triage below is the gate that decides whether staging proceeds).
        assertion_hit = ""
        if not df.empty:
            try:
                assert_slr_records(df, sat_id, mission_span=span_for_gate, source=child.name)
            except PhysicalQCViolation as exc:
                assertion_hit = str(exc.assertion)

        qc_rows.append({
            "sat_id": sat_id, "file": child.name, "format": fmt,
            "size_bytes": size, "records_parsed": int(len(df)),
            "records_rejected": rejected,
            "reject_reasons": _join_reasons(reasons),
            "window_unavailable": window_unavailable,
            "assertion_hit": assertion_hit,
            "parse_seconds": round(seconds, 4),
            "rows_per_second": round(len(df) / seconds, 1) if seconds > 0 else 0.0,
            "note": decode_note,
        })

        if not df.empty:
            frame = df.loc[:, [c for c in SLR_EVIDENCE_COLUMNS if c in df.columns]].copy()
            frame["epoch"] = pd.to_datetime(frame["epoch"], utc=True)
            frames.append(frame)
            parsed_rows += len(frame)
            pending_rows += len(frame)
            if pending_rows >= compaction_rows:
                compacted = _compact(frames)
                frames = [compacted] if compacted is not None else []
                pending_rows = len(frames[0]) if frames else 0

    # Unknown-content children (excluded above) still land in the QC ledger
    # so the file census is exhaustive.
    known = {child.name for child in files}
    for child in sorted(slr_dir.glob("*")) if slr_dir.is_dir() else []:
        if child.name not in known:
            qc_rows.append({
                "sat_id": sat_id, "file": child.name,
                "format": _detect_child_slr_format(child),
                "size_bytes": child.stat().st_size, "records_parsed": 0,
                "records_rejected": 0, "reject_reasons": "",
                "window_unavailable": False, "assertion_hit": "",
                "parse_seconds": 0.0, "rows_per_second": 0.0,
                "note": "non_slr_content_skipped",
            })
    qc_rows.sort(key=lambda row: row["file"])

    ledger: Dict[str, Any] = {
        "sat_id": sat_id,
        "source": "slr",
        "files": len(files),
        "files_old_list": len(old_list),
        "files_new_list": len(files),
        "formats": _join_reasons(formats),
        "window_start": window[0].isoformat() if window and window[0] else "",
        "window_end": window[1].isoformat() if window and window[1] else "",
        "records_rejected_total": rejected_total,
        "note": "",
    }

    merged = _compact(frames)
    if merged is None or merged.empty:
        ledger.update({
            "staged_rows": 0, "staged_normal_point": 0, "staged_full_rate": 0,
            "dup_rows_dropped": 0, "outside_window_rows": 0,
            "tof_assert_green": True,
            "staged_epoch_min_utc": "", "staged_epoch_max_utc": "",
            "duration_s": round(time.perf_counter() - started, 2),
            "rows_per_second": 0.0,
            "note": "no_slr_content",
        })
        return ledger, qc_rows

    merged["epoch"] = pd.to_datetime(merged["epoch"], utc=True)
    keys = [key for key in ("station_id", "epoch") if key in merged.columns]
    if keys:
        merged = merged.drop_duplicates(subset=keys)
    merged = merged.sort_values("epoch").reset_index(drop=True)
    dup_dropped = parsed_rows - len(merged)

    # Record triage: shared Class-A assertions (counted row-level
    # rejections -- offending rows marked qc_rejected) + source-semantics
    # markers -- identical to the canonical loader's post-merge step.
    # The except branch is only reachable for a gate that raises without
    # identifiable rows (no report mode).
    try:
        merged, qc_hits = _triage_with_counted_gate(merged, sat_id, span_for_gate)
    except PhysicalQCViolation as exc:
        ledger.update({
            "staged_rows": len(merged), "dup_rows_dropped": dup_dropped,
            "tof_assert_green": False,
            "note": f"physical_qc_violation[{exc.assertion}] row={exc.row_index} value={exc.value}",
            "duration_s": round(time.perf_counter() - started, 2),
        })
        return ledger, qc_rows

    alerts = dict(merged.attrs.get("slr_qc_alerts") or {})
    epochs = pd.to_datetime(merged["epoch"], utc=True)
    outside = 0
    if window is not None:
        lo = pd.to_datetime(window[0], utc=True) if window[0] else None
        hi = pd.to_datetime(window[1], utc=True) if window[1] else None
        mask = pd.Series(False, index=merged.index)
        if lo is not None:
            mask = mask | (epochs < lo)
        if hi is not None:
            mask = mask | (epochs > hi)
        outside = int(mask.sum())

    out = merged.copy()
    # qc_status: the triage's qc_rejected boolean, resolved into the shipped
    # status vocabulary (cross_target / range_implausible / qc_rejected / ok;
    # physical_qc.slr_qc_status is the single implementation).
    out["qc_status"] = slr_qc_status(
        out, qc_rejected=out["qc_rejected"] if "qc_rejected" in out.columns else None
    )
    for column in SLR_EVIDENCE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan if column in SLR_NUMERIC_COLUMNS else None
    out = out.loc[:, SLR_EVIDENCE_COLUMNS]
    _write_parquet(out, Path(dest_dir) / f"{sat_id}.parquet")

    duration = time.perf_counter() - started
    ledger.update({
        "staged_rows": int(len(out)),
        "staged_normal_point": int((out["record_type"] == "normal_point").sum()),
        "staged_full_rate": int((out["record_type"] == "full_rate").sum()),
        "dup_rows_dropped": int(dup_dropped),
        "outside_window_rows": outside,
        "tof_assert_green": not qc_hits,
        # e.g. "slr_range_in_band:3" -- band hits on rows the shipped
        # release already carries (kept; see _triage_with_counted_gate).
        "physical_qc_hits": _join_reasons(qc_hits),
        "source_zero_sigma": alerts.get("source_zero_sigma", ""),
        "source_zero_num_returns": alerts.get("source_zero_num_returns", ""),
        "sigma_above_1m": alerts.get("sigma_above_1m", ""),
        "staged_epoch_min_utc": _iso_z(epochs.min()),
        "staged_epoch_max_utc": _iso_z(epochs.max()),
        "duration_s": round(duration, 2),
        "rows_per_second": round(parsed_rows / duration, 1) if duration > 0 else 0.0,
    })
    return ledger, qc_rows


def _iso_z(value: pd.Timestamp) -> str:
    """Compact ISO-8601 UTC rendering with trailing Z."""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return ""
    if ts.tz is not None:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def stage_tle_for_sat(sat_id: str, raw_root: Path, dest_dir: Path) -> Dict[str, Any]:
    """Stage one satellite's TLE evidence parquet via the canonical loader."""
    tle_dir = Path(raw_root) / sat_id / "tle"
    started = time.perf_counter()
    tle_files = (
        [p for p in sorted(tle_dir.iterdir())
         if p.suffix.lower() in {".tle", ".txt"} and not p.name.startswith(".")]
        if tle_dir.is_dir() else []
    )
    ledger: Dict[str, Any] = {
        "sat_id": sat_id, "source": "tle", "files": len(tle_files),
        "files_old_list": "", "files_new_list": "", "formats": "tle_line_pairs",
        "tof_assert_green": "",  # not applicable: TLE has no SLR physics gate
        "note": "",
    }
    tle = load_tle_history(tle_dir)
    if tle.empty:
        ledger.update({
            "staged_rows": 0, "tof_assert_green": True,
            "duration_s": round(time.perf_counter() - started, 2),
            "note": "no_tle_content",
        })
        return ledger
    tle = tle.copy()
    tle["sat_id"] = sat_id
    # Dataset-facing schema contract (same rename as release_export).
    tle = tle.rename(columns={
        "mean_motion": "mean_motion_rad_per_min",
        "inclination": "inclination_rad",
        "bstar": "bstar_per_earth_radius",
    })
    tle = tle.loc[:, [c for c in TLE_EVIDENCE_COLUMNS if c in tle.columns]]
    _write_parquet(tle, Path(dest_dir) / f"{sat_id}.parquet")
    epochs = pd.to_datetime(tle["epoch"], utc=True)
    ledger.update({
        "staged_rows": int(len(tle)),
        "staged_epoch_min_utc": _iso_z(epochs.min()),
        "staged_epoch_max_utc": _iso_z(epochs.max()),
        "duration_s": round(time.perf_counter() - started, 2),
    })
    return ledger


def build_reconciliation(
    ledgers: List[Dict[str, Any]],
    shipped_root: Path,
) -> pd.DataFrame:
    """Join the staged ledgers against the shipped dataset/mission_reported/evidence parquets."""
    rows: List[Dict[str, Any]] = []
    for ledger in ledgers:
        sat_id = ledger["sat_id"]
        source = ledger["source"]
        shipped_path = Path(shipped_root) / "mission_reported" / "evidence" / source / f"{sat_id}.parquet"
        row = dict(ledger)
        if shipped_path.exists():
            shipped = pd.read_parquet(shipped_path)
            row["shipped_rows"] = int(len(shipped))
            if "record_type" in shipped.columns:
                row["shipped_normal_point"] = int((shipped["record_type"] == "normal_point").sum())
                row["shipped_full_rate"] = int((shipped["record_type"] == "full_rate").sum())
            if "epoch" in shipped.columns:
                epochs = pd.to_datetime(shipped["epoch"], utc=True)
                row["shipped_epoch_min_utc"] = _iso_z(epochs.min())
                row["shipped_epoch_max_utc"] = _iso_z(epochs.max())
        else:
            row["shipped_rows"] = ""
            row["note"] = (row.get("note") or "") + "no_shipped_parquet"
        row["delta_rows"] = (
            int(row["staged_rows"]) - int(row["shipped_rows"])
            if str(row.get("shipped_rows", "")) != "" else ""
        )
        rows.append(row)
    columns = [
        "sat_id", "source", "files", "files_old_list", "files_new_list", "formats",
        "staged_rows", "shipped_rows", "delta_rows",
        "staged_normal_point", "staged_full_rate",
        "shipped_normal_point", "shipped_full_rate",
        "staged_epoch_min_utc", "staged_epoch_max_utc",
        "shipped_epoch_min_utc", "shipped_epoch_max_utc",
        "window_start", "window_end", "outside_window_rows", "tof_assert_green",
        "physical_qc_hits",
        "dup_rows_dropped", "records_rejected_total",
        "source_zero_sigma", "source_zero_num_returns", "sigma_above_1m",
        "duration_s", "rows_per_second", "note",
    ]
    return pd.DataFrame(rows, columns=columns)


QC_COLUMNS = [
    "sat_id", "file", "format", "size_bytes", "records_parsed",
    "records_rejected", "reject_reasons", "window_unavailable",
    "assertion_hit", "parse_seconds", "rows_per_second", "note",
]


class StagingSLRError(RuntimeError):
    """A satellite's SLR staging aborted on a hard per-file failure.

    Carries the per-file QC rows collected before the failure so the L4
    ledger still records what was parsed up to that point.
    """

    def __init__(self, sat_id: str, cause: BaseException, qc_rows: List[Dict[str, Any]]):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.sat_id = sat_id
        self.qc_rows = qc_rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-build regenerated SLR/TLE evidence parquets into staging."
    )
    parser.add_argument(
        "--raw-root", default=str(DEFAULT_RAW_ROOT),
        help="Raw archive root containing <sat_id>/{slr,tle} "
             "(default: data/raw/reference_validation in this repository).",
    )
    parser.add_argument(
        "--staging-root", default=str(STAGING_ROOT),
        help=f"Staging output root (default: {STAGING_ROOT}).",
    )
    parser.add_argument(
        "--shipped-root", default=str(SHIPPED_ROOT),
        help="Shipped dataset root for the reconciliation table "
             "(default: <repo>/dataset; evidence is read from its "
             "mission_reported/ subset; read-only).",
    )
    parser.add_argument(
        "--sats", default=None,
        help="Comma-separated satellite subset (default: all 11 reference targets).",
    )
    parser.add_argument(
        "--source", choices=["slr", "tle", "both"], default="both",
        help="Which evidence family to stage (default: both).",
    )
    parser.add_argument(
        "--compaction-rows", type=int, default=DEFAULT_COMPACTION_ROWS,
        help="Accumulated-row threshold for intermediate dedupe compaction.",
    )
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root)
    staging_root = ensure_directory(Path(args.staging_root))
    if not raw_root.is_dir():
        print(f"error: raw root not found: {raw_root}", file=sys.stderr)
        return 2
    sats = [s.strip() for s in args.sats.split(",")] if args.sats else list(_TARGETS)

    ledgers: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []
    run_start = time.perf_counter()

    for sat_id in sats:
        if args.source in {"slr", "both"}:
            dest = ensure_directory(staging_root / "slr")
            try:
                ledger, rows = stage_slr_for_sat(
                    sat_id, raw_root, dest, compaction_rows=args.compaction_rows
                )
                ledgers.append(ledger)
                qc_rows.extend(rows)
                if str(ledger.get("note", "")).startswith("physical_qc_violation"):
                    # identity / span defect: no parquet written for this sat
                    failures.append(f"slr:{sat_id}:{ledger.get('note', '')}")
                elif ledger.get("physical_qc_hits"):
                    warnings.append(
                        f"slr:{sat_id}: physical_qc {ledger['physical_qc_hits']} "
                        "(rows kept: already shipped)"
                    )
                print(
                    f"  slr {sat_id}: {ledger.get('staged_rows', 0)} rows from "
                    f"{ledger.get('files', 0)} files "
                    f"(old-list {ledger.get('files_old_list', 0)}; "
                    f"dup_dropped {ledger.get('dup_rows_dropped', 0)}; "
                    f"tof_green={ledger.get('tof_assert_green')}; "
                    f"{ledger.get('duration_s', 0)} s)",
                    flush=True,
                )
            except StagingSLRError as exc:
                qc_rows.extend(exc.qc_rows)
                failures.append(f"slr:{sat_id}:{exc}")
                print(f"  slr {sat_id}: FAILED {exc}", flush=True)
            except Exception as exc:
                failures.append(f"slr:{sat_id}:{type(exc).__name__}:{exc}")
                print(f"  slr {sat_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
        if args.source in {"tle", "both"}:
            dest = ensure_directory(staging_root / "tle")
            try:
                ledger = stage_tle_for_sat(sat_id, raw_root, dest)
                ledgers.append(ledger)
                print(
                    f"  tle {sat_id}: {ledger.get('staged_rows', 0)} epochs from "
                    f"{ledger.get('files', 0)} files ({ledger.get('duration_s', 0)} s)",
                    flush=True,
                )
            except Exception as exc:
                failures.append(f"tle:{sat_id}:{type(exc).__name__}:{exc}")
                print(f"  tle {sat_id}: FAILED {type(exc).__name__}: {exc}", flush=True)

    if qc_rows:
        pd.DataFrame(qc_rows, columns=QC_COLUMNS).to_csv(
            staging_root / "slr_parse_qc.csv", index=False
        )
    reconciliation = build_reconciliation(ledgers, Path(args.shipped_root))
    reconciliation.to_csv(staging_root / "slr_tle_reconciliation.csv", index=False)

    total_staged = int(reconciliation["staged_rows"].fillna(0).sum()) if len(reconciliation) else 0
    print(
        f"staged {total_staged} rows total in {time.perf_counter() - run_start:.1f} s; "
        f"{len(failures)} failure(s)",
        flush=True,
    )
    if failures:
        for failure in failures:
            print(f"  FAILURE {failure}", flush=True)
        return 1
    for warning in warnings:
        print(f"  WARNING {warning}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
