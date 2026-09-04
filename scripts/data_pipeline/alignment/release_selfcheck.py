"""Release self-consistency gate for the MAD-LEO public dataset (WP6).

Recomputes per-window evidence coverage (TLE / orbit / SLR) and the derived
confidence tier directly from the shipped evidence artifacts and diffs the
result row-by-row against ``dataset/mission_reported/annotations/*.csv``.
Divergences are
categorized; the gate exits non-zero on the ``unexpected`` class OR when any
declared evidence artifact is missing/unreadable (``artifact_errors``) — the
latter even when the affected windows' shipped statuses already say missing,
so no per-window diff is produced.

D4 coverage semantics (DECIDED, observation-presence, ms contract):

- orbit covered  <=> at least one orbit sample epoch in ``[ws, we]``;
- SLR covered    <=> at least one normal-point epoch in ``[ws - 24h, we + 24h]``;
- TLE covered    <=> an epoch <= ``ws`` AND an epoch >= ``we`` exists (bracket).

All comparisons are inclusive and performed at millisecond precision: the
release schema serializes window bounds as ISO-8601 UTC with millisecond
resolution, while evidence parquets carry nanosecond epochs. Flooring epochs
to milliseconds before comparison is what keeps deterministic anchoring
(e.g. stable windows anchored exactly at the first TLE epoch of a target)
from flipping on sub-millisecond rounding.

Legacy diagnostic columns ``*_file_span_status``: the historical audit
derived coverage from provider *file-name spans* (whole days/months). Parquet
snapshots have no per-file spans, so sample presence IS the span proxy — the
``*_file_span_status`` columns are emitted equal to the presence statuses and
kept only for downstream parity.

sentinel-6a asymmetry (explicit): sentinel-6a ships NO orbit parquet. The
evidence registry routes its orbit source to ``s6a_gnss_coverage_manifest.csv``
(semantics ``gnss_tracking_availability``), generated from the 28 RINEX 3.x
observation files (mixed 3.03/3.04 in the archive) of the legacy raw archive.
Spans are read from the data itself — ``TIME OF FIRST OBS`` /
``TIME OF LAST OBS`` headers, with the 19 files whose headers omit
``TIME OF LAST OBS`` (the GMV RINEX 3.03 writers) falling back to their last
``>`` observation-epoch marker — never from file names.
s6a orbit coverage = manifest file-span overlap with ``[ws, we]`` — the
ORIGINAL span semantics, not observation presence, because no orbit samples
ship for s6a. RINEX header times carry a GNSS-system label (GPS); they are
taken as UTC — the GPS-UTC leap-second offset (<= 18 s) is immaterial at the
30 h window granularity and matches how the historical audit treated these
files. While the manifest has not been generated yet, s6a orbit coverage is
"missing" and covered->missing downgrades are the known
``s6a_manifest_missing`` class (not a gate failure).

Divergence categories (one diff row per window and status field; the tier is
a pure function of the statuses, so it only produces its own row when the
statuses agree but the shipped tier does not — ``tier_inconsistent``):

- ``a1_backfill_pending``      — jason-3 / sentinel-3a / sentinel-3b windows
                                 whose band falls outside the shipped orbit
                                 parquet span (2023+ pod.csv backfill; A1,
                                 fixed in WP7);
- ``s6a_manifest_missing``     — sentinel-6a orbit downgrades observed while
                                 the s6a manifest has not been generated yet;
- ``semantics_flip_orbit``     — other old-covered/new-missing orbit diffs
                                 (span-overlap -> observation-presence, and
                                 s6a windows outside the header spans once
                                 the manifest exists);
- ``semantics_flip_slr``       — old-covered/new-missing SLR diffs (file-day
                                 spans -> normal-point presence +/- 24 h);
- ``unexpected``               — anything else: covered upgrades (shipped
                                 table says missing but samples exist), TLE
                                 downgrades, missing/unreadable evidence
                                 parquets, unreadable s6a manifest, tier
                                 inconsistencies, duplicate or orphan
                                 annotation ids.

Staging: the recomputation diffs/windows/report go to
``results/selfcheck/`` (git-ignored). The two gate-owned release artifacts
— the evidence registry (``evidence_sources.csv``) and the sentinel-6a GNSS
coverage manifest — are written on demand into ``experiments/validation/``
alongside the other published technical-validation tables. The legacy raw
archive is read-only.

Example:
    python scripts/data_pipeline/alignment/release_selfcheck.py --dataset dataset/ \\
        --report results/selfcheck_pre.json
    python scripts/data_pipeline/alignment/release_selfcheck.py --dataset dataset/ \\
        --write-s6a-manifest
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path  # noqa: E402
from benchmarking.stable_windows import coverage_confidence_tier  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "selfcheck"
# Gate-owned release artifacts (evidence registry, s6a GNSS coverage
# manifest) live with the published experiment tables, not in the dataset.
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "experiments" / "validation"
DEFAULT_RINEX_DIR = Path(
    os.environ.get(
        "MADLEO_RAW_ARCHIVE",
        str(REPO_ROOT / "data" / "raw" / "reference_validation"),
    )
) / "sentinel-6a" / "pod"

REGISTRY_NAME = "evidence_sources.csv"
S6A_MANIFEST_NAME = "s6a_gnss_coverage_manifest.csv"
DIFFS_NAME = "selfcheck_diffs.csv"
WINDOWS_NAME = "selfcheck_windows.csv"
REPORT_NAME = "selfcheck_report.json"

SENTINEL_6A = "sentinel-6a"
A1_BACKFILL_SATS = ("jason-3", "sentinel-3a", "sentinel-3b")
SLR_MARGIN = pd.Timedelta(hours=24)
SOURCE_TYPES = ("tle", "slr", "orbit")

SEMANTICS_TLE_EPOCHS = "tle_epochs"
SEMANTICS_SLR_NPS = "slr_normal_points"
SEMANTICS_ORBIT_STATES = "orbit_states"
SEMANTICS_GNSS_TRACKING = "gnss_tracking_availability"

CATEGORY_A1_BACKFILL_PENDING = "a1_backfill_pending"
CATEGORY_S6A_MANIFEST_MISSING = "s6a_manifest_missing"
CATEGORY_SEMANTICS_FLIP_ORBIT = "semantics_flip_orbit"
CATEGORY_SEMANTICS_FLIP_SLR = "semantics_flip_slr"
CATEGORY_UNEXPECTED = "unexpected"
KNOWN_CATEGORIES = (
    CATEGORY_A1_BACKFILL_PENDING,
    CATEGORY_S6A_MANIFEST_MISSING,
    CATEGORY_SEMANTICS_FLIP_ORBIT,
    CATEGORY_SEMANTICS_FLIP_SLR,
)
ALL_CATEGORIES = KNOWN_CATEGORIES + (CATEGORY_UNEXPECTED,)

STATUS_COLUMNS = {source: f"{source}_status" for source in SOURCE_TYPES}

REQUIRED_WINDOW_COLUMNS = [
    "annotation_id",
    "sat_id",
    "window_start_utc",
    "window_end_utc",
    "confidence_tier",
    *STATUS_COLUMNS.values(),
]

DIFF_COLUMNS = [
    "table",
    "annotation_id",
    "sat_id",
    "field",
    "shipped_value",
    "recomputed_value",
    "category",
    "kind",
    "window_start_utc",
    "window_end_utc",
]

RECOMPUTED_WINDOW_COLUMNS = [
    "table",
    "annotation_id",
    "sat_id",
    "batch",
    "window_start_utc",
    "window_end_utc",
    "tle_status",
    "slr_status",
    "orbit_status",
    "tle_file_span_status",
    "slr_file_span_status",
    "orbit_file_span_status",
    "tle_epochs_before",
    "tle_epochs_after",
    "orbit_sample_count",
    "slr_np_count",
    "confidence_tier",
    "missing_sources",
    "aligned",
    "quality_flags",
]


# ---------------------------------------------------------------------------
# small helpers


def _iso_utc(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        return ""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")
    if ts.microsecond:
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _window_bounds(window: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Accept (start, end) tuples, mappings/Series, or attribute objects."""
    if isinstance(window, (tuple, list)) and len(window) == 2:
        return _parse_utc(window[0]), _parse_utc(window[1])
    if isinstance(window, Mapping):
        return _parse_utc(window["window_start_utc"]), _parse_utc(window["window_end_utc"])
    if isinstance(window, pd.Series) and "window_start_utc" in window.index:
        return _parse_utc(window["window_start_utc"]), _parse_utc(window["window_end_utc"])
    start = getattr(window, "window_start_utc", None)
    end = getattr(window, "window_end_utc", None)
    if start is not None and end is not None:
        return _parse_utc(start), _parse_utc(end)
    raise TypeError(f"cannot extract window bounds from {type(window)!r}")


def _normalize_epochs(epochs: Any) -> pd.Series:
    """Sorted UTC epoch series floored to the release's ms precision."""
    if epochs is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    if isinstance(epochs, pd.Series):
        series = epochs.copy()
    elif isinstance(epochs, pd.Index):
        series = pd.Series(epochs)
    else:
        items = list(epochs)
        if not items:
            return pd.Series(dtype="datetime64[ns, UTC]")
        series = pd.Series(items)
    if series.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    # format="ISO8601": released evidence carries both whole-second and
    # millisecond ISO strings; pandas' single-format inference rejects the
    # mix (ms-floor anchor contract still applied below).
    series = pd.to_datetime(series, utc=True, format="ISO8601").dt.floor("ms")
    return series.sort_values().reset_index(drop=True)


# id(series) -> (length, ns array, strong reference). The strong reference
# keeps the cached id valid for the process lifetime, so a garbage-collected
# frame's id can never be reused by a different Series; only the evidence
# store's per-run Series are cached, so the cache stays small while
# per-window recomputation over ~19M orbit epochs stays O(log n).
_NS_CACHE: dict[int, tuple[int, np.ndarray, pd.Series]] = {}


def _ns_array(epochs: Any) -> np.ndarray:
    """Inclusive-comparison ready int64-ns array for an epoch collection."""
    if isinstance(epochs, pd.Series) and not epochs.empty:
        cached = _NS_CACHE.get(id(epochs))
        if cached is not None and cached[0] == len(epochs) and cached[2] is epochs:
            return cached[1]
        normalized = _normalize_epochs(epochs)
        ns = normalized.astype("int64").to_numpy()
        _NS_CACHE[id(epochs)] = (len(epochs), ns, epochs)
        return ns
    normalized = _normalize_epochs(epochs)
    if normalized.empty:
        return np.empty(0, dtype="int64")
    return normalized.astype("int64").to_numpy()


def _presence_count(ns: np.ndarray, lo_ns: int, hi_ns: int) -> int:
    """Count epochs in the inclusive [lo, hi] integer-ns range."""
    left = int(np.searchsorted(ns, lo_ns, side="left"))
    right = int(np.searchsorted(ns, hi_ns, side="right"))
    return max(0, right - left)


# ---------------------------------------------------------------------------
# D4 coverage core


def coverage_from_parquet(
    orbit_epochs: Any,
    tle_epochs: Any,
    slr_epochs: Any,
    window: Any,
) -> dict:
    """D4 observation-presence coverage for one window from parquet epochs.

    Semantics (decided, inclusive bounds, ms precision):

    - ``orbit_status``: covered  <=> >= 1 orbit sample in [ws, we];
    - ``slr_status``:   covered  <=> >= 1 SLR normal point in [ws-24h, we+24h];
    - ``tle_status``:   covered  <=> an epoch <= ws AND an epoch >= we exists.

    ``*_file_span_status`` are legacy diagnostics: parquet snapshots carry no
    per-file spans, so sample presence IS the span proxy and the columns are
    emitted equal to the presence statuses. Documented asymmetry:
    sentinel-6a orbit coverage is NOT computed here — it is routed through
    the RINEX manifest span overlap in ``run_selfcheck`` (pass ``None`` as
    ``orbit_epochs`` and override, or use ``manifest_span_coverage``).

    ``window`` accepts a ``(start, end)`` tuple or any object exposing
    ``window_start_utc`` / ``window_end_utc``. Epoch arguments accept
    iterables of parseable timestamps or tz-aware UTC Series (the evidence
    store's pre-normalized Series are converted once and cached).
    """
    start, end = _window_bounds(window)
    orbit_ns = _ns_array(orbit_epochs)
    tle_ns = _ns_array(tle_epochs)
    slr_ns = _ns_array(slr_epochs)

    start_ns, end_ns = start.value, end.value
    orbit_count = _presence_count(orbit_ns, start_ns, end_ns)
    slr_count = _presence_count(slr_ns, int((start - SLR_MARGIN).value), int((end + SLR_MARGIN).value))
    tle_before = int(np.searchsorted(tle_ns, start_ns, side="right")) if len(tle_ns) else 0
    tle_after = len(tle_ns) - (int(np.searchsorted(tle_ns, end_ns, side="left")) if len(tle_ns) else 0)

    result = {
        "orbit_status": "covered" if orbit_count else "missing",
        "slr_status": "covered" if slr_count else "missing",
        "tle_status": "covered" if (tle_before and tle_after) else "missing",
        "orbit_sample_count": orbit_count,
        "slr_np_count": slr_count,
        "tle_epochs_before": tle_before,
        "tle_epochs_after": tle_after,
    }
    for source in SOURCE_TYPES:
        result[f"{source}_file_span_status"] = result[f"{source}_status"]
    return result


def manifest_span_coverage(manifest: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """sentinel-6a orbit coverage from RINEX manifest file spans.

    Covered <=> any manifest row's [span_start, span_end] overlaps the window
    (strict overlap, mirroring the historical span audit). This keeps the
    ORIGINAL span semantics for s6a because no orbit samples ship for it.
    """
    if manifest is None or manifest.empty:
        return {"status": "missing", "file_count": 0, "files": ""}
    starts = pd.to_datetime(manifest["span_start_utc"], utc=True)
    ends = pd.to_datetime(manifest["span_end_utc"], utc=True)
    mask = (starts < end) & (ends > start)
    files = manifest.loc[mask, "file_name"].astype(str).tolist()
    return {
        "status": "covered" if files else "missing",
        "file_count": len(files),
        "files": ";".join(files),
    }


# ---------------------------------------------------------------------------
# evidence registry (declarative routing)


def _registry_sat_ids(dataset_root: Path) -> list[str]:
    sat_ids: set[str] = set()
    annotations = dataset_root / "mission_reported" / "annotations"
    if annotations.is_dir():
        for path in sorted(annotations.glob("*.csv")):
            try:
                frame = pd.read_csv(path, usecols=["sat_id"])
            except (ValueError, pd.errors.ParserError):
                continue
            sat_ids.update(frame["sat_id"].dropna().astype(str))
    return sorted(sat_ids)


def build_evidence_registry(dataset_root: str | Path) -> pd.DataFrame:
    """One row per (satellite, source): where the artifact lives + semantics.

    - parquet-backed satellites: orbit ->
      ``mission_reported/evidence/orbit/<sat>.parquet``
      with semantics ``orbit_states``;
    - sentinel-6a: orbit -> ``s6a_gnss_coverage_manifest.csv`` with semantics
      ``gnss_tracking_availability`` (resolved against the self-check output
      directory until WP7 wires it into the release tree);
    - all satellites: tle/slr -> parquet with presence semantics
      (``tle_epochs`` / ``slr_normal_points``).
    """
    dataset_root = Path(dataset_root)
    rows = []
    for sat_id in _registry_sat_ids(dataset_root):
        orbit_parquet = dataset_root / "mission_reported" / "evidence" / "orbit" / f"{sat_id}.parquet"
        if orbit_parquet.exists():
            orbit_artifact, orbit_semantics = f"mission_reported/evidence/orbit/{sat_id}.parquet", SEMANTICS_ORBIT_STATES
        elif sat_id == SENTINEL_6A:
            orbit_artifact, orbit_semantics = S6A_MANIFEST_NAME, SEMANTICS_GNSS_TRACKING
        else:
            orbit_artifact, orbit_semantics = "", SEMANTICS_ORBIT_STATES
        rows.append({"sat_id": sat_id, "source_type": "orbit", "artifact": orbit_artifact, "semantics": orbit_semantics})
        for source_type, semantics in (("tle", SEMANTICS_TLE_EPOCHS), ("slr", SEMANTICS_SLR_NPS)):
            parquet = dataset_root / "mission_reported" / "evidence" / source_type / f"{sat_id}.parquet"
            rows.append({
                "sat_id": sat_id,
                "source_type": source_type,
                "artifact": f"mission_reported/evidence/{source_type}/{sat_id}.parquet" if parquet.exists() else "",
                "semantics": semantics,
            })
    return pd.DataFrame(rows, columns=["sat_id", "source_type", "artifact", "semantics"])


def write_evidence_registry(dataset_root: str | Path, output_dir: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    output_dir = ensure_directory(Path(output_dir))
    registry = build_evidence_registry(dataset_root)
    path = output_dir / REGISTRY_NAME
    registry.to_csv(path, index=False)
    return path


def _resolve_artifact(artifact: str, dataset_root: Path, output_dir: Path) -> Path | None:
    if not artifact:
        return None
    path = Path(artifact)
    if path.is_absolute():
        return path
    if artifact.startswith("mission_reported/evidence/"):
        return dataset_root / artifact
    return output_dir / artifact


# ---------------------------------------------------------------------------
# s6a RINEX manifest generator


def parse_rinex_header(path: Path, *, body_fallback: bool = True) -> dict:
    """Read a gzipped RINEX 3.x observation file's observation span.

    Spans come from the header's ``TIME OF FIRST OBS`` / ``TIME OF LAST OBS``
    (never from file names). 19 of the 28 archive files (the GMV-written
    RINEX 3.03 ones) omit ``TIME OF LAST OBS``; for those the span end falls
    back to the LAST ``>`` observation-epoch marker in the body
    (``body_fallback``), which is still a header/body read of actual data,
    not a file-name convention. ``last_obs_source`` records which rule fired.
    """
    version = ""
    obs_type = ""
    first_obs: pd.Timestamp | None = None
    last_obs: pd.Timestamp | None = None
    last_obs_source = ""
    in_body = False
    last_epoch_line = ""

    def _parse_obs_time(tokens: Sequence[str]) -> pd.Timestamp | None:
        if len(tokens) < 6:
            return None
        try:
            moment = datetime(int(tokens[0]), int(tokens[1]), int(tokens[2]),
                              int(tokens[3]), int(tokens[4])) + timedelta(seconds=float(tokens[5]))
        except ValueError:
            return None
        return pd.Timestamp(moment).tz_localize("UTC")

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not in_body:
                label = line[60:].strip()
                body = line[:60]
                if label == "RINEX VERSION / TYPE":
                    tokens = body.split()
                    version = tokens[0] if tokens else ""
                    obs_type = tokens[1] if len(tokens) > 1 else ""
                elif label == "TIME OF FIRST OBS":
                    first_obs = _parse_obs_time(body.split())
                elif label == "TIME OF LAST OBS":
                    last_obs = _parse_obs_time(body.split())
                    last_obs_source = "header"
                elif label == "END OF HEADER":
                    if last_obs is not None or not body_fallback:
                        break
                    in_body = True  # keep streaming to find the last epoch
                continue
            if line.startswith(">"):
                last_epoch_line = line
    if last_obs is None and last_epoch_line:
        parsed = _parse_obs_time(last_epoch_line[1:].split())
        if parsed is not None:
            last_obs = parsed
            last_obs_source = "body_last_epoch"
    return {
        "rinex_version": version,
        "obs_type": obs_type,
        "first_obs": first_obs,
        "last_obs": last_obs,
        "last_obs_source": last_obs_source,
    }


def write_s6a_rinex_coverage_manifest(
    dataset_root: str | Path | None = None,
    rinex_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Write the sentinel-6a GNSS tracking-availability manifest (D3).

    Scans ``*.rnx.gz`` under ``rinex_dir`` (default: the legacy raw archive's
    ``sentinel-6a/pod``), minimally validates ``RINEX VERSION / TYPE`` — a
    RINEX 3.x version and an observation ("O…") type; the archive is mixed
    3.03/3.04 and both are accepted, with the exact version recorded per
    file — extracts the observation span from ``TIME OF FIRST OBS`` /
    ``TIME OF LAST OBS`` headers; files whose headers omit ``TIME OF LAST
    OBS`` (the 19 GMV RINEX 3.03 files in the archive) fall back to their
    last ``>`` observation-epoch marker in the body — and writes
    ``file_name, rinex_version, span_start_utc, span_end_utc, bytes`` to
    ``output_path`` (default staged under ``experiments/validation/``;
    ``dataset_root`` is accepted for wiring parity).
    Any file that fails validation aborts loudly — no partial silent
    manifest.

    Header times are taken as UTC; the GPS-system leap-second offset (<=18 s)
    is immaterial at the 30 h window granularity.
    """
    del dataset_root  # registry/manifest live under experiments/validation
    if rinex_dir is None:
        rinex_dir = DEFAULT_RINEX_DIR
    rinex_dir = Path(rinex_dir)
    if output_path is None:
        output_path = DEFAULT_ARTIFACTS_DIR / S6A_MANIFEST_NAME
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    paths = sorted(rinex_dir.glob("*.rnx.gz")) if rinex_dir.is_dir() else []
    rows: list[dict] = []
    errors: list[str] = []
    for path in paths:
        header = parse_rinex_header(path)
        version = header["rinex_version"]
        obs_type = header["obs_type"]
        if not version.startswith("3.") or not obs_type.upper().startswith("O"):
            errors.append(
                f"{path.name}: expected RINEX VERSION / TYPE 3.x OBS, got "
                f"{version!r} {obs_type!r}"
            )
            continue
        if header["first_obs"] is None or header["last_obs"] is None:
            if header["first_obs"] is None:
                errors.append(f"{path.name}: missing TIME OF FIRST OBS header")
            else:
                errors.append(f"{path.name}: missing TIME OF LAST OBS header and no '>' epoch records in body")
            continue
        rows.append({
            "file_name": path.name,
            "rinex_version": header["rinex_version"],
            "span_start_utc": _iso_utc(header["first_obs"]),
            "span_end_utc": _iso_utc(header["last_obs"]),
            "bytes": path.stat().st_size,
        })
    if errors:
        raise ValueError("s6a RINEX manifest validation failed: " + "; ".join(errors))
    if not paths:
        raise ValueError(f"no *.rnx.gz files found under {rinex_dir}")

    ensure_directory(output_path.parent)
    manifest = pd.DataFrame(rows, columns=["file_name", "rinex_version", "span_start_utc", "span_end_utc", "bytes"])
    manifest = manifest.sort_values("span_start_utc").reset_index(drop=True)
    manifest.to_csv(output_path, index=False)
    spans = pd.to_datetime(manifest["span_start_utc"], utc=True, format="ISO8601")
    return {
        "output_path": str(output_path),
        "rows": int(len(manifest)),
        "span_start_utc": _iso_utc(spans.min()) if len(spans) else "",
        "span_end_utc": _iso_utc(pd.to_datetime(manifest["span_end_utc"], utc=True, format="ISO8601").max()) if len(manifest) else "",
    }


# ---------------------------------------------------------------------------
# self-check


@dataclass
class SelfcheckReport:
    dataset_root: Path
    output_dir: Path
    registry_path: Path
    table_window_counts: dict[str, int]
    category_counts: dict[str, int]
    diffs: pd.DataFrame
    windows: pd.DataFrame
    artifact_errors: list[dict] = field(default_factory=list)
    s6a_manifest_rows: int = 0
    s6a_manifest_available: bool = False

    @property
    def ok(self) -> bool:
        # An artifact error must fail the gate even when no per-window diff
        # results (e.g. the affected windows' shipped statuses already say
        # missing) — otherwise a deleted/corrupt parquet passes silently.
        return self.category_counts.get(CATEGORY_UNEXPECTED, 0) == 0 and not self.artifact_errors


def _load_epoch_parquet(path: Path) -> pd.Series:
    frame = pd.read_parquet(path, columns=["epoch"])
    if frame.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return _normalize_epochs(frame["epoch"])


class _EvidenceStore:
    """Normalized epoch Series + s6a manifest, loaded per registry row.

    A declared-but-absent s6a manifest is NOT an artifact error: it is the
    known pre-generation state covered by ``s6a_manifest_missing``. A missing
    or unreadable parquet (and an unreadable manifest) IS an error and turns
    every affected covered->missing downgrade into ``unexpected``.
    """

    def __init__(self, registry: pd.DataFrame, dataset_root: Path, output_dir: Path, artifacts_dir: Path) -> None:
        self.errors: list[dict] = []
        self.series: dict[tuple[str, str], pd.Series] = {}
        self.manifest: pd.DataFrame = pd.DataFrame()
        for row in registry.itertuples(index=False):
            key = (str(row.sat_id), str(row.source_type))
            artifact = _resolve_artifact(str(row.artifact), dataset_root, artifacts_dir)
            semantics = str(row.semantics)
            if artifact is None:
                self.errors.append({
                    "sat_id": key[0], "source_type": key[1], "artifact": str(row.artifact),
                    "kind": "artifact_unregistered",
                    "error": "registry row has no artifact (evidence file absent at registry build time)",
                })
                continue
            if semantics == SEMANTICS_GNSS_TRACKING:
                if not artifact.exists():
                    continue  # known pre-generation state, not an error
                try:
                    manifest = pd.read_csv(artifact)
                except Exception as exc:
                    self.errors.append({"sat_id": key[0], "source_type": key[1], "artifact": str(artifact),
                                        "kind": "artifact_unreadable", "error": str(exc)})
                    continue
                if not manifest.empty:
                    self.manifest = manifest
                continue
            try:
                epochs = _load_epoch_parquet(artifact)
            except Exception as exc:
                self.errors.append({"sat_id": key[0], "source_type": key[1], "artifact": str(artifact),
                                    "kind": "artifact_missing" if not artifact.exists() else "artifact_unreadable",
                                    "error": str(exc)})
                continue
            self.series[key] = epochs

    def artifact_error(self, sat_id: str, source_type: str) -> dict | None:
        for error in self.errors:
            if error["sat_id"] == sat_id and error["source_type"] == source_type:
                return error
        return None

    @property
    def s6a_manifest_available(self) -> bool:
        return len(self.manifest) > 0


def _classify_status_diff(
    sat_id: str,
    source: str,
    old: str,
    new: str,
    *,
    orbit_band_in_parquet_span: bool,
    s6a_manifest_available: bool,
) -> tuple[str, str] | None:
    """Categorize one (sat, source) status diff; None when values agree."""
    if old == new:
        return None
    if old not in ("covered", "missing") or new not in ("covered", "missing"):
        return CATEGORY_UNEXPECTED, "invalid_status"
    if old == "covered" and new == "missing":
        if source == "orbit" and sat_id in A1_BACKFILL_SATS and not orbit_band_in_parquet_span:
            return CATEGORY_A1_BACKFILL_PENDING, "status_downgrade"
        if source == "orbit" and sat_id == SENTINEL_6A and not s6a_manifest_available:
            return CATEGORY_S6A_MANIFEST_MISSING, "status_downgrade"
        if source == "orbit":
            return CATEGORY_SEMANTICS_FLIP_ORBIT, "status_downgrade"
        if source == "slr":
            return CATEGORY_SEMANTICS_FLIP_SLR, "status_downgrade"
        return CATEGORY_UNEXPECTED, "tle_downgrade"
    return CATEGORY_UNEXPECTED, "status_upgrade"


def _discover_window_tables(dataset_root: Path) -> list[Path]:
    annotations = dataset_root / "mission_reported" / "annotations"
    if not annotations.is_dir():
        return []
    tables = []
    for path in sorted(annotations.glob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except pd.errors.ParserError:
            continue
        if all(column in header.columns for column in REQUIRED_WINDOW_COLUMNS):
            tables.append(path)
    return tables


def run_selfcheck(
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    rinex_dir: str | Path | None = None,
    generate_manifest: bool = False,
    artifacts_dir: str | Path | None = None,
) -> SelfcheckReport:
    """Recompute all window coverage + tiers and diff the annotation tables.

    Loads evidence per the declarative registry (writing it on first run),
    optionally generates the s6a RINEX manifest first, then categorizes every
    per-window divergence. Known classes keep the gate green; the report is
    red on the ``unexpected`` class or on any artifact error
    (``artifact_errors``), even when no per-window diff results.
    """
    dataset_root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir = ensure_directory(output_dir)
    artifacts_dir = ensure_directory(Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR)

    if generate_manifest:
        write_s6a_rinex_coverage_manifest(rinex_dir=rinex_dir, output_path=artifacts_dir / S6A_MANIFEST_NAME)

    registry_path = artifacts_dir / REGISTRY_NAME
    if not registry_path.exists():
        registry_path = write_evidence_registry(dataset_root, artifacts_dir)
    registry = pd.read_csv(registry_path)

    store = _EvidenceStore(registry, dataset_root, output_dir, artifacts_dir)

    diff_records: list[dict] = []
    window_records: list[dict] = []
    table_window_counts: dict[str, int] = {}
    category_counts = {category: 0 for category in ALL_CATEGORIES}

    tables = _discover_window_tables(dataset_root)
    maneuver_ids: set[str] | None = None
    maneuver_path = dataset_root / "mission_reported" / "annotations" / "maneuver_annotations.csv"
    if maneuver_path.exists():
        maneuver_ids = set(pd.read_csv(maneuver_path, usecols=["annotation_id"])["annotation_id"].astype(str))

    for table_path in tables:
        table_name = table_path.name
        frame = pd.read_csv(table_path)
        table_window_counts[table_name] = int(len(frame))
        seen_ids: set[str] = set()
        for row in frame.itertuples(index=False):
            annotation_id = str(row.annotation_id)
            sat_id = str(row.sat_id)
            start, end = _window_bounds(row)

            def _record(field: str, shipped: str, recomputed: str, category: str, kind: str) -> None:
                diff_records.append({
                    "table": table_name, "annotation_id": annotation_id, "sat_id": sat_id,
                    "field": field, "shipped_value": shipped, "recomputed_value": recomputed,
                    "category": category, "kind": kind,
                    "window_start_utc": _iso_utc(start), "window_end_utc": _iso_utc(end),
                })

            if annotation_id in seen_ids:
                _record("annotation_id", "", "", CATEGORY_UNEXPECTED, "duplicate_id")
                continue
            seen_ids.add(annotation_id)
            if maneuver_ids is not None and table_name.startswith("event_windows") and annotation_id not in maneuver_ids:
                _record("annotation_id", "", "", CATEGORY_UNEXPECTED, "orphan_event_window")

            # -- route coverage through the registry ----------------------
            orbit_series = store.series.get((sat_id, "orbit"))
            coverage = coverage_from_parquet(
                orbit_series,
                store.series.get((sat_id, "tle")),
                store.series.get((sat_id, "slr")),
                (start, end),
            )
            if sat_id == SENTINEL_6A and (sat_id, "orbit") not in store.series:
                span = manifest_span_coverage(store.manifest, start, end)
                coverage["orbit_status"] = span["status"]
                coverage["orbit_file_span_status"] = span["status"]

            statuses = {source: coverage[f"{source}_status"] for source in SOURCE_TYPES}
            tier = coverage_confidence_tier(statuses["tle"], statuses["slr"], statuses["orbit"])
            missing = [source for source in SOURCE_TYPES if statuses[source] != "covered"]
            window_records.append({
                "table": table_name,
                "annotation_id": annotation_id,
                "sat_id": sat_id,
                "batch": str(getattr(row, "batch", "")),
                "window_start_utc": _iso_utc(start),
                "window_end_utc": _iso_utc(end),
                **{STATUS_COLUMNS[source]: statuses[source] for source in SOURCE_TYPES},
                **{f"{source}_file_span_status": coverage[f"{source}_file_span_status"] for source in SOURCE_TYPES},
                "tle_epochs_before": coverage["tle_epochs_before"],
                "tle_epochs_after": coverage["tle_epochs_after"],
                "orbit_sample_count": coverage["orbit_sample_count"],
                "slr_np_count": coverage["slr_np_count"],
                "confidence_tier": tier,
                "missing_sources": ",".join(missing),
                "aligned": not missing,
                "quality_flags": ",".join(f"missing_{source}" for source in missing),
            })

            # -- diff statuses; tier only when statuses agree but it differs
            has_status_diff = False
            for source in SOURCE_TYPES:
                old = str(getattr(row, STATUS_COLUMNS[source]))
                new = statuses[source]
                artifact_error = store.artifact_error(sat_id, source)
                if artifact_error is not None:
                    if old != new:
                        has_status_diff = True
                        _record(STATUS_COLUMNS[source], old, new, CATEGORY_UNEXPECTED, artifact_error["kind"])
                    continue
                orbit_in_span = True
                if source == "orbit" and orbit_series is not None:
                    ns = _ns_array(orbit_series)
                    if len(ns):
                        orbit_in_span = bool(ns[0] <= start.value and ns[-1] >= end.value)
                classified = _classify_status_diff(
                    sat_id, source, old, new,
                    orbit_band_in_parquet_span=orbit_in_span,
                    s6a_manifest_available=store.s6a_manifest_available,
                )
                if classified is None:
                    continue
                has_status_diff = True
                category, kind = classified
                _record(STATUS_COLUMNS[source], old, new, category, kind)
            old_tier = str(getattr(row, "confidence_tier"))
            if old_tier != tier and not has_status_diff:
                _record("confidence_tier", old_tier, tier, CATEGORY_UNEXPECTED, "tier_inconsistent")

    diffs = pd.DataFrame(diff_records, columns=DIFF_COLUMNS)
    windows = pd.DataFrame(window_records, columns=RECOMPUTED_WINDOW_COLUMNS)
    if not diffs.empty:
        for category, count in diffs["category"].value_counts().items():
            category_counts[str(category)] = int(count)
    return SelfcheckReport(
        dataset_root=dataset_root,
        output_dir=output_dir,
        registry_path=registry_path,
        table_window_counts=table_window_counts,
        category_counts=category_counts,
        diffs=diffs,
        windows=windows,
        artifact_errors=store.errors,
        s6a_manifest_rows=len(store.manifest),
        s6a_manifest_available=len(store.manifest) > 0,
    )


def write_selfcheck_outputs(report: SelfcheckReport, report_json_path: str | Path | None = None) -> dict[str, Path]:
    """Stage diffs, recomputed windows, and the JSON summary (git-ignored)."""
    written: dict[str, Path] = {}
    diffs_path = report.output_dir / DIFFS_NAME
    report.diffs.to_csv(diffs_path, index=False)
    written["diffs"] = diffs_path
    windows_path = report.output_dir / WINDOWS_NAME
    report.windows.to_csv(windows_path, index=False)
    written["windows"] = windows_path
    json_path = Path(report_json_path) if report_json_path else report.output_dir / REPORT_NAME
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    payload = {
        "dataset_root": str(report.dataset_root),
        "output_dir": str(report.output_dir),
        "registry_path": str(report.registry_path),
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "table_window_counts": report.table_window_counts,
        "category_counts": report.category_counts,
        "unexpected_count": int(report.category_counts.get(CATEGORY_UNEXPECTED, 0)),
        "ok": bool(report.ok),
        "s6a_manifest": {"available": report.s6a_manifest_available, "rows": report.s6a_manifest_rows},
        "artifact_errors": report.artifact_errors,
    }
    ensure_directory(json_path.parent)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written["report"] = json_path
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Release self-consistency gate (recompute coverage, diff annotations)")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR),
                        help="Where the gate-owned release artifacts (evidence registry, "
                             "s6a GNSS coverage manifest) are read/written "
                             "(default: experiments/validation)")
    parser.add_argument("--report", default=None, help="JSON report path (default: <output>/selfcheck_report.json)")
    parser.add_argument("--write-registry", action="store_true", help="Force-regenerate the evidence registry")
    parser.add_argument("--write-s6a-manifest", action="store_true",
                        help="Generate the s6a RINEX manifest from the legacy archive before checking")
    parser.add_argument("--rinex-dir", default=None, help="Override the sentinel-6a RINEX archive directory")
    args = parser.parse_args(argv)

    dataset_root = resolve_repo_path(args.dataset) or DEFAULT_DATASET_ROOT
    output_dir = resolve_repo_path(args.output) or DEFAULT_OUTPUT_DIR
    output_dir = ensure_directory(output_dir)
    artifacts_dir = ensure_directory(resolve_repo_path(args.artifacts_dir) or DEFAULT_ARTIFACTS_DIR)
    rinex_dir = resolve_repo_path(args.rinex_dir) if args.rinex_dir else None

    if args.write_registry:
        write_evidence_registry(dataset_root, artifacts_dir)
    if args.write_s6a_manifest:
        try:
            summary = write_s6a_rinex_coverage_manifest(rinex_dir=rinex_dir, output_path=artifacts_dir / S6A_MANIFEST_NAME)
            print(json.dumps({"s6a_manifest": summary}, sort_keys=True), flush=True)
        except (OSError, ValueError) as exc:
            print(f"ERROR: s6a manifest generation failed: {exc}", file=sys.stderr, flush=True)
            return 2

    report = run_selfcheck(dataset_root, output_dir=output_dir, artifacts_dir=artifacts_dir)
    written = write_selfcheck_outputs(report, report_json_path=args.report)
    print(
        json.dumps(
            {
                "ok": bool(report.ok),
                "category_counts": report.category_counts,
                "unexpected_count": int(report.category_counts.get(CATEGORY_UNEXPECTED, 0)),
                "tables": report.table_window_counts,
                "s6a_manifest_rows": report.s6a_manifest_rows,
                "artifact_errors": len(report.artifact_errors),
                "diffs_csv": str(written["diffs"]),
                "report_json": str(written["report"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
