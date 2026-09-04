"""Freeze the external cross-validation against the Shorten et al. benchmark.

Compares maneuver timestamps in the mirrored benchmark dataset
(data/external/TLE_observation_benchmark_dataset) against the mission-reported
annotations, per satellite. The benchmark carries a systematic +1 day offset
(traced to day-of-year parsing on their side; ours verified against raw IDS
records), so matches are evaluated within a one-day tolerance.

Matching methodology (REVIEW_FINDINGS 7.1.5 / B20): a UNIQUE greedy match --
all candidate pairs are ordered by ascending |dt| and accepted greedily only
when neither side is already taken, which makes the match a bijection and
forbids many-to-one in both directions. Tolerance sensitivity is reported as
a 0.5/1/2-day sweep, and the matched-pair signed-offset distribution (the
evidence behind the "+1 day systematic offset" statement) is exported as a
histogram table for the paper.

Both directions are reported: forward (benchmark events matched by MAD-LEO)
and reverse (MAD-LEO events matched by the benchmark). The reverse direction
is evaluated at per-event resolution with three denominators -- all events,
shared-target events, and events restricted to the benchmark's per-target
time span -- plus a magnitude comparison of matched versus unmatched events
(the benchmark is a detection method with a sensitivity floor, so MAD-LEO-only
in-span events are expected to be smaller). Targets the benchmark does not
cover (SWOT) are listed explicitly with benchmark_events=0.

All tolerances and the sweep grid come from
``benchmarking.experiment_params`` (P4 single source).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.experiment_params import (
    BENCHMARK_MATCH_TOLERANCE_DAYS,
    BENCHMARK_MATCH_TOLERANCE_SWEEP_DAYS,
)

BENCHMARK_DIR = REPO_ROOT / "data" / "external" / "TLE_observation_benchmark_dataset" / "processed_files"
INTERIM = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"
DEFAULT_RESPONSE_TABLE = REPO_ROOT / "results" / "tables" / "maneuver_event_response_validation.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "external_benchmark_crossvalidation.csv"
DEFAULT_PER_EVENT_OUTPUT = REPO_ROOT / "results" / "tables" / "external_benchmark_crossvalidation_per_event.csv"
DEFAULT_SWEEP_OUTPUT = REPO_ROOT / "results" / "tables" / "external_benchmark_tolerance_sweep.csv"
DEFAULT_HISTOGRAM_OUTPUT = REPO_ROOT / "results" / "tables" / "external_benchmark_match_offsets.csv"

MATCH_TOLERANCE_SECONDS = int(BENCHMARK_MATCH_TOLERANCE_DAYS * 86400)
TOLERANCE_SWEEP_DAYS = BENCHMARK_MATCH_TOLERANCE_SWEEP_DAYS
OFFSET_HISTOGRAM_BIN_MINUTES = 60.0

OFFSET_NOTE = (
    "unique greedy bijection on ascending |dt| (no many-to-one); "
    "benchmark timestamps carry a systematic +1 day offset (their day-of-year parsing); "
    "matched within 1-day tolerance"
)
NOT_COVERED_NOTE = "not covered by benchmark; validated by internal experiments only"

NAME_MAP = {
    "CryoSat-2": "cryosat-2", "Haiyang-2A": "hy-2a", "Jason-1": "jason-1", "Jason-2": "jason-2",
    "Jason-3": "jason-3", "SARAL": "saral", "Sentinel-3A": "sentinel-3a", "Sentinel-3B": "sentinel-3b",
    "Sentinel-6A": "sentinel-6a", "TOPEX": "topex-poseidon",
}

# Benchmark-covered targets first (benchmark order), then targets outside the
# benchmark's coverage. SWOT is currently the only uncovered target.
TARGET_ORDER = [*NAME_MAP.values(), "swot"]


def _to_epoch_ns(values) -> np.ndarray:
    """UTC-parse an iterable of timestamps to epoch nanoseconds (int64).

    Integer input is passed through (already epoch nanoseconds).  ISO8601
    format hint: the tables mix whole-second and millisecond precision
    strings, which pandas' single-format inference rejects.
    """
    series = pd.Series(list(values))
    if pd.api.types.is_integer_dtype(series):
        return series.to_numpy(dtype=np.int64)
    return pd.to_datetime(series, utc=True, format="ISO8601").astype("int64").to_numpy(dtype=np.int64)


def greedy_unique_match_pairs(
    query, reference, tolerance_seconds: float
) -> list[tuple[int, int, float]]:
    """Unique greedy (bijective) match between two timestamp sets.

    All candidate pairs within ``tolerance_seconds`` are ordered by ascending
    ``|dt|`` (ties broken by query then reference index for determinism) and
    accepted greedily only while BOTH sides are still free, so every event is
    used at most once on each side (bijection; many-to-one impossible). The
    result does not depend on which argument is the query.

    Returns ``[(query_index, reference_index, signed_offset_seconds)]`` where
    the signed offset is ``reference - query`` (positive: the reference side
    is later).
    """
    q_ns = _to_epoch_ns(query)
    r_ns = _to_epoch_ns(reference)
    limit_ns = int(round(float(tolerance_seconds) * 1e9))
    candidates: list[tuple[int, int, int, int]] = []
    for i, q in enumerate(q_ns):
        for j, r in enumerate(r_ns):
            delta = int(r) - int(q)
            if abs(delta) <= limit_ns:
                candidates.append((abs(delta), i, j, delta))
    candidates.sort()
    used_query: set[int] = set()
    used_reference: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for adt, i, j, delta in candidates:
        if i in used_query or j in used_reference:
            continue
        used_query.add(i)
        used_reference.add(j)
        pairs.append((i, j, delta / 1e9))
    return pairs


def _nearest_offset_seconds(ref_values: np.ndarray, value: np.datetime64) -> float | None:
    """Absolute offset (seconds) to the nearest reference timestamp, or None."""
    idx = int(np.searchsorted(ref_values, value))
    best = None
    for j in (idx - 1, idx):
        if 0 <= j < len(ref_values):
            delta = abs((ref_values[j] - value) / np.timedelta64(1, "s"))
            if best is None or delta < best:
                best = delta
    return best


def _load_benchmark_timestamps(benchmark_dir: Path) -> dict[str, pd.Series]:
    """Load sorted benchmark maneuver timestamps per sat_id (skip missing files)."""
    table: dict[str, pd.Series] = {}
    for name, sat_id in NAME_MAP.items():
        yaml_path = benchmark_dir / f"manoeuvres_{name}.yaml"
        if not yaml_path.exists():
            continue
        table[sat_id] = pd.to_datetime(
            pd.Series(yaml.safe_load(yaml_path.read_text())["manoeuvre_timestamps"]), utc=True
        ).sort_values().reset_index(drop=True)
    return table


def _load_events(interim_root: Path) -> dict[str, pd.DataFrame]:
    """Per-satellite mission-reported annotation rows in file order."""
    events: dict[str, pd.DataFrame] = {}
    for sat_id in TARGET_ORDER:
        ann_path = Path(interim_root) / sat_id / "maneuver_annotations.csv"
        if not ann_path.exists():
            continue
        ours = pd.read_csv(ann_path)
        ours = ours.assign(
            _event_ns=_to_epoch_ns(ours["event_time_utc"]),
            _event_ts=pd.to_datetime(ours["event_time_utc"], utc=True, format="ISO8601"),
        )
        events[sat_id] = ours
    return events


def _load_response_magnitudes(response_table: Path | None) -> dict[str, float]:
    """Map annotation_id -> |delta_sma| in meters from the event-response table."""
    if response_table is None or not Path(response_table).exists():
        return {}
    response = pd.read_csv(response_table)
    magnitudes = response["delta_sma_km"].abs() * 1000.0
    return magnitudes.groupby(response["annotation_id"]).first().to_dict()


def _median_dsma_m(frame: pd.DataFrame) -> float | str:
    """Median of ``abs_delta_sma_m`` with the empty-string placeholder guarded.

    C9: the column mixes ``""`` placeholders for events without a computable
    response magnitude; a plain ``.median()`` on that object column raises
    TypeError once any placeholder reaches the selected subset.  Coerce to
    numeric first and emit ``""`` when nothing computable remains.
    """
    values = pd.to_numeric(frame["abs_delta_sma_m"], errors="coerce").dropna()
    return round(float(values.median()), 1) if len(values) else ""


def _match_benchmark(
    benchmark: dict[str, pd.Series], events: dict[str, pd.DataFrame], tolerance_seconds: float
) -> dict[str, list[tuple[int, int, float]]]:
    """Bijection per satellite at one tolerance (query = our events)."""
    return {
        sat_id: greedy_unique_match_pairs(events[sat_id]["_event_ns"], benchmark[sat_id], tolerance_seconds)
        for sat_id in events
        if sat_id in benchmark
    }


def build_per_event_match(
    benchmark_dir: Path = BENCHMARK_DIR,
    interim_root: Path = INTERIM,
    response_table: Path | None = DEFAULT_RESPONSE_TABLE,
    tolerance_seconds: float = MATCH_TOLERANCE_SECONDS,
) -> pd.DataFrame:
    """One row per mission-reported event: reverse-match status against the benchmark.

    ``matched_within_1d`` comes from the unique greedy bijection at the
    primary tolerance (REVIEW_FINDINGS 7.1.5); ``signed_offset_seconds`` is
    the matched pair offset (benchmark - ours); ``abs_offset_seconds`` keeps
    the nearest-event diagnostic for every covered event.
    """
    benchmark = _load_benchmark_timestamps(Path(benchmark_dir))
    magnitudes = _load_response_magnitudes(response_table)
    events = _load_events(Path(interim_root))
    pairs_by_sat = _match_benchmark(benchmark, events, tolerance_seconds)
    rows = []
    for sat_id in TARGET_ORDER:
        ours = events.get(sat_id)
        if ours is None:
            continue
        bm_ts = benchmark.get(sat_id)
        covered = bm_ts is not None and len(bm_ts) > 0
        matched_signed = {i: signed for i, _, signed in pairs_by_sat.get(sat_id, [])}
        if covered:
            ref_values = bm_ts.dt.tz_localize(None).to_numpy()
            span_lo = bm_ts.iloc[0] - pd.Timedelta(seconds=tolerance_seconds)
            span_hi = bm_ts.iloc[-1] + pd.Timedelta(seconds=tolerance_seconds)
        for row_number, (_, row) in enumerate(ours.iterrows()):
            event_ts = row["_event_ts"]
            offset = _nearest_offset_seconds(ref_values, event_ts.to_datetime64()) if covered else None
            matched = row_number in matched_signed
            signed = matched_signed.get(row_number)
            within_span = bool(covered and span_lo <= event_ts <= span_hi)
            annotation_id = row["annotation_id"]
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "sat_id": sat_id,
                    "event_time_utc": row["event_time_utc"],
                    "benchmark_covered_target": covered,
                    "within_benchmark_span": within_span,
                    "matched_within_1d": matched,
                    "signed_offset_seconds": round(signed, 1) if signed is not None else "",
                    "abs_offset_seconds": round(offset, 1) if offset is not None else "",
                    "abs_delta_sma_m": round(magnitudes[annotation_id], 1) if annotation_id in magnitudes else "",
                    "note": "" if covered else NOT_COVERED_NOTE,
                }
            )
    return pd.DataFrame(rows)


def build_crossvalidation(
    benchmark_dir: Path = BENCHMARK_DIR,
    interim_root: Path = INTERIM,
    response_table: Path | None = DEFAULT_RESPONSE_TABLE,
) -> pd.DataFrame:
    """Per-satellite summary plus an ALL row with the three reverse-match denominators."""
    benchmark = _load_benchmark_timestamps(Path(benchmark_dir))
    events = _load_events(Path(interim_root))
    pairs_by_sat = _match_benchmark(benchmark, events, MATCH_TOLERANCE_SECONDS)
    per_event = build_per_event_match(Path(benchmark_dir), Path(interim_root), response_table)
    rate_columns = (
        "reverse_match_rate_all_events",
        "reverse_match_rate_shared_targets",
        "reverse_match_rate_in_benchmark_span",
    )
    rows = []
    total_benchmark_matched = 0
    for sat_id in TARGET_ORDER:
        sub = per_event[per_event["sat_id"] == sat_id]
        if sub.empty:
            continue
        bm_ts = benchmark.get(sat_id)
        if bm_ts is None:
            rows.append(
                {
                    "sat_id": sat_id,
                    "benchmark_events": 0,
                    "madleo_events": len(sub),
                    "benchmark_matched": 0,
                    "madleo_matched": 0,
                    "median_abs_offset_minutes": "",
                    "madleo_events_in_benchmark_span": 0,
                    "madleo_matched_in_benchmark_span": 0,
                    "median_abs_dsma_matched_m": "",
                    "median_abs_dsma_unmatched_m": "",
                    **dict.fromkeys(rate_columns, ""),
                    "note": NOT_COVERED_NOTE,
                }
            )
            continue
        pairs = pairs_by_sat.get(sat_id, [])
        total_benchmark_matched += len(pairs)
        offsets_min = [abs(signed) / 60.0 for _, _, signed in pairs]
        in_span = sub[sub["within_benchmark_span"]]
        unmatched = sub[~sub["matched_within_1d"]]
        matched_span = in_span[in_span["matched_within_1d"]]
        # On current data every in-span event is matched, so the unmatched set
        # is exactly the out-of-span population (the benchmark's coverage
        # boundary); its median magnitude evidences that the uniquely reported
        # events are ordinary maneuvers, not sub-threshold detections.
        rows.append(
            {
                "sat_id": sat_id,
                "benchmark_events": len(bm_ts),
                "madleo_events": len(sub),
                "benchmark_matched": len(pairs),
                "madleo_matched": int(sub["matched_within_1d"].sum()),
                "median_abs_offset_minutes": round(float(np.median(offsets_min)), 1) if offsets_min else "",
                "madleo_events_in_benchmark_span": len(in_span),
                "madleo_matched_in_benchmark_span": int(matched_span.shape[0]),
                "median_abs_dsma_matched_m": _median_dsma_m(matched_span),
                "median_abs_dsma_unmatched_m": _median_dsma_m(unmatched),
                **dict.fromkeys(rate_columns, ""),
                "note": OFFSET_NOTE,
            }
        )
    covered = per_event[per_event["benchmark_covered_target"]]
    in_span = covered[covered["within_benchmark_span"]]
    rows.append(
        {
            "sat_id": "ALL",
            "benchmark_events": int(sum(len(ts) for ts in benchmark.values())),
            "madleo_events": len(per_event),
            "benchmark_matched": total_benchmark_matched,
            "madleo_matched": int(per_event["matched_within_1d"].sum()),
            "median_abs_offset_minutes": "",
            "madleo_events_in_benchmark_span": len(in_span),
            "madleo_matched_in_benchmark_span": int(in_span["matched_within_1d"].sum()),
            "median_abs_dsma_matched_m": _median_dsma_m(in_span[in_span["matched_within_1d"]]),
            "median_abs_dsma_unmatched_m": _median_dsma_m(covered[~covered["matched_within_1d"]]),
            "reverse_match_rate_all_events": round(float(per_event["matched_within_1d"].mean()), 4),
            "reverse_match_rate_shared_targets": round(float(covered["matched_within_1d"].mean()), 4),
            "reverse_match_rate_in_benchmark_span": round(float(in_span["matched_within_1d"].mean()), 4),
            "note": "",
        }
    )
    return pd.DataFrame(rows)


def build_tolerance_sweep(
    benchmark_dir: Path = BENCHMARK_DIR,
    interim_root: Path = INTERIM,
    tolerances_days: tuple[float, ...] = TOLERANCE_SWEEP_DAYS,
) -> pd.DataFrame:
    """Tolerance sensitivity of the unique match: 0.5/1/2 days, per satellite + ALL.

    One row per (satellite, tolerance) for benchmark-covered targets plus an
    ALL aggregate over covered targets; match rates use the bijection, so
    ``benchmark_matched == madleo_matched == unique_pair_count`` is structurally
    impossible to violate per side counts (each pair consumes one event on
    each side).
    """
    benchmark = _load_benchmark_timestamps(Path(benchmark_dir))
    events = _load_events(Path(interim_root))
    rows = []
    for tolerance_days in tolerances_days:
        tolerance_seconds = tolerance_days * 86400.0
        pairs_by_sat = _match_benchmark(benchmark, events, tolerance_seconds)
        for sat_id in TARGET_ORDER:
            if sat_id not in events or sat_id not in benchmark:
                continue
            pairs = pairs_by_sat[sat_id]
            rows.append(
                {
                    "sat_id": sat_id,
                    "tolerance_days": tolerance_days,
                    "benchmark_events": len(benchmark[sat_id]),
                    "madleo_events": len(events[sat_id]),
                    "unique_pair_count": len(pairs),
                    "benchmark_matched": len(pairs),
                    "madleo_matched": len(pairs),
                    "benchmark_match_rate": round(len(pairs) / max(len(benchmark[sat_id]), 1), 4),
                    "madleo_match_rate": round(len(pairs) / max(len(events[sat_id]), 1), 4),
                    "note": OFFSET_NOTE,
                }
            )
        covered_pairs = sum(len(pairs_by_sat[s]) for s in pairs_by_sat)
        covered_benchmark = sum(len(benchmark[s]) for s in pairs_by_sat)
        covered_events = sum(len(events[s]) for s in pairs_by_sat)
        rows.append(
            {
                "sat_id": "ALL",
                "tolerance_days": tolerance_days,
                "benchmark_events": covered_benchmark,
                "madleo_events": covered_events,
                "unique_pair_count": covered_pairs,
                "benchmark_matched": covered_pairs,
                "madleo_matched": covered_pairs,
                "benchmark_match_rate": round(covered_pairs / max(covered_benchmark, 1), 4),
                "madleo_match_rate": round(covered_pairs / max(covered_events, 1), 4),
                "note": "ALL aggregates benchmark-covered targets only",
            }
        )
    return pd.DataFrame(rows)


def build_offset_histogram(
    benchmark_dir: Path = BENCHMARK_DIR,
    interim_root: Path = INTERIM,
    tolerance_seconds: float = MATCH_TOLERANCE_SECONDS,
    bin_minutes: float = OFFSET_HISTOGRAM_BIN_MINUTES,
) -> pd.DataFrame:
    """Distribution of matched-pair signed offsets (paper artifact).

    The "+1 day systematic offset" statement is backed by this table: matched
    offsets (benchmark minus ours, minutes) of the unique bijection at the
    primary tolerance, binned on a common hourly grid across satellites, with
    per-satellite rows and an ALL aggregate. Empty (header-only) when no
    matched pairs exist.
    """
    columns = ["sat_id", "offset_bin_start_min", "offset_bin_end_min", "matched_count"]
    benchmark = _load_benchmark_timestamps(Path(benchmark_dir))
    events = _load_events(Path(interim_root))
    pairs_by_sat = _match_benchmark(benchmark, events, tolerance_seconds)
    offsets_by_sat = {
        sat_id: np.asarray([signed / 60.0 for _, _, signed in pairs], dtype=float)
        for sat_id, pairs in pairs_by_sat.items()
        if pairs
    }
    if not offsets_by_sat:
        return pd.DataFrame(columns=columns)
    all_offsets = np.concatenate(list(offsets_by_sat.values()))
    # Common hourly grid; the upper edge sits one bin beyond the largest
    # observation so a value exactly on a bin edge belongs to the bin it
    # starts (numpy's last bin would otherwise close on the right).
    grid_lo = float(np.floor(all_offsets.min() / bin_minutes) * bin_minutes)
    grid_hi = float(np.floor(all_offsets.max() / bin_minutes) * bin_minutes + bin_minutes)
    edges = np.arange(grid_lo, grid_hi + bin_minutes / 2.0, bin_minutes)
    rows = []
    for sat_id in TARGET_ORDER:
        if sat_id not in offsets_by_sat:
            continue
        counts, _ = np.histogram(offsets_by_sat[sat_id], bins=edges)
        for left, count in zip(edges[:-1], counts):
            if count:
                rows.append(
                    {
                        "sat_id": sat_id,
                        "offset_bin_start_min": float(left),
                        "offset_bin_end_min": float(left + bin_minutes),
                        "matched_count": int(count),
                    }
                )
    total, _ = np.histogram(all_offsets, bins=edges)
    for left, count in zip(edges[:-1], total):
        if count:
            rows.append(
                {
                    "sat_id": "ALL",
                    "offset_bin_start_min": float(left),
                    "offset_bin_end_min": float(left + bin_minutes),
                    "matched_count": int(count),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-event-output", default=str(DEFAULT_PER_EVENT_OUTPUT))
    parser.add_argument("--sweep-output", default=str(DEFAULT_SWEEP_OUTPUT))
    parser.add_argument("--histogram-output", default=str(DEFAULT_HISTOGRAM_OUTPUT))
    parser.add_argument("--response-table", default=str(DEFAULT_RESPONSE_TABLE))
    args = parser.parse_args()
    response_table = resolve_repo_path(args.response_table)
    frame = build_crossvalidation(response_table=response_table)
    out = resolve_repo_path(args.output)
    ensure_directory(out.parent)
    frame.to_csv(out, index=False)
    per_event = build_per_event_match(response_table=response_table)
    per_event_out = resolve_repo_path(args.per_event_output)
    ensure_directory(per_event_out.parent)
    per_event.to_csv(per_event_out, index=False)
    sweep_out = resolve_repo_path(args.sweep_output)
    build_tolerance_sweep().to_csv(sweep_out, index=False)
    histogram_out = resolve_repo_path(args.histogram_output)
    build_offset_histogram().to_csv(histogram_out, index=False)
    all_row = frame[frame["sat_id"] == "ALL"].iloc[0]
    print(
        f"wrote {out} + {per_event_out} + {sweep_out} + {histogram_out} "
        f"forward={all_row['benchmark_matched']}/{all_row['benchmark_events']} "
        f"reverse={all_row['madleo_matched']}/{all_row['madleo_events']} "
        f"(all={all_row['reverse_match_rate_all_events']:.3f} "
        f"shared={all_row['reverse_match_rate_shared_targets']:.3f} "
        f"in_span={all_row['reverse_match_rate_in_benchmark_span']:.3f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
