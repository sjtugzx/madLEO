"""TLE SMA-shift convention comparison: Kepler-on-Kozai vs sgp4 ``a``
(REVIEW_FINDINGS 7.1.7).

The published per-event ``delta_sma_km`` applies Kepler's third law to the
TLE mean motion, which is the **Kozai** mean motion.  python-sgp4's
accelerated ``Satrec`` exposes the semi-major axis ``a`` derived from the
un-Kozai (Brouwer) mean motion, in WGS72 earth radii (x 6378.135 km).
Because the J2 conversion between the two mean-motion conventions is nearly
common-mode between the two TLEs bracketing an event, the two delta-a
conventions agree at the sub-meter level -- this experiment recomputes the
full event table both ways and reports the difference distribution, closing
the reviewer question with one small table.

Implementation notes:

- Uses ``satrec.a`` exclusively.  The ``no_unkozai`` attribute exists only
  in the pure-Python propagation path; the accelerated ``Satrec`` (the one
  constructed by ``Satrec.twoline2rv`` here) does NOT expose it
  (``hasattr(satrec, "no_unkozai") is False``).
- Input: the shipped per-event response table (``delta_sma_km`` published
  values) plus the raw TLE archive; the before/after TLEs are located by
  exact epoch match against the response table's
  ``tle_nearest_before_utc`` / ``tle_nearest_after_utc``.

Run via the experiment dispatcher:

    python scripts/experiments/run_experiments.py kozai-comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec

from analyzers.event_response import mean_motion_to_sma_m
from analyzers.state_estimation import load_tle_line_history
from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path

DEFAULT_RESPONSE_TABLE = REPO_ROOT / "experiments" / "validation" / "maneuver_event_response_validation.csv"
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "reference_validation"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "kozai_comparison.csv"
DEFAULT_PER_EVENT_OUTPUT = REPO_ROOT / "results" / "tables" / "kozai_comparison_per_event.csv"

#: sgp4 semi-major axis unit: WGS72 earth radii (Vallado STR#3 / sgp4's
#: xke definition), converted to km with the matching radius.
EARTH_RADIUS_WGS72_KM = 6378.135

PER_EVENT_COLUMNS = [
    "annotation_id",
    "sat_id",
    "tle_nearest_before_utc",
    "tle_nearest_after_utc",
    "delta_sma_km_published",
    "delta_sma_km_kepler_recomputed",
    "delta_sma_km_sgp4_a",
    "difference_kepler_vs_sgp4_m",
    "status",
]

DISTRIBUTION_COLUMNS = [
    "sat_id",
    "n_events",
    "n_compared",
    "n_missing",
    "median_abs_difference_m",
    "mean_abs_difference_m",
    "p95_abs_difference_m",
    "max_abs_difference_m",
    "median_abs_delta_sma_m",
    "sub_meter_fraction",
]


def kepler_sma_km_from_mean_motion_rad_per_min(mean_motion_rad_per_min: float) -> float:
    """Semi-major axis (km) via Kepler's third law (the published convention)."""
    return mean_motion_to_sma_m(float(mean_motion_rad_per_min)) / 1000.0


def delta_sma_sgp4_km(line1_before: str, line2_before: str, line1_after: str, line2_after: str) -> float:
    """Delta SMA (km) from the sgp4 un-Kozai semi-major axis ``satrec.a``.

    ``satrec.a`` is in WGS72 earth radii on the accelerated Satrec; the
    Kozai mean motion itself must NOT be used for this comparison
    (REVIEW_FINDINGS 7.1.7, 2026-08-28 revision).
    """
    before = Satrec.twoline2rv(line1_before, line2_before)
    after = Satrec.twoline2rv(line1_after, line2_after)
    return (after.a - before.a) * EARTH_RADIUS_WGS72_KM


def _tle_line_index(raw_root: Path, sat_id: str) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """(sorted epoch-ns array, parallel (line1, line2) list) for one target."""
    lines = load_tle_line_history(Path(raw_root) / sat_id / "tle")
    if lines.empty:
        return np.asarray([], dtype=np.int64), []
    epochs_ns = pd.to_datetime(lines["epoch"], utc=True).astype("int64").to_numpy(dtype=np.int64)
    order = np.argsort(epochs_ns)
    pairs = [(str(lines.iloc[i]["line1"]), str(lines.iloc[i]["line2"])) for i in order]
    return epochs_ns[order], pairs


#: The published table rounds TLE epochs to milliseconds while the archive
#: parse keeps microsecond precision; TLE cadence is hours, so a nearest-
#: neighbor match within this window is unambiguous (and exact when the
#: archive itself stores whole milliseconds).
EPOCH_MATCH_TOLERANCE_NS = 50_000_000


def _nearest_line_pair(
    index: tuple[np.ndarray, list[tuple[str, str]]], target_ns: int
) -> tuple[str, str] | None:
    """(line1, line2) of the archive epoch nearest ``target_ns`` (or None)."""
    epochs_ns, pairs = index
    if len(epochs_ns) == 0:
        return None
    position = int(np.searchsorted(epochs_ns, target_ns))
    best = None
    for j in (position - 1, position):
        if 0 <= j < len(epochs_ns) and abs(int(epochs_ns[j]) - target_ns) <= EPOCH_MATCH_TOLERANCE_NS:
            if best is None or abs(int(epochs_ns[j]) - target_ns) < abs(int(epochs_ns[best]) - target_ns):
                best = j
    return pairs[best] if best is not None else None


def per_event_comparison(response: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    """Recompute every event's delta SMA under both conventions.

    ``response`` needs ``annotation_id``, ``sat_id``,
    ``tle_nearest_before_utc``, ``tle_nearest_after_utc`` and the published
    ``delta_sma_km``.  Status per event: ``compared``; ``no_published_delta``
    (missing delta_sma_km); ``tle_archive_missing`` (no raw TLE directory);
    ``tle_epoch_not_found`` (bracketing epoch absent from the archive).
    """
    cache: dict[str, tuple[np.ndarray, list[tuple[str, str]]]] = {}
    rows = []
    for _, event in response.iterrows():
        sat_id = str(event["sat_id"])
        published = pd.to_numeric(pd.Series([event.get("delta_sma_km")]), errors="coerce").iloc[0]
        row = {
            "annotation_id": event.get("annotation_id", ""),
            "sat_id": sat_id,
            "tle_nearest_before_utc": event.get("tle_nearest_before_utc", ""),
            "tle_nearest_after_utc": event.get("tle_nearest_after_utc", ""),
            "delta_sma_km_published": published,
            "delta_sma_km_kepler_recomputed": np.nan,
            "delta_sma_km_sgp4_a": np.nan,
            "difference_kepler_vs_sgp4_m": np.nan,
            "status": "",
        }
        if pd.isna(published):
            row["status"] = "no_published_delta"
            rows.append(row)
            continue
        if sat_id not in cache:
            cache[sat_id] = _tle_line_index(Path(raw_root), sat_id)
        index = cache[sat_id]
        if len(index[0]) == 0:
            row["status"] = "tle_archive_missing"
            rows.append(row)
            continue
        try:
            before_ns = int(pd.Timestamp(event["tle_nearest_before_utc"]).value)
            after_ns = int(pd.Timestamp(event["tle_nearest_after_utc"]).value)
        except (TypeError, ValueError):
            row["status"] = "tle_epoch_not_found"
            rows.append(row)
            continue
        before_pair = _nearest_line_pair(index, before_ns)
        after_pair = _nearest_line_pair(index, after_ns)
        if before_pair is None or after_pair is None:
            row["status"] = "tle_epoch_not_found"
            rows.append(row)
            continue
        line1_before, line2_before = before_pair
        line1_after, line2_after = after_pair
        try:
            da_sgp4 = delta_sma_sgp4_km(line1_before, line2_before, line1_after, line2_after)
        except Exception:
            row["status"] = "tle_epoch_not_found"
            rows.append(row)
            continue
        # Kepler recomputation from the same raw lines keeps the published
        # path auditable (same parser elements, independent of the CSV).
        n_before = _mean_motion_rad_per_min(line1_before, line2_before)
        n_after = _mean_motion_rad_per_min(line1_after, line2_after)
        da_kepler = (
            kepler_sma_km_from_mean_motion_rad_per_min(n_after)
            - kepler_sma_km_from_mean_motion_rad_per_min(n_before)
        )
        row["delta_sma_km_sgp4_a"] = da_sgp4
        row["delta_sma_km_kepler_recomputed"] = da_kepler
        row["difference_kepler_vs_sgp4_m"] = (da_kepler - da_sgp4) * 1000.0
        row["status"] = "compared"
        rows.append(row)
    return pd.DataFrame(rows, columns=PER_EVENT_COLUMNS)


def _mean_motion_rad_per_min(line1: str, line2: str) -> float:
    """Mean motion (rad/min) parsed from the TLE line-2 fixed-width field."""
    # Columns 53-63 (1-based) of line 2: mean motion in revolutions/day.
    rev_per_day = float(line2[52:63])
    return rev_per_day * 2.0 * np.pi / 1440.0


def comparison_distribution(per_event: pd.DataFrame) -> pd.DataFrame:
    """Per-target + ALL difference distribution of the two delta-a conventions."""
    rows = []
    groups = [(sat_id, group) for sat_id, group in per_event.groupby("sat_id", sort=True)]
    groups.append(("ALL", per_event))
    for sat_id, group in groups:
        compared = group[group["status"] == "compared"]
        diffs = pd.to_numeric(
            compared["difference_kepler_vs_sgp4_m"] if "difference_kepler_vs_sgp4_m" in compared.columns else pd.Series(dtype=float),
            errors="coerce",
        ).dropna().abs()
        deltas = pd.to_numeric(
            compared["delta_sma_km_kepler_recomputed"] if "delta_sma_km_kepler_recomputed" in compared.columns else pd.Series(dtype=float),
            errors="coerce",
        ).dropna().abs() * 1000.0
        rows.append(
            {
                "sat_id": sat_id,
                "n_events": len(group),
                "n_compared": len(diffs),
                "n_missing": int(len(group) - len(diffs)),
                "median_abs_difference_m": round(float(diffs.median()), 6) if len(diffs) else "",
                "mean_abs_difference_m": round(float(diffs.mean()), 6) if len(diffs) else "",
                "p95_abs_difference_m": round(float(diffs.quantile(0.95)), 6) if len(diffs) else "",
                "max_abs_difference_m": round(float(diffs.max()), 6) if len(diffs) else "",
                "median_abs_delta_sma_m": round(float(deltas.median()), 3) if len(deltas) else "",
                "sub_meter_fraction": round(float((diffs < 1.0).mean()), 4) if len(diffs) else "",
            }
        )
    return pd.DataFrame(rows, columns=DISTRIBUTION_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-table", default=str(DEFAULT_RESPONSE_TABLE))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-event-output", default=str(DEFAULT_PER_EVENT_OUTPUT))
    args = parser.parse_args()

    response = pd.read_csv(resolve_repo_path(args.response_table))
    raw_root = resolve_repo_path(args.raw_root)
    per_event = per_event_comparison(response, raw_root)
    per_event_path = resolve_repo_path(args.per_event_output)
    ensure_directory(per_event_path.parent)
    per_event.to_csv(per_event_path, index=False)

    distribution = comparison_distribution(per_event)
    out = resolve_repo_path(args.output)
    ensure_directory(out.parent)
    distribution.to_csv(out, index=False)

    all_row = distribution[distribution["sat_id"] == "ALL"]
    if not all_row.empty:
        row = all_row.iloc[0]
        print(
            f"wrote {out} + {per_event_path} compared={row['n_compared']}/{row['n_events']} "
            f"median_abs_difference={row['median_abs_difference_m']} m "
            f"max_abs_difference={row['max_abs_difference_m']} m "
            f"sub_meter_fraction={row['sub_meter_fraction']}",
            flush=True,
        )
    else:
        print(f"wrote {out} (empty: no events compared)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
