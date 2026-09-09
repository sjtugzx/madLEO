"""Distribution-completeness validation of the released evidence snapshots (Q6).

Evidence snapshots are only useful if they are free of sampling bias. These
checks quantify, per target, whether the released TLE, orbit, and SLR
snapshots cover the full mission with the expected distributions:

- TLE element distributions (Q6.1/G1-G2): inclination, eccentricity, and SMA
  altitude cluster tightly at the mission orbit, with epoch coverage over the
  full mission period. The release TLE schema carries no RAAN field, so
  orbital-plane coverage is assessed through temporal coverage instead.
- Orbit state distributions (Q6.2/G3): radius modulus range, cadence, and
  gap statistics per orbit product.
- SLR normal-point distributions (Q6.3/G4): sigma_m and num_returns per
  target.
- Tier-distribution equivalence (Q6.4/G5): KS tests of per-event TLE element
  distributions across confidence tiers. Tiers encode evidence coverage, not
  data quality, so the element distributions of tier A/B/C events should be
  statistically indistinguishable.

Reads the released Parquet snapshots (data/release/mad-leo/evidence/) and
writes four summary tables to results/tables/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.experiment_params import EARTH_RADIUS_KM

EVIDENCE = REPO_ROOT / "dataset" / "mission_reported" / "evidence"
INTERIM = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"
TABLES = REPO_ROOT / "results" / "tables"

MU_EARTH_KM3_S2 = 398600.4418
# EARTH_RADIUS_KM is imported from benchmarking.experiment_params (P4 single
# source).  The former KS ``equivalent`` verdict column was a deprecated KS
# misuse (a high p-value does not show equivalence); the pre-registered TOST
# tables (analyzers.tv_hardening) replaced it, so the KS output keeps only
# the descriptive statistic/p-value columns.
ELEMENTS = ("inclination", "eccentricity", "sma_altitude")


def _iter_parquets(root: Path):
    for path in sorted(Path(root).glob("*.parquet")):
        yield path.stem, pd.read_parquet(path)


def _sma_altitude_km(mean_motion_rad_per_min: pd.Series) -> pd.Series:
    n_per_s = mean_motion_rad_per_min / 60.0
    return (MU_EARTH_KM3_S2 / n_per_s**2) ** (1 / 3) - EARTH_RADIUS_KM


def _coverage_fields(epochs: pd.Series) -> dict[str, float | int | str]:
    ordered = epochs.sort_values()
    gaps_days = ordered.diff().dt.total_seconds().dropna() / 86400.0
    return {
        "first_epoch_utc": ordered.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_epoch_utc": ordered.iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mission_years_covered": int(ordered.dt.year.nunique()),
        "cadence_median_hours": round(float(gaps_days.median() * 24.0), 3),
        "max_gap_days": round(float(gaps_days.max()), 2),
    }


def tle_element_distribution_summary(tle_root: Path = EVIDENCE / "tle") -> pd.DataFrame:
    """Per-target TLE element distribution and mission-coverage statistics."""
    rows = []
    for sat_id, frame in _iter_parquets(tle_root):
        inclination_deg = np.degrees(frame["inclination_rad"])
        altitude_km = _sma_altitude_km(frame["mean_motion_rad_per_min"])
        row = {
            "sat_id": sat_id,
            "epoch_count": len(frame),
            **{
                f"inclination_{stat}_deg": value
                for stat, value in (
                    ("median", round(float(inclination_deg.median()), 4)),
                    ("iqr", round(float(inclination_deg.quantile(0.75) - inclination_deg.quantile(0.25)), 4)),
                )
            },
            "eccentricity_median": round(float(frame["eccentricity"].median()), 6),
            "eccentricity_iqr": round(float(frame["eccentricity"].quantile(0.75) - frame["eccentricity"].quantile(0.25)), 6),
            "sma_altitude_median_km": round(float(altitude_km.median()), 2),
            "sma_altitude_iqr_km": round(float(altitude_km.quantile(0.75) - altitude_km.quantile(0.25)), 2),
            **_coverage_fields(frame["epoch"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def orbit_state_distribution_summary(orbit_root: Path = EVIDENCE / "orbit") -> pd.DataFrame:
    """Per-target orbit-product state distribution, cadence, and gap statistics."""
    rows = []
    for sat_id, frame in _iter_parquets(orbit_root):
        radius_km = np.sqrt(frame["x_m"] ** 2 + frame["y_m"] ** 2 + frame["z_m"] ** 2) / 1000.0
        rows.append(
            {
                "sat_id": sat_id,
                "state_count": len(frame),
                "radius_median_km": round(float(radius_km.median()), 3),
                "radius_min_km": round(float(radius_km.min()), 3),
                "radius_max_km": round(float(radius_km.max()), 3),
                **_coverage_fields(frame["epoch"]),
            }
        )
    return pd.DataFrame(rows)


def slr_distribution_summary(slr_root: Path = EVIDENCE / "slr") -> pd.DataFrame:
    """Per-target SLR normal-point precision and return-count distributions.

    Rows flagged ``qc_status != 'ok'`` (implausible range / cross-target
    contamination; 20 rows in the shipped release) are excluded from the
    statistics and counted in ``qc_flagged_rows``.
    """
    rows = []
    for sat_id, frame in _iter_parquets(slr_root):
        flagged = 0
        if "qc_status" in frame.columns:
            flagged = int((frame["qc_status"] != "ok").sum())
            frame = frame[frame["qc_status"] == "ok"]
        sigma_mm = frame["sigma_m"] * 1000.0
        rows.append(
            {
                "sat_id": sat_id,
                "normal_point_count": len(frame),
                "qc_flagged_rows": flagged,
                "station_count": int(frame["station_id"].nunique()),
                "sigma_median_mm": round(float(sigma_mm.median()), 3),
                "sigma_iqr_mm": round(float(sigma_mm.quantile(0.75) - sigma_mm.quantile(0.25)), 3),
                "num_returns_median": round(float(frame["num_returns"].median()), 1),
                **_coverage_fields(frame["epoch"]),
            }
        )
    return pd.DataFrame(rows)


def _event_elements(tle: pd.DataFrame, event_times: pd.Series) -> pd.DataFrame:
    """TLE elements at the nearest catalog epoch for each event time.

    The returned frame carries ``_event_row`` (the position of the event in
    ``event_times``) so callers can attach per-event metadata by KEY merge
    instead of positional truncation (REVIEW_FINDINGS C15: a dropped row
    used to shift every subsequent tier assignment silently).
    """
    ordered = tle.sort_values("epoch").reset_index(drop=True)
    epoch_values = ordered["epoch"].values
    idx = np.searchsorted(epoch_values, event_times.values)
    positions = []
    for i, value in enumerate(event_times.values):
        best = None
        for j in (idx[i] - 1, idx[i]):
            if 0 <= j < len(epoch_values):
                if best is None or abs(epoch_values[j] - value) < abs(epoch_values[best] - value):
                    best = j
        positions.append((i, best))
    picked = ordered.iloc[[p for _, p in positions if p is not None]]
    picked = picked.assign(_event_row=[i for i, p in positions if p is not None])
    return pd.DataFrame(
        {
            "_event_row": picked["_event_row"].to_numpy(),
            "inclination": np.degrees(picked["inclination_rad"].to_numpy()),
            "eccentricity": picked["eccentricity"].to_numpy(),
            "sma_altitude": _sma_altitude_km(picked["mean_motion_rad_per_min"]).to_numpy(),
        }
    )


def tier_distribution_equivalence(
    tle_root: Path, annotations: pd.DataFrame, min_group: int = 5
) -> pd.DataFrame:
    """Within-target KS equivalence of per-event TLE element distributions.

    Tiers are not uniformly distributed across satellites (modern missions are
    mostly tier A, sparse-archive missions mostly tier C), so a pooled test
    would confound orbital regime with tier. The equivalence claim — tiers
    encode evidence coverage, not data quality — is therefore tested within
    each target that has at least two tiers with ``min_group`` events each.
    """
    rows = []
    for sat_id, group in annotations.groupby("sat_id"):
        tle_path = Path(tle_root) / f"{sat_id}.parquet"
        if not tle_path.exists():
            continue
        # format="ISO8601": event times mix whole-second and millisecond
        # precision strings; pandas' single-format inference rejects the mix.
        elements = _event_elements(
            pd.read_parquet(tle_path),
            pd.to_datetime(group["event_time_utc"], utc=True, format="ISO8601"),
        )
        # C15: attach tiers by KEY merge on the event row, never by positional
        # truncation -- a dropped element row used to shift every subsequent
        # tier assignment silently.
        tier_frame = pd.DataFrame(
            {
                "_event_row": np.arange(len(group)),
                "confidence_tier": group["confidence_tier"].to_numpy(),
            }
        )
        elements = elements.drop(columns=["confidence_tier"], errors="ignore").merge(
            tier_frame, on="_event_row", how="left"
        )
        tiers = sorted(elements["confidence_tier"].dropna().unique())
        pairs = [(left, right) for i, left in enumerate(tiers) for right in tiers[i + 1 :]]
        for element in ELEMENTS:
            for left, right in pairs:
                a = elements.loc[elements["confidence_tier"] == left, element].dropna()
                b = elements.loc[elements["confidence_tier"] == right, element].dropna()
                if len(a) < min_group or len(b) < min_group:
                    continue
                statistic, pvalue = stats.ks_2samp(a, b)
                rows.append(
                    {
                        "sat_id": sat_id,
                        "element": element,
                        "tier_pair": f"{left}_vs_{right}",
                        "n_left": len(a),
                        "n_right": len(b),
                        "median_left": round(float(a.median()), 4),
                        "median_right": round(float(b.median()), 4),
                        "median_abs_delta": round(float(abs(a.median() - b.median())), 4),
                        "ks_statistic": round(float(statistic), 4),
                        "ks_pvalue": round(float(pvalue), 4),
                    }
                )
    result = pd.DataFrame(rows)
    summary_rows = []
    for (element, tier_pair), group in result.groupby(["element", "tier_pair"]):
        summary_rows.append(
            {
                "sat_id": "ALL_WITHIN_TARGET",
                "element": element,
                "tier_pair": tier_pair,
                "n_left": int(group["n_left"].sum()),
                "n_right": int(group["n_right"].sum()),
                "median_left": "",
                "median_right": "",
                "median_abs_delta": round(float(group["median_abs_delta"].median()), 4),
                "ks_statistic": round(float(group["ks_statistic"].median()), 4),
                "ks_pvalue": round(float(group["ks_pvalue"].median()), 4),
            }
        )
    return pd.concat([result, pd.DataFrame(summary_rows)], ignore_index=True)


def _load_annotations(interim_root: Path = INTERIM) -> pd.DataFrame:
    """Per-event (annotation_id, sat_id, event_time_utc, confidence_tier)."""
    frames = []
    for ann_path in sorted(Path(interim_root).glob("*/maneuver_annotations.csv")):
        sat_id = ann_path.parent.name
        windows_path = ann_path.parent / "annotated_event_windows.csv"
        if not windows_path.exists():
            continue
        annotations = pd.read_csv(ann_path, usecols=["annotation_id", "sat_id", "event_time_utc"])
        windows = pd.read_csv(windows_path, usecols=["annotation_id", "confidence_tier"])
        frames.append(annotations.merge(windows, on="annotation_id", how="inner"))
    if not frames:
        return pd.DataFrame(columns=["annotation_id", "sat_id", "event_time_utc", "confidence_tier"])
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", default=str(EVIDENCE))
    parser.add_argument("--output-dir", default=str(TABLES))
    args = parser.parse_args()
    evidence_root = Path(resolve_repo_path(args.evidence_root))
    output = ensure_directory(resolve_repo_path(args.output_dir))

    tle_element_distribution_summary(evidence_root / "tle").to_csv(output / "tle_element_distribution_summary.csv", index=False)
    print("wrote tle_element_distribution_summary.csv", flush=True)
    orbit_state_distribution_summary(evidence_root / "orbit").to_csv(output / "orbit_state_distribution_summary.csv", index=False)
    print("wrote orbit_state_distribution_summary.csv", flush=True)
    slr_distribution_summary(evidence_root / "slr").to_csv(output / "slr_distribution_summary.csv", index=False)
    print("wrote slr_distribution_summary.csv", flush=True)
    annotations = _load_annotations()
    if annotations.empty:
        # Tier equivalence needs the pipeline workspace (interim normalized
        # annotations); the three distribution summaries above were produced
        # from the released evidence snapshots alone.
        print(
            "skip: tier_distribution_equivalence needs the pipeline workspace "
            "(run the data pipeline first); distribution summaries were written",
            flush=True,
        )
        return 0
    tier_distribution_equivalence(evidence_root / "tle", annotations).to_csv(
        output / "tier_distribution_equivalence.csv", index=False
    )
    print("wrote tier_distribution_equivalence.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
