"""Align mission-reported maneuver windows with source-response evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarking.config import ensure_directory, resolve_repo_path
from benchmarking.normalization import to_utc_iso_series


ALIGNMENT_COLUMNS = [
    "annotation_id",
    "sat_id",
    "window_start_utc",
    "window_end_utc",
    "tle_response_status",
    "pod_response_status",
    "precise_orbit_response_status",
    "slr_audit_status",
    "source_conflict_status",
    "quality_flags",
    "evidence_ids",
    "metrics_json",
    "event_label",
    "confidence_tier",
]
TLE_FIELDS = [
    "mean_motion_rad_per_min",
    "eccentricity",
    "inclination_rad",
    "raan_rad",
    "argument_of_perigee_rad",
    "mean_anomaly_rad",
    "bstar",
]
ANGULAR_TLE_FIELDS = {"inclination_rad", "raan_rad", "argument_of_perigee_rad", "mean_anomaly_rad"}


def _parse_utc(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _iso(value: object) -> str:
    return to_utc_iso_series(pd.Series([value])).iloc[0]


def _source_slice(df: pd.DataFrame | None, sat_id: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()
    if "epoch" not in result.columns:
        return pd.DataFrame()
    result["epoch"] = pd.to_datetime(result["epoch"], utc=True)
    if "sat_id" in result.columns:
        result = result[result["sat_id"].astype(str) == sat_id]
    return result.sort_values("epoch").reset_index(drop=True)


def _angle_delta(after: float, before: float) -> tuple[float, bool]:
    raw = float(after) - float(before)
    wrapped = (raw + math.pi) % (2 * math.pi) - math.pi
    return wrapped, not math.isclose(raw, wrapped, rel_tol=0.0, abs_tol=1e-12)


def _tle_response(tle_df: pd.DataFrame | None, sat_id: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, dict, list[str]]:
    tle = _source_slice(tle_df, sat_id)
    if tle.empty:
        return "missing_source", {}, ["missing_tle"]

    before = tle[tle["epoch"] <= start]
    after = tle[tle["epoch"] >= end]
    if before.empty or after.empty:
        return "insufficient_coverage", {"sample_count": int(len(tle))}, ["tle_insufficient_coverage"]

    nearest_before = before.iloc[-1]
    nearest_after = after.iloc[0]
    metrics: dict[str, Any] = {
        "nearest_before_epoch": _iso(nearest_before["epoch"]),
        "nearest_after_epoch": _iso(nearest_after["epoch"]),
    }
    response_detected = False
    angular_unwrap_applied = False
    for field in TLE_FIELDS:
        if field not in tle.columns:
            continue
        before_value = nearest_before.get(field)
        after_value = nearest_after.get(field)
        if pd.isna(before_value) or pd.isna(after_value):
            continue
        if field in ANGULAR_TLE_FIELDS:
            delta, unwrapped = _angle_delta(float(after_value), float(before_value))
            angular_unwrap_applied = angular_unwrap_applied or unwrapped
        else:
            delta = float(after_value) - float(before_value)
        metrics[f"delta_{field}"] = round(float(delta), 12)
        if abs(float(delta)) > 1e-9:
            response_detected = True
    metrics["angular_unwrap_applied"] = angular_unwrap_applied
    return ("response_detected" if response_detected else "coverage_only"), metrics, []


def _state_response(
    df: pd.DataFrame | None,
    sat_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    missing_flag: str,
    available_when_missing: bool = False,
) -> tuple[str, dict, list[str]]:
    source = _source_slice(df, sat_id)
    if source.empty:
        status = "not_available" if available_when_missing else "missing_source"
        return status, {}, ([] if available_when_missing else [missing_flag])
    response_start = start - pd.Timedelta(hours=12)
    response_end = end + pd.Timedelta(hours=12)
    during = source[(source["epoch"] >= response_start) & (source["epoch"] <= response_end)]
    metrics = {"sample_count": int(len(during))}
    if during.empty:
        source_name = missing_flag.removeprefix("missing_")
        return "insufficient_coverage", metrics, [f"{source_name}_insufficient_coverage"]

    response_detected = False
    if {"x_m", "y_m", "z_m"}.issubset(during.columns) and len(during) >= 2:
        first = during.iloc[0][["x_m", "y_m", "z_m"]].astype(float).to_numpy()
        last = during.iloc[-1][["x_m", "y_m", "z_m"]].astype(float).to_numpy()
        displacement = float(np.linalg.norm(last - first))
        metrics["position_displacement_m"] = round(displacement, 6)
        response_detected = displacement > 0
    return ("response_detected" if response_detected or len(during) > 1 else "coverage_only"), metrics, []


def _slr_audit(slr_df: pd.DataFrame | None, sat_id: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, dict, list[str]]:
    slr = _source_slice(slr_df, sat_id)
    if slr.empty:
        return "insufficient_coverage", {}, ["slr_insufficient_coverage"]
    before = slr[(slr["epoch"] >= start - pd.Timedelta(hours=12)) & (slr["epoch"] < start)]
    during = slr[(slr["epoch"] >= start) & (slr["epoch"] <= end)]
    after = slr[(slr["epoch"] > end) & (slr["epoch"] <= end + pd.Timedelta(hours=12))]
    station_count = int(slr["station_id"].dropna().nunique()) if "station_id" in slr.columns else 0
    metrics = {
        "obs_before": int(len(before)),
        "obs_during": int(len(during)),
        "obs_after": int(len(after)),
        "station_count": station_count,
        "residual_available": bool("residual_m" in slr.columns and slr["residual_m"].notna().any()),
    }
    if len(before) > 0 and len(during) > 0 and len(after) > 0:
        return "support_available", metrics, []
    return "insufficient_coverage", metrics, ["slr_insufficient_coverage"]


def classify_public_event_status(
    tle_response_status: str,
    pod_response_status: str,
    precise_orbit_response_status: str,
    slr_audit_status: str,
) -> dict:
    """Classify mission-reported event status without making SLR mandatory."""
    quality_flags: list[str] = []
    for source, status in {
        "tle": tle_response_status,
        "pod": pod_response_status,
        "precise_orbit": precise_orbit_response_status,
        "slr": slr_audit_status,
    }.items():
        if status == "conflict":
            quality_flags.append(f"{source}_conflict")
    if slr_audit_status == "insufficient_coverage":
        quality_flags.append("slr_insufficient_coverage")

    if any(flag.endswith("_conflict") for flag in quality_flags):
        return {
            "event_label": "ignore",
            "confidence_tier": "audit_only",
            "source_conflict_status": "source_conflict",
            "quality_flags": quality_flags,
        }

    if tle_response_status == "missing_source":
        quality_flags.append("missing_tle")
    elif tle_response_status == "insufficient_coverage":
        quality_flags.append("tle_insufficient_coverage")
    if pod_response_status == "missing_source" and precise_orbit_response_status in {"missing_source", "not_available"}:
        quality_flags.append("missing_pod")
    elif pod_response_status == "insufficient_coverage" and precise_orbit_response_status in {"missing_source", "not_available"}:
        quality_flags.append("pod_insufficient_coverage")

    has_tle_response = tle_response_status in {"response_detected", "coverage_only"}
    has_orbit_response = pod_response_status == "response_detected" or precise_orbit_response_status == "response_detected"
    if has_tle_response and has_orbit_response:
        return {
            "event_label": "event",
            "confidence_tier": "A" if slr_audit_status == "support_available" else "B",
            "source_conflict_status": "none",
            "quality_flags": quality_flags,
        }
    return {
        "event_label": "event",
        "confidence_tier": "C",
        "source_conflict_status": "coverage_insufficient",
        "quality_flags": quality_flags,
    }


def align_maneuver_evidence(
    event_windows: pd.DataFrame,
    tle_df: pd.DataFrame | None = None,
    pod_df: pd.DataFrame | None = None,
    precise_orbit_df: pd.DataFrame | None = None,
    slr_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return maneuver evidence alignment rows for mission-reported windows."""
    rows = []
    for event in event_windows.itertuples(index=False):
        start = _parse_utc(event.window_start_utc)
        end = _parse_utc(event.window_end_utc)
        sat_id = str(event.sat_id)
        tle_status, tle_metrics, tle_flags = _tle_response(tle_df, sat_id, start, end)
        pod_status, pod_metrics, pod_flags = _state_response(pod_df, sat_id, start, end, "missing_pod")
        precise_status, precise_metrics, precise_flags = _state_response(
            precise_orbit_df,
            sat_id,
            start,
            end,
            "missing_precise_orbit",
            available_when_missing=True,
        )
        slr_status, slr_metrics, slr_flags = _slr_audit(slr_df, sat_id, start, end)
        classification = classify_public_event_status(tle_status, pod_status, precise_status, slr_status)
        flags = sorted(set(tle_flags + pod_flags + precise_flags + slr_flags + classification["quality_flags"]))
        metrics = {
            "tle": tle_metrics,
            "pod": pod_metrics,
            "precise_orbit": precise_metrics,
            "slr": slr_metrics,
        }
        rows.append(
            {
                "annotation_id": event.annotation_id,
                "sat_id": sat_id,
                "window_start_utc": event.window_start_utc,
                "window_end_utc": event.window_end_utc,
                "tle_response_status": tle_status,
                "pod_response_status": pod_status,
                "precise_orbit_response_status": precise_status,
                "slr_audit_status": slr_status,
                "source_conflict_status": classification["source_conflict_status"],
                "quality_flags": ",".join(flags),
                "evidence_ids": "",
                "metrics_json": json.dumps(metrics, ensure_ascii=True, sort_keys=True),
                "event_label": classification["event_label"],
                "confidence_tier": classification["confidence_tier"],
            }
        )
    return pd.DataFrame(rows, columns=ALIGNMENT_COLUMNS)


def write_maneuver_evidence_alignment(alignment: pd.DataFrame, output_root: str | Path, sat_id: str) -> Path:
    """Write a target-level maneuver evidence alignment table."""
    target_dir = ensure_directory(Path(output_root) / sat_id)
    output_path = target_dir / "maneuver_evidence_alignment.csv"
    alignment.to_csv(output_path, index=False)
    return output_path


def _read_csv_if_present(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    resolved = resolve_repo_path(path)
    if resolved is None or not resolved.exists():
        return None
    return pd.read_csv(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align maneuver event windows with source evidence")
    parser.add_argument("--event-windows", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sat-id", required=True)
    parser.add_argument("--tle")
    parser.add_argument("--pod")
    parser.add_argument("--precise-orbit")
    parser.add_argument("--slr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_windows = pd.read_csv(resolve_repo_path(args.event_windows))
    alignment = align_maneuver_evidence(
        event_windows,
        tle_df=_read_csv_if_present(args.tle),
        pod_df=_read_csv_if_present(args.pod),
        precise_orbit_df=_read_csv_if_present(args.precise_orbit),
        slr_df=_read_csv_if_present(args.slr),
    )
    output_path = write_maneuver_evidence_alignment(alignment, resolve_repo_path(args.output_root), args.sat_id)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
