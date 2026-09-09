"""Starlink operational-subset distribution and consistency validation (Q4).

I3/Q4.2 Shell distribution: the released TLE snapshot must reproduce the
known Starlink shell structure (inclination bands near 53.0/53.2/70/97.6
deg), with per-shell altitude and plane counts.

I4/Q4.3 Orbital-plane structure: RAAN histogram plane counts per shell
(folded into the shell summary table).

I5/Q4.4 Ephemeris state distributions: radius/speed modulus ranges and
60-second cadence compliance of the operator ephemerides.

I6/Q4.5 TLE-vs-ephemeris consistency (frame-consistent version): a
fixed-seed sample of satellites present in both sources; the nearest prior
catalog TLE is propagated with SGP4 to each sampled ephemeris epoch and
converted TEME -> GCRS, and the ephemeris state is transformed from the
operator's MEME convention into the same GCRS frame before differencing.
The operator files specify MEME without an equinox epoch; the of-date
reading is excluded because the J2000->date precession rotation displaces
LEO positions by tens of km, orders above the observed sub-km medians, so
the product is treated as a mean frame of fixed equinox (MEME of J2000,
offset from GCRS at the metre level). A companion sensitivity table reports
the residual under four admissible frame chains (legacy GCRS-vs-raw,
MEME-J2000, MEME-of-date, direct TEME) so the frame assumption is
inspectable rather than implicit. Residuals still include SGP4 propagation
error, so this remains an upper-bound consistency audit, not an
orbit-accuracy assessment.

Reads the released operational/starlink Parquet tables plus the raw
Space-Track archive day files (needed for SGP4 element provenance) and
writes the summary tables to ``experiments/starlink/`` (analysis artifacts
tracked with the code, not part of the released dataset).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.utils.iers
from astropy.coordinates import GCRS, TEME, CartesianRepresentation, PrecessedGeocentric
from astropy.time import Time
from sgp4.api import WGS72, Satrec, jday

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.experiment_params import (
    EARTH_RADIUS_KM,
    SHELL_TOLERANCE_DEG,
    STARLINK_SAMPLE_SEED as SAMPLE_SEED,
    STARLINK_TLE_MAX_AGE_HOURS,
)

astropy.utils.iers.conf.auto_download = False
astropy.utils.iers.conf.iers_degraded_accuracy = "ignore"

STARLINK_DIR = REPO_ROOT / "dataset" / "operational" / "starlink"
RAW_TLE_DIR = REPO_ROOT / "data" / "raw" / "operational_constellation" / "starlink" / "tle_archive" / "full_data"
TABLES = REPO_ROOT / "experiments" / "starlink"

MU_EARTH_KM3_S2 = 398600.4418
SHELL_CENTERS = {"43.0": 43.0, "53.0": 53.0, "53.2": 53.2, "70": 70.0, "97.6": 97.6}
RAAN_BIN_DEG = 5.0
RAAN_PLANE_MIN_SHARE = 0.05
# SAMPLE_SEED, EARTH_RADIUS_KM, and SHELL_TOLERANCE_DEG are imported from
# benchmarking.experiment_params (P4 single source).
SAMPLE_SATELLITES = 8
COMPARE_STEP_SECONDS = 600
MAX_TLE_AGE_SECONDS = STARLINK_TLE_MAX_AGE_HOURS * 3600.0
SGP4_EPOCH_JD_OFFSET = 2433281.5  # sgp4init expects days since 1950 Jan 0

TLE_RAW_COLUMNS = [
    "Satellite_Number", "Satellite_Name", "Timestamp", "Mean_Anomaly",
    "Right_Ascension_of_Node", "Argument_of_Perigee", "Eccentricity",
    "Inclination", "Mean_Motion", "First_Derivative_of_Mean_Motion",
    "Second_Derivative_of_Mean_Motion", "BSTAR_Drag_Term",
]


def assign_shell(inclination_deg: float) -> str:
    """Map an inclination to the nearest Starlink shell label."""
    nearest = min(SHELL_CENTERS.items(), key=lambda item: abs(inclination_deg - item[1]))
    return nearest[0] if abs(inclination_deg - nearest[1]) <= SHELL_TOLERANCE_DEG else "other"


def _sma_altitude_km(mean_motion_rad_per_min: pd.Series) -> pd.Series:
    n_per_s = mean_motion_rad_per_min / 60.0
    return (MU_EARTH_KM3_S2 / n_per_s**2) ** (1 / 3) - EARTH_RADIUS_KM


def raan_plane_count(raan_rad: pd.Series, satellite_count: int) -> int:
    """Number of significant RAAN planes (5 deg bins holding >= 50% of the
    shell's largest bin; plane populations are roughly uniform)."""
    if satellite_count == 0 or raan_rad.empty:
        return 0
    bins = np.arange(0.0, 2 * np.pi + np.radians(RAAN_BIN_DEG), np.radians(RAAN_BIN_DEG))
    counts, _ = np.histogram(raan_rad.to_numpy() % (2 * np.pi), bins=bins)
    return int((counts >= max(1, 0.5 * counts.max())).sum())


def representative_tle_elements(tle: pd.DataFrame) -> pd.DataFrame:
    """Latest element set per satellite, with shell labels and derived fields."""
    latest = tle.sort_values("epoch").groupby("sat_id", as_index=False).last()
    latest = latest.assign(
        inclination_deg=np.degrees(latest["inclination_rad"]),
        sma_altitude_km=_sma_altitude_km(latest["mean_motion_rad_per_min"]),
        raan_deg=np.degrees(latest["raan_rad"]) % 360.0,
    )
    latest["shell"] = latest["inclination_deg"].map(assign_shell)
    return latest


def tle_shell_distribution(tle: pd.DataFrame) -> pd.DataFrame:
    """Per-shell satellite counts and element distributions (I3 + I4)."""
    latest = representative_tle_elements(tle)
    rows = []
    for shell, group in latest.groupby("shell"):
        rows.append(
            {
                "shell": shell,
                "satellite_count": len(group),
                "epoch_count": int(tle[tle["sat_id"].isin(group["sat_id"])].shape[0]),
                "inclination_median_deg": round(float(group["inclination_deg"].median()), 4),
                "sma_altitude_median_km": round(float(group["sma_altitude_km"].median()), 2),
                "sma_altitude_iqr_km": round(
                    float(group["sma_altitude_km"].quantile(0.75) - group["sma_altitude_km"].quantile(0.25)), 2
                ),
                "eccentricity_median": round(float(group["eccentricity"].median()), 6),
                "raan_plane_count": raan_plane_count(group["raan_rad"], len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("shell").reset_index(drop=True)


def ephemeris_state_distribution(parquet_path: Path, shell_by_sat: dict[str, str], batch_rows: int = 2_000_000) -> pd.DataFrame:
    """Radius/speed distributions and 60 s cadence compliance (I5).

    States flagged ``quality_flag == 'below_surface'`` (operator-published
    predictions of actively deorbiting objects that continue below the
    surface; 5,861 rows in the shipped release) are excluded from the
    radius/speed statistics and counted in ``qc_flagged_rows``;
    ``state_count`` stays the total shipped rows and cadence accounting
    covers every state (publishing cadence is independent of the position
    validity).
    """
    import pyarrow.parquet as pq

    scopes: dict[str, dict[str, list]] = {}

    def _bucket(scope: str) -> dict[str, list]:
        return scopes.setdefault(scope, {"radius": [], "speed": [], "sats": set(), "flagged": 0, "total": 0})

    last_epoch: dict[str, pd.Timestamp] = {}
    cadence_ok: dict[str, int] = {}
    cadence_total: dict[str, int] = {}

    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=["sat_id", "epoch", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "quality_flag"]):
        frame = batch.to_pandas()
        flagged = frame["quality_flag"].eq("below_surface").to_numpy()
        good = ~flagged
        radius = np.sqrt(frame["x_m"] ** 2 + frame["y_m"] ** 2 + frame["z_m"] ** 2) / 1000.0
        speed = np.sqrt(frame["vx_mps"] ** 2 + frame["vy_mps"] ** 2 + frame["vz_mps"] ** 2) / 1000.0
        epochs = pd.to_datetime(frame["epoch"], utc=True)
        sat_ids = frame["sat_id"].astype(str)
        shells = sat_ids.map(shell_by_sat).fillna("other")
        for scope_name, mask in (("ALL", np.ones(len(frame), dtype=bool)),):
            bucket = _bucket(scope_name)
            bucket["radius"].append(radius.to_numpy()[mask & good])
            bucket["speed"].append(speed.to_numpy()[mask & good])
            bucket["flagged"] += int((mask & flagged).sum())
            bucket["total"] += int(mask.sum())
        for shell in ("43.0", "53.0", "53.2", "70", "97.6", "other"):
            mask = (shells == shell).to_numpy()
            if mask.any():
                bucket = _bucket(shell)
                bucket["radius"].append(radius.to_numpy()[mask & good])
                bucket["speed"].append(speed.to_numpy()[mask & good])
                bucket["flagged"] += int((mask & flagged).sum())
                bucket["total"] += int(mask.sum())
        all_sats = sat_ids.to_numpy()
        _bucket("ALL")["sats"].update(all_sats.tolist())
        shell_values = shells.to_numpy()
        for shell in ("43.0", "53.0", "53.2", "70", "97.6", "other"):
            members = all_sats[shell_values == shell]
            if members.size:
                _bucket(shell)["sats"].update(members.tolist())

        # Vectorized cadence accounting: consecutive per-sat epoch diffs within
        # the batch, with batch boundaries bridged through last_epoch.
        cadence_frame = pd.DataFrame({"sat": all_sats, "epoch": epochs.to_numpy()})
        cadence_frame = cadence_frame.sort_values(["sat", "epoch"], kind="stable")
        within_s = cadence_frame.groupby("sat", sort=False)["epoch"].diff().dt.total_seconds()
        cadence_frame["ok"] = np.isclose(within_s.to_numpy(dtype=float, na_value=np.nan), 60.0)
        cadence_frame["counted"] = within_s.notna()
        for sat, count in cadence_frame.groupby("sat")["counted"].sum().items():
            cadence_total[sat] = cadence_total.get(sat, 0) + int(count)
        for sat, count in cadence_frame.groupby("sat")["ok"].sum().items():
            cadence_ok[sat] = cadence_ok.get(sat, 0) + int(count)
        first_rows = cadence_frame.groupby("sat", sort=False).head(1)
        for sat, epoch in zip(first_rows["sat"], first_rows["epoch"]):
            if sat in last_epoch:
                cadence_total[sat] += 1
                if abs((epoch - last_epoch[sat]).total_seconds() - 60.0) < 1e-6:
                    cadence_ok[sat] += 1
        last_rows = cadence_frame.groupby("sat", sort=False).tail(1)
        for sat, epoch in zip(last_rows["sat"], last_rows["epoch"]):
            last_epoch[sat] = epoch

    rows = []
    for scope, bucket in sorted(scopes.items()):
        radius = np.concatenate(bucket["radius"])
        speed = np.concatenate(bucket["speed"])
        sats = bucket["sats"]
        ok = sum(cadence_ok.get(s, 0) for s in sats)
        total = sum(cadence_total.get(s, 0) for s in sats)
        rows.append(
            {
                "scope": scope,
                "satellite_count": len(sats),
                "state_count": int(bucket["total"]),
                "qc_flagged_rows": int(bucket["flagged"]),
                "radius_median_km": round(float(np.median(radius)), 2),
                "radius_min_km": round(float(radius.min()), 2),
                "radius_max_km": round(float(radius.max()), 2),
                "speed_median_kmps": round(float(np.median(speed)), 4),
                "cadence_60s_fraction": round(ok / total, 4) if total else "",
            }
        )
    return pd.DataFrame(rows)


def _satrec_from_row(row: dict | pd.Series) -> Satrec:
    """Build a Satrec from a Space-Track archive CSV row (elements in radians)."""
    ts = pd.Timestamp(row["Timestamp"], tz="utc")
    jd, fr = jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second + ts.microsecond / 1e6)
    sat = Satrec()
    sat.sgp4init(
        WGS72, "i", int(row["Satellite_Number"]), jd + fr - SGP4_EPOCH_JD_OFFSET,
        float(row["BSTAR_Drag_Term"]), float(row["First_Derivative_of_Mean_Motion"]),
        float(row["Second_Derivative_of_Mean_Motion"]), float(row["Eccentricity"]),
        float(row["Argument_of_Perigee"]), float(row["Inclination"]), float(row["Mean_Anomaly"]),
        float(row["Mean_Motion"]), float(row["Right_Ascension_of_Node"]),
    )
    return sat


def _teme_to_gcrs(r_km: np.ndarray, epoch: pd.Timestamp) -> np.ndarray:
    """Convert a TEME position (km) to GCRS for comparison with MEME products."""
    teme = TEME(CartesianRepresentation(r_km * u.km), obstime=Time(epoch.to_datetime64()))
    gcrs = teme.transform_to(GCRS(obstime=Time(epoch.to_datetime64())))
    return gcrs.cartesian.xyz.to_value(u.km)


def _meme_to_gcrs(pos_m: np.ndarray, epoch: pd.Timestamp, equinox: str) -> np.ndarray:
    """Transform an operator ephemeris position from an MEME interpretation
    (equinox given by ``equinox``, e.g. "J2000" or an ISO epoch) to GCRS (m).

    The of-date reading of the MEME convention is excluded by magnitude in
    the published audit: the J2000->date precession rotation displaces LEO
    positions by tens of km, incompatible with the observed sub-km medians.
    The published chain therefore treats the product as MEME of J2000,
    which sits within ~1 m of GCRS.
    """
    obstime = Time(epoch.to_datetime64())
    frame = PrecessedGeocentric(equinox=Time(equinox), obstime=obstime)
    state = frame.realize_frame(CartesianRepresentation(*(np.asarray(pos_m, dtype=float) * u.m)))
    out = state.transform_to(GCRS(obstime=obstime))
    return out.cartesian.xyz.to_value(u.m)


def tle_ephemeris_consistency(
    tle_rows: pd.DataFrame,
    ephemeris: pd.DataFrame,
    sample_count: int = SAMPLE_SATELLITES,
    seed: int = SAMPLE_SEED,
    step_seconds: int = COMPARE_STEP_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-satellite SGP4-vs-ephemeris residuals under four frame chains.

    The published residual (chain ``meme_j2000``) transforms the TLE state
    TEME->GCRS and the ephemeris MEME(J2000)->GCRS, so both sides of the
    difference live in one frame. The companion sensitivity chains quantify
    the frame assumption: ``gcrs_vs_raw`` is the legacy mixed-frame chain,
    ``meme_of_date`` the excluded of-date reading, ``teme_direct`` the
    no-transform reading.
    """
    tle = tle_rows.copy()
    tle["sat_id"] = tle["Satellite_Number"].astype(str)
    tle["epoch_dt"] = pd.to_datetime(tle["Timestamp"], utc=True)
    eph = ephemeris.copy()
    eph["sat_id"] = eph["sat_id"].astype(str)
    eph["epoch_dt"] = pd.to_datetime(eph["epoch"], utc=True)

    shared = sorted(set(tle["sat_id"]) & set(eph["sat_id"]))
    if not shared:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    sampled = sorted(rng.choice(shared, size=min(sample_count, len(shared)), replace=False).tolist())

    chains = ("gcrs_vs_raw", "meme_j2000", "meme_of_date", "teme_direct")
    rows = []
    sample_rows: list[dict] = []
    sens_rows: list[dict] = []
    for sat_id in sampled:
        sat_tle = tle[tle["sat_id"] == sat_id].sort_values("epoch_dt")
        sat_eph = eph[eph["sat_id"] == sat_id].sort_values("epoch_dt")
        if sat_eph.empty:
            continue
        offsets_s = (sat_eph["epoch_dt"] - sat_eph["epoch_dt"].iloc[0]).dt.total_seconds()
        sat_eph = sat_eph[(offsets_s % step_seconds) < 60.0]
        tle_epochs = sat_tle["epoch_dt"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        satrecs = [_satrec_from_row(record) for record in sat_tle.to_dict("records")]
        per_chain: dict[str, list[float]] = {chain: [] for chain in chains}
        ages_hours: list[float] = []
        for record in sat_eph.to_dict("records"):
            epoch = record["epoch_dt"]
            epoch64 = epoch.to_datetime64()
            idx = int(np.searchsorted(tle_epochs, epoch64)) - 1
            if idx < 0:
                continue
            age_s = (epoch64 - tle_epochs[idx]) / np.timedelta64(1, "s")
            if age_s > MAX_TLE_AGE_SECONDS:
                continue
            error, r_km, _ = satrecs[idx].sgp4(
                *jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1e6)
            )
            if error != 0:
                continue
            raw_m = np.array([record["x_m"], record["y_m"], record["z_m"]], dtype=float)
            teme_m = np.asarray(r_km) * 1000.0
            gcrs_m = np.asarray(_teme_to_gcrs(np.asarray(r_km), epoch)) * 1000.0
            per_chain["gcrs_vs_raw"].append(float(np.linalg.norm(gcrs_m - raw_m) / 1000.0))
            per_chain["teme_direct"].append(float(np.linalg.norm(teme_m - raw_m) / 1000.0))
            per_chain["meme_j2000"].append(
                float(np.linalg.norm(gcrs_m - _meme_to_gcrs(raw_m, epoch, "J2000")) / 1000.0)
            )
            per_chain["meme_of_date"].append(
                float(np.linalg.norm(gcrs_m - _meme_to_gcrs(raw_m, epoch, epoch.tz_convert(None).isoformat())) / 1000.0)
            )
            residual = per_chain["meme_j2000"][-1]
            ages_hours.append(float(age_s) / 3600.0)
            sample_rows.append(
                {
                    "sat_id": sat_id,
                    "epoch_utc": epoch.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "propagation_age_hours": round(float(age_s) / 3600.0, 3),
                    "residual_km": round(residual, 4),
                }
            )
        for chain in chains:
            values = np.asarray(per_chain[chain])
            if values.size:
                sens_rows.append(
                    {
                        "sat_id": sat_id,
                        "frame_chain": chain,
                        "n_compared": len(values),
                        "residual_median_km": round(float(np.median(values)), 3),
                        "residual_p75_km": round(float(np.quantile(values, 0.75)), 3),
                        "residual_p95_km": round(float(np.quantile(values, 0.95)), 3),
                        "residual_max_km": round(float(values.max()), 3),
                    }
                )
        if per_chain["meme_j2000"]:
            residuals = np.asarray(per_chain["meme_j2000"])
            rows.append(
                {
                    "sat_id": sat_id,
                    "n_compared": len(residuals),
                    "propagation_age_median_hours": round(float(np.median(ages_hours)), 2),
                    "residual_median_km": round(float(np.median(residuals)), 3),
                    "residual_p75_km": round(float(np.quantile(residuals, 0.75)), 3),
                    "residual_p95_km": round(float(np.quantile(residuals, 0.95)), 3),
                    "residual_max_km": round(float(residuals.max()), 3),
                    "frame_assumption": "tle:TEME->GCRS; eph:MEME(J2000)->GCRS",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(sample_rows), pd.DataFrame(sens_rows)


def _load_raw_tle(tle_raw_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load archive day files overlapping [start, end] (Starlink rows only)."""
    frames = []
    for path in sorted(Path(tle_raw_dir).glob("*_space_track_data.csv")):
        day = path.name.split("_")[0]
        file_start = pd.Timestamp(day, tz="utc")
        if file_start > end + pd.Timedelta(days=1) or file_start < start - pd.Timedelta(days=2):
            continue
        frame = pd.read_csv(path, usecols=TLE_RAW_COLUMNS)
        frames.append(frame[frame["Satellite_Name"].str.contains("STARLINK", na=False)])
    if not frames:
        return pd.DataFrame(columns=TLE_RAW_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starlink-dir", default=str(STARLINK_DIR))
    parser.add_argument("--tle-raw-dir", default=str(RAW_TLE_DIR))
    parser.add_argument("--output-dir", default=str(TABLES))
    args = parser.parse_args()
    starlink_dir = Path(resolve_repo_path(args.starlink_dir))
    output = ensure_directory(resolve_repo_path(args.output_dir))

    tle = pd.read_parquet(starlink_dir / "tle_elements.parquet")
    shell_table = tle_shell_distribution(tle)
    shell_table.to_csv(output / "starlink_tle_distribution_summary.csv", index=False)
    print("wrote starlink_tle_distribution_summary.csv", flush=True)

    latest = representative_tle_elements(tle)
    shell_by_sat = dict(zip(latest["sat_id"].astype(str), latest["shell"]))
    state_table = ephemeris_state_distribution(starlink_dir / "ephemeris_state.parquet", shell_by_sat)
    state_table.to_csv(output / "starlink_ephemeris_distribution.csv", index=False)
    print("wrote starlink_ephemeris_distribution.csv", flush=True)

    ephemeris_path = starlink_dir / "ephemeris_state.parquet"
    # Light columnar passes via pyarrow keep memory bounded on the 43M-row
    # slice; the heavy position columns are read only for the sampled sats.
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    epoch_col = pq.read_table(ephemeris_path, columns=["epoch"]).column("epoch")
    span_lo, span_hi = pc.min_max(epoch_col).as_py().values()
    span_lo = pd.Timestamp(span_lo).tz_convert("utc")
    span_hi = pd.Timestamp(span_hi).tz_convert("utc")
    raw_tle = _load_raw_tle(resolve_repo_path(args.tle_raw_dir), span_lo, span_hi)
    available = set(pc.unique(pq.read_table(ephemeris_path, columns=["sat_id"]).column("sat_id")).to_pylist())
    raw_sats = set(raw_tle["Satellite_Number"].astype(int))
    shared = sorted(available & raw_sats)
    if raw_tle.empty or not shared:
        # The consistency audit needs the raw Space-Track archive day files
        # (acquisition stage); the shell and ephemeris-distribution tables
        # above were still produced from the released snapshot.
        print(
            "skip: TLE-vs-ephemeris consistency requires the raw TLE archive "
            "under data/raw/operational_constellation/starlink/tle_archive "
            "(populate via the acquisition stage)",
            flush=True,
        )
        return 0
    rng = np.random.default_rng(SAMPLE_SEED)
    sampled = sorted(int(s) for s in rng.choice(shared, size=min(SAMPLE_SATELLITES, len(shared)), replace=False))
    # Pre-filtered frames: the consistency function's internal sampling then
    # selects all of them deterministically (sample size == population).
    ephemeris = pd.read_parquet(
        ephemeris_path, columns=["sat_id", "epoch", "x_m", "y_m", "z_m"], filters=[("sat_id", "in", sampled)]
    )
    consistency, consistency_samples, frame_sensitivity = tle_ephemeris_consistency(
        raw_tle[raw_tle["Satellite_Number"].isin(sampled)], ephemeris
    )
    consistency.to_csv(output / "starlink_tle_ephemeris_consistency.csv", index=False)
    consistency_samples.to_csv(output / "starlink_tle_ephemeris_consistency_samples.csv", index=False)
    frame_sensitivity.to_csv(output / "starlink_frame_sensitivity.csv", index=False)
    print("wrote starlink_tle_ephemeris_consistency.csv + samples + frame sensitivity", flush=True)
    if not frame_sensitivity.empty:
        pivot = frame_sensitivity.pivot_table(index="sat_id", columns="frame_chain", values="residual_median_km")
        print(pivot.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
