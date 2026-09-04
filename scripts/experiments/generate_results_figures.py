"""Generate Nature-style technical validation figures into results/.

Style: Okabe-Ito colorblind-safe palette, Arial/Helvetica, 89 mm single /
183 mm double column widths, thin spines (top/right off), bold lowercase
panel labels. Reads decision tables from results/tables/; shipped dataset
snapshots (dataset/) are preferred for evidence inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.config import REPO_ROOT, ensure_directory, resolve_repo_path
from benchmarking.experiment_params import BOOTSTRAP_SEED, EARTH_RADIUS_KM

TABLES = REPO_ROOT / "results" / "tables"
INTERIM_ANN = REPO_ROOT / "data" / "interim" / "normalized_sources" / "reference_validation"
OUTPUT = REPO_ROOT / "results"
# C18 (partial): figures read the shipped `dataset/` snapshot FIRST and fall
# back to the legacy pipeline workspace only when the snapshot file is absent,
# so the released package alone can regenerate as many figures as possible.
DATASET_ROOT = REPO_ROOT / "dataset"
# Legacy fallbacks (only reached when the shipped dataset/ snapshot lacks
# the file):
# - LEGACY_STARLINK_DIR: the pipeline release workspace no longer exists
#   (data/release/mad-leo/ is absent); dataset/operational/starlink/ is the
#   only live source of tle_elements.parquet.
# - LEGACY_RAW_ROOT: the local raw provider workspace (content depends on
#   the acquisition stage having run). All current figures read the shipped
#   dataset/mission_reported/evidence/ snapshot, so this path is normally
#   not reached.
LEGACY_STARLINK_DIR = REPO_ROOT / "data" / "release" / "mad-leo" / "operational" / "starlink"
LEGACY_RAW_ROOT = REPO_ROOT / "data" / "raw" / "reference_validation"


def dataset_first(dataset_rel: str | Path, legacy: Path, dataset_root: Path = DATASET_ROOT) -> Path:
    """Resolve an input path preferring the dataset snapshot over the workspace."""
    candidate = dataset_root / dataset_rel
    return candidate if candidate.exists() else legacy

# Okabe-Ito palette
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
PINK = "#CC79A7"
GREY = "#7F7F7F"
TIER_COLORS = {"A": BLUE, "B": ORANGE, "C": GREY}
SOURCE_COLORS = {"tle": BLUE, "orbit": GREEN, "slr": VERMILION}

MM = 1 / 25.4
SINGLE = 89 * MM
DOUBLE = 183 * MM


def nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label: str) -> None:
    ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def save(fig, name: str, output: Path) -> None:
    fig.savefig(output / f"{name}.png")
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {name}", flush=True)


def short_target(sat_id: str) -> str:
    return sat_id.replace("sentinel-", "S").replace("jason-", "J").replace("topex-poseidon", "TOPEX").replace("cryosat-2", "CS2").replace("hy-2a", "HY2A").replace("saral", "SARAL").replace("swot", "SWOT")


def fig_dataset_overview(tables: Path, output: Path) -> None:
    """Events per target with confidence-tier composition."""
    alignment = pd.read_csv(tables / "reference_all_window_alignment_status.csv")
    alignment["aligned"] = alignment["aligned"].astype(str) == "True"
    tiers = alignment.groupby("sat_id")["aligned"].sum().rename("A")
    counts = alignment.groupby("sat_id").size().rename("total")
    b_counts = alignment[~alignment["aligned"]].assign(
        is_b=lambda d: d["missing_sources"].fillna("") == "slr"
    ).groupby("sat_id")["is_b"].sum().rename("B")
    summary = pd.concat([counts, tiers, b_counts], axis=1).fillna(0)
    summary["C"] = summary["total"] - summary["A"] - summary["B"]
    summary = summary.sort_values("A")

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.75))
    y = np.arange(len(summary))
    ax.barh(y, summary["A"], color=TIER_COLORS["A"], height=0.7, label="A (three-source)")
    ax.barh(y, summary["B"], left=summary["A"], color=TIER_COLORS["B"], height=0.7, label="B (missing SLR)")
    ax.barh(y, summary["C"], left=summary["A"] + summary["B"], color=TIER_COLORS["C"], height=0.7, label="C (incomplete)")
    ax.set_yticks(y, [short_target(s) for s in summary.index])
    ax.set_xlabel("Mission-reported maneuver windows")
    ax.legend(loc="lower right")
    despine(ax)
    save(fig, "a_q1_dataset_overview", output)


def fig_source_coverage(tables: Path, output: Path) -> None:
    """Per-source coverage fraction per target."""
    alignment = pd.read_csv(tables / "reference_all_window_alignment_status.csv")
    rows = []
    for sat_id, group in alignment.groupby("sat_id"):
        rows.append(
            {
                "sat_id": sat_id,
                "tle": (group["tle_status"] == "covered").mean(),
                "orbit": (group["orbit_status"] == "covered").mean(),
                "slr": (group["slr_status"] == "covered").mean(),
            }
        )
    summary = pd.DataFrame(rows).set_index("sat_id").loc[alignment.groupby("sat_id").size().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.95))
    data = summary[["orbit", "slr"]].to_numpy() * 100
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_yticks(range(len(summary)), [short_target(s) for s in summary.index])
    ax.set_xticks(range(2), ["Orbit", "SLR"])
    for y in range(data.shape[0]):
        for x in range(2):
            ax.text(x, y, f"{data[y, x]:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if data[y, x] < 60 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Coverage (%)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    save(fig, "a_q1_source_coverage", output)


def fig_response_distributions(tables: Path, output: Path) -> None:
    """TLE delta-sma distribution per target + overall histogram."""
    response = pd.read_csv(tables / "maneuver_event_response_validation.csv")
    delta_col = "delta_sma_km" if "delta_sma_km" in response.columns else None
    if delta_col is None:
        return
    response = response[response[delta_col].notna()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, SINGLE * 0.8))
    # C12: the boxplot shows |delta|; sort targets by the median ABSOLUTE value
    # so the ordering is monotonically consistent with what is drawn.
    order = response.assign(_abs=response[delta_col].abs()).groupby("sat_id")["_abs"].median().sort_values().index
    data = [response.loc[response["sat_id"] == s, delta_col].abs() * 1000 for s in order]
    bp = ax1.boxplot(data, vert=True, showfliers=False, widths=0.6, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1})
    for patch in bp["boxes"]:
        patch.set_facecolor(SKY)
        patch.set_alpha(0.7)
    ax1.set_xticks(range(1, len(order) + 1), [short_target(s) for s in order], rotation=45, ha="right")
    ax1.set_ylabel(r"$|\Delta\mathrm{sma}|$ across event (m)")
    ax1.set_yscale("log")
    panel_label(ax1, "a")
    despine(ax1)

    values = np.sort(response[delta_col].abs() * 1000)
    ecdf = np.arange(1, len(values) + 1) / len(values)
    shown = values <= 500
    ax2.plot(values[shown], ecdf[shown], color=BLUE, linewidth=1.2)
    tail_n = int((~shown).sum())
    ax2.annotate(f"ECDF(500 m) = {ecdf[shown][-1]:.2f}; {tail_n} large-maneuver events beyond (see tables)",
                 xy=(0.97, 0.12), xycoords="axes fraction", fontsize=5.8, ha="right")
    median_v = np.median(values)
    ax2.plot([median_v, median_v], [0, 0.5], color=VERMILION, linewidth=0.9, linestyle="--")
    ax2.plot([values[0], median_v], [0.5, 0.5], color=VERMILION, linewidth=0.9, linestyle="--",
             label=f"median = {median_v:.1f} m")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel(r"$|\Delta\mathrm{sma}|$ across event (m)")
    ax2.set_ylabel("ECDF")
    ax2.legend(loc="lower right")
    panel_label(ax2, "b")
    despine(ax2)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "c_q2_tle_response", output)


def fig_propagation_curve(tables: Path, output: Path) -> None:
    """E2: SGP4 propagation residual vs propagation distance."""
    propagation = pd.read_csv(tables / "sgp4_propagation_residuals.csv")
    bin_centers, medians, p75s = [], [], []
    for label, median, p75 in zip(propagation["propagation_bin_hours"], propagation["median"], propagation["p75"]):
        lo, hi = [float(v) for v in label.strip("()]").split(",")]
        bin_centers.append((lo + hi) / 2.0 if np.isfinite(hi) else 72.0)
        medians.append(median)
        p75s.append(min(p75, 4.5))  # clip display; the 48-96 h p75 outlier is reported in the table

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.8))
    counts = propagation["count"].to_numpy()
    ax.plot(bin_centers, medians, "o-", color=BLUE, markersize=4, label="median")
    ax.plot(bin_centers, p75s, "s--", color=VERMILION, markersize=3.5, label="75th percentile")
    for cx, cm, n in zip(bin_centers, medians, counts):
        ax.annotate(f"n={n}", (cx, cm), textcoords="offset points", xytext=(0, -14), fontsize=5.2, ha="center",
                    color=GREY)
    edge_p75, edge_n = None, None
    for label, count, p75 in zip(propagation["propagation_bin_hours"], propagation["count"], propagation["p75"]):
        lo, hi = [float(v) for v in str(label).strip("()]").split(",")]
        if lo == 48.0 and hi == 96.0:
            edge_p75, edge_n = float(p75), int(count)
            break
    if edge_p75 is not None:
        ax.annotate(f"48–96 h: p75={edge_p75:.0f} km (n={edge_n}, edge-extrapolated)",
                    xy=(72, 4.5), xytext=(6, 2.55), fontsize=5.8,
                    arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SGP4 propagation distance (h)")
    ax.set_ylabel("TLE vs POD residual (km)")
    ax.set_ylim(0.3, 5)
    ax.legend(loc="upper left")
    despine(ax)
    save(fig, "d_q5_propagation_residuals", output)


def fig_interpolation_selfcheck(tables: Path, output: Path) -> None:
    """E3: per-target interpolation error budget."""
    check = pd.read_csv(tables / "orbit_interpolation_selfcheck.csv").sort_values("interpolation_error_median_m")
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    y = np.arange(len(check))
    ax.hlines(y, check["interpolation_error_median_m"], check["interpolation_error_p75_m"],
              color=GREEN, linewidth=0.8, alpha=0.6)
    ax.plot(check["interpolation_error_median_m"], y, "o", color=GREEN, markersize=4.5, label="median")
    ax.plot(check["interpolation_error_p75_m"], y, "|", color=GREEN, markersize=7, label="p75")
    ax.set_yticks(y, [short_target(s) for s in check["sat_id"]])
    ax.set_xscale("log")
    ax.set_xlabel("Interpolation self-check error (m)")
    ax.axvline(0.01, color=VERMILION, linewidth=0.8, linestyle="--", label="1 cm")
    ax.set_xlim(right=3e-2)
    ax.legend(loc="lower right", fontsize=5.8)
    despine(ax)
    save(fig, "d_q5_interpolation_selfcheck", output)


def fig_leave_one_out(tables: Path, output: Path) -> None:
    """E5: leave-POD-out TLE-only residuals with 3-sigma coverage."""
    loo = pd.read_csv(tables / "harmonization_validation_summary.csv").sort_values("tle_only_residual_median_km")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, SINGLE * 0.85))
    y = np.arange(len(loo))
    ax1.hlines(y, loo["tle_only_residual_median_km"], loo["tle_only_residual_p75_km"],
               color=BLUE, linewidth=0.9, alpha=0.55)
    ax1.plot(loo["tle_only_residual_median_km"], y, "o", color=BLUE, markersize=5, label="median")
    ax1.plot(loo["tle_only_residual_p75_km"], y, "o", color=SKY, markersize=4, label="p75")
    ax1.set_yticks(y, [short_target(s) for s in loo["sat_id"]])
    ax1.set_xlabel("TLE-only vs held-out POD residual (km)")
    ax1.legend(loc="lower right")
    panel_label(ax1, "a")
    despine(ax1)

    ax2.plot(loo["within_3sigma_fraction"] * 100, y, "o", color=GREEN, markersize=5, label="all windows")
    if "within_3sigma_fraction_no_extrapolation" in loo.columns:
        clean = pd.to_numeric(loo["within_3sigma_fraction_no_extrapolation"], errors="coerce") * 100
        ax2.plot(clean, y, "o", color=GREEN, markersize=5, markerfacecolor="none", label="excl. edge-extrapolated")
    ax2.set_yticks(y, [short_target(s) for s in loo["sat_id"]])
    ax2.set_xlabel(r"Windows within predicted 3$\sigma$ (%)")
    ax2.set_xlim(60, 102)
    ax2.axvline(95, color=VERMILION, linewidth=0.8, linestyle="--", label="95%")
    ax2.legend(loc="lower left", fontsize=5.8)
    panel_label(ax2, "b")
    despine(ax2)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "d_q5_sigma_leave_one_out", output)


def fig_bias_calibration(tables: Path, output: Path) -> None:
    """E4: TLE-vs-POD systematic bias components per target."""
    bias = pd.read_csv(tables / "source_bias_calibration.csv")
    per_target = (
        bias.groupby("sat_id")[["radial_bias_m", "along_track_bias_m", "cross_track_bias_m"]]
        .median()
        .loc[bias.groupby("sat_id")["norm_median_km"].median().sort_values().index]
    )
    data = per_target.to_numpy()
    vmax = float(np.abs(data).max())
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.95))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(per_target)), [short_target(s) for s in per_target.index])
    ax.set_xticks(range(3), ["radial", "along-track", "cross-track"])
    for yy in range(data.shape[0]):
        for xx in range(3):
            ax.text(xx, yy, f"{data[yy, xx]:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(data[yy, xx]) > vmax * 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("TLE − POD bias, median (m)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    save(fig, "d_q5_bias_calibration", output)


def fig_specificity(tables: Path, output: Path) -> None:
    """E6 (revised): event vs stable-window TLE response distributions."""
    events = pd.read_csv(tables / "maneuver_event_response_validation.csv")["delta_sma_km"].abs().dropna() * 1000
    stable = pd.read_csv(tables / "stable_window_response_validation.csv")["delta_sma_km"].abs().dropna() * 1000
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.8))
    for values, color, label in (
        (np.sort(events), VERMILION, f"events (median {events.median():.1f} m)"),
        (np.sort(stable), BLUE, f"stable windows (median {stable.median():.1f} m)"),
    ):
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, linewidth=1.4, label=label)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(right=500)
    ax.set_xlabel(r"$|\Delta\mathrm{sma}|$ across window (m)")
    ax.set_ylabel("ECDF")
    tail_events = int((events > 500).sum())
    ax.annotate(f"display capped at 500 m\n({tail_events} large-maneuver events beyond)",
                xy=(0.97, 0.03), xycoords="axes fraction", fontsize=5.8, ha="right")
    ax.legend(loc="upper left")
    despine(ax)
    save(fig, "c_q5_specificity", output)


def fig_tle_vs_pod_mean_sma(tables: Path, output: Path) -> None:
    """Mean-element-scale comparison: TLE Δsma vs POD period-averaged Δsma."""
    response = pd.read_csv(tables / "maneuver_event_response_validation.csv")
    needed = {"delta_sma_km", "orbit_mean_sma_shift_m"}
    if not needed.issubset(response.columns):
        return
    pair = response.dropna(subset=list(needed))
    x = pair["delta_sma_km"].to_numpy()
    y = pair["orbit_mean_sma_shift_m"].to_numpy() / 1000.0
    if len(pair) < 10:
        return
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    ax.plot(x, y, ".", color=BLUE, markersize=1.5, alpha=0.45, rasterized=True)
    lim = max(abs(x).max(), abs(y).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=VERMILION, linewidth=0.8, linestyle="--", label="1:1")
    ax.plot([-lim, lim], [intercept - slope * lim, intercept + slope * lim], color="black", linewidth=0.8,
            label=f"fit: slope {slope:.2f}, r = {corr:.2f} (n={len(pair)})")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel(r"TLE mean-element $\Delta$sma (km)")
    ax.set_ylabel(r"POD period-averaged $\Delta$sma (km)")
    ax.legend(loc="upper left")
    despine(ax)
    save(fig, "c_q2_same_scale_response", output)


SHELL_COLORS = {"43.0": GREEN, "53.0": BLUE, "53.2": SKY, "70": ORANGE, "97.6": VERMILION, "other": GREY}


def fig_starlink_shell_distributions(tables: Path, output: Path, dataset_root: Path = DATASET_ROOT) -> None:
    """Q4.2 (I3/I4): Starlink shell structure — inclination x altitude + RAAN planes."""
    # Flat import (C1): the run_experiments dispatcher puts this directory on
    # sys.path; there is no `experiments` package, so `from experiments.` always
    # raised ModuleNotFoundError and results-figures never completed.
    from generate_starlink_distributions import representative_tle_elements

    summary = pd.read_csv(tables / "starlink_tle_distribution_summary.csv", dtype={"shell": str})
    starlink_dir = dataset_first("operational/starlink", LEGACY_STARLINK_DIR, dataset_root)
    tle = pd.read_parquet(starlink_dir / "tle_elements.parquet")
    representative = representative_tle_elements(tle)
    counts = summary.set_index("shell")["satellite_count"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, SINGLE * 0.95))
    from matplotlib.lines import Line2D

    handles = []
    for shell, group in representative.groupby("shell"):
        color = SHELL_COLORS.get(shell, GREY)
        label = f"{shell}° (n={counts.get(shell, len(group))})" if shell != "other" else f"other (n={counts.get(shell, len(group))})"
        ax1.plot(group["inclination_deg"], group["sma_altitude_km"], ".", color=color,
                 markersize=0.8, alpha=0.35, rasterized=True)
        handles.append(Line2D([0], [0], marker=".", color=color, linestyle="", markersize=6, label=label))
    # J2: reference-mission orbital regimes against the constellation shells
    reference = pd.read_csv(tables / "tle_element_distribution_summary.csv")
    ax1.plot(reference["inclination_median_deg"], reference["sma_altitude_median_km"], "D",
             color="black", markersize=4.5, markerfacecolor="none", markeredgewidth=1.0)
    handles.append(Line2D([0], [0], marker="D", color="black", markerfacecolor="none",
                          linestyle="", markersize=4.5, label=f"reference missions (n={len(reference)})"))
    ax1.set_xlabel("inclination (deg)")
    ax1.set_ylabel("SMA altitude (km)")
    ax1.legend(handles=handles, fontsize=5.5, loc="lower right", markerscale=1.0)
    panel_label(ax1, "a")
    despine(ax1)

    bins = np.arange(0.0, 365.0, 5.0)
    plane_counts = summary.set_index("shell")["raan_plane_count"]
    shells_sorted = [s for s in ("43.0", "53.0", "53.2", "70", "97.6") if s in plane_counts.index]
    y = np.arange(len(shells_sorted))
    ax2.barh(y, [plane_counts[s] for s in shells_sorted],
             color=[SHELL_COLORS.get(s, GREY) for s in shells_sorted], height=0.55)
    for ypos, shell in enumerate(shells_sorted):
        ax2.text(plane_counts[shell] + 0.5, ypos,
                 f"{plane_counts[shell]} planes / {counts.get(shell, 0)} sats", va="center", fontsize=6)
    ax2.set_yticks(y, [f"{s}°" for s in shells_sorted])
    ax2.set_xlabel("significant RAAN planes (5° bins, ≥50% of largest bin)")
    ax2.set_xlim(right=max(plane_counts[s] for s in shells_sorted) * 1.75)
    panel_label(ax2, "b")
    despine(ax2)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "e_q4_starlink_shell_distributions", output)


def fig_tle_ephemeris_consistency(tables: Path, output: Path) -> None:
    """Q4.4 (I6): TLE-vs-ephemeris position residuals — ECDF + per-satellite medians."""
    consistency = pd.read_csv(tables / "starlink_tle_ephemeris_consistency.csv")
    samples = pd.read_csv(tables / "starlink_tle_ephemeris_consistency_samples.csv")
    consistency = consistency.sort_values("residual_median_km").reset_index(drop=True)
    ecdf_colors = [BLUE, VERMILION, GREEN, ORANGE, SKY, PINK, GREY, "#000000"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, SINGLE * 0.95))
    for i, (sat_id, group) in enumerate(samples.groupby("sat_id")):
        ordered = np.sort(group["residual_km"].to_numpy())
        ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax1.plot(ordered, ecdf, color=ecdf_colors[i % len(ecdf_colors)], linewidth=0.9,
                 label=str(sat_id), rasterized=True)
    ax1.set_xscale("log")
    ax1.set_xlabel("TLE-vs-ephemeris position residual (km)")
    ax1.set_ylabel("ECDF")
    ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=5.2, loc="lower right", title="sat_id", title_fontsize=5.5)
    panel_label(ax1, "a")
    despine(ax1)

    x = np.arange(len(consistency))
    ax2.bar(x, consistency["residual_median_km"], color=BLUE, width=0.65, alpha=0.85)
    for xi, value in zip(x, consistency["residual_median_km"]):
        ax2.annotate(f"{value:.1f}", (xi, value), xytext=(0, 2), textcoords="offset points",
                     ha="center", fontsize=5.5)
    ax2.set_xticks(x, [str(s) for s in consistency["sat_id"]], rotation=45, ha="right")
    ax2.set_ylabel("median residual (km)")
    ax2.set_ylim(0, consistency["residual_median_km"].max() * 1.18)
    panel_label(ax2, "b")
    despine(ax2)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "f_q4_tle_ephemeris_consistency", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Nature-style technical validation figures")
    parser.add_argument(
        "--dataset-root",
        default=str(DATASET_ROOT),
        help="Shipped dataset snapshot root (preferred input; legacy workspace files are fallback)",
    )
    args = parser.parse_args()
    dataset_root = Path(resolve_repo_path(args.dataset_root) or args.dataset_root)
    nature_style()
    tables = resolve_repo_path(str(TABLES))
    output = ensure_directory(resolve_repo_path(str(OUTPUT)))
    fig_dataset_overview(tables, output)
    fig_source_coverage(tables, output)
    fig_response_distributions(tables, output)
    fig_propagation_curve(tables, output)
    fig_interpolation_selfcheck(tables, output)
    fig_leave_one_out(tables, output)
    fig_bias_calibration(tables, output)
    fig_specificity(tables, output)
    fig_tle_vs_pod_mean_sma(tables, output)
    fig_sigma_split_validation(tables, output)
    fig_coverage_matching(tables, output)
    fig_orbital_regimes(tables, output, dataset_root)
    fig_maneuver_anatomy(tables, output, dataset_root)
    fig_external_validation(tables, output)
    fig_evidence_distributions(tables, output)
    fig_starlink_shell_distributions(tables, output, dataset_root)
    fig_tle_ephemeris_consistency(tables, output)
    print(f"all figures -> {output}", flush=True)




def fig_sigma_split_validation(tables: Path, output: Path) -> None:
    """TV-2: per-window residuals vs propagation age, colored by evaluation group."""
    from analyzers.tv_hardening import SIGMA_EVAL_HISTORICAL_TARGETS, SIGMA_FIT_TARGETS, fit_tle_sigma_model

    estimates = pd.read_csv(tables / "maneuver_event_state_estimates.csv")
    model = fit_tle_sigma_model(estimates)
    usable = estimates[(estimates["tle_state_status"] == "computed") & (estimates["orbit_state_status"] == "computed")].copy()
    usable["residual_km"] = np.sqrt(
        (usable["itrs_x_m"] - usable["orbit_x_m"]) ** 2
        + (usable["itrs_y_m"] - usable["orbit_y_m"]) ** 2
        + (usable["itrs_z_m"] - usable["orbit_z_m"]) ** 2
    ) / 1000.0
    usable["prop_hours"] = usable["tle_propagation_seconds"].abs() / 3600.0

    def group_of(sat):
        if sat in SIGMA_FIT_TARGETS:
            return "fit"
        if sat in SIGMA_EVAL_HISTORICAL_TARGETS:
            return "eval_historical"
        return "eval_contemporary"

    usable["sigma_eval_group"] = usable["sat_id"].map(group_of)
    colors = {"fit": BLUE, "eval_contemporary": ORANGE, "eval_historical": VERMILION}

    fig, ax = plt.subplots(figsize=(DOUBLE, SINGLE * 0.85))
    for group, color in colors.items():
        sub = usable[usable["sigma_eval_group"] == group]
        ax.plot(sub["prop_hours"], sub["residual_km"], ".", color=color, markersize=1.6, alpha=0.5,
                rasterized=True, label=f"{group} (n={len(sub)})")
    grid = np.linspace(0, 96, 100)
    ax.plot(grid, model["base_km"] + model["growth_km_per_24h"] * grid / 24.0, color="black", linewidth=1.0,
            label=r"fitted $\sigma$ model")
    ax.plot(grid, 3 * (model["base_km"] + model["growth_km_per_24h"] * grid / 24.0), color="black",
            linewidth=0.8, linestyle="--", label=r"3$\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SGP4 propagation distance (h)")
    ax.set_ylabel("TLE-vs-POD residual (km)")
    ax.set_xlim(0.1, 120)
    ax.set_ylim(0.05, 30)
    ax.legend(loc="upper left", fontsize=6.5)
    despine(ax)
    save(fig, "d_q5_sigma_eval_groups", output)


def fig_external_validation(tables: Path, output: Path) -> None:
    """A4: bidirectional benchmark match rates + matched/unmatched magnitudes."""
    summary = pd.read_csv(tables / "external_benchmark_crossvalidation.csv")
    events = pd.read_csv(tables / "external_benchmark_crossvalidation_per_event.csv")
    per_sat = summary[(summary["sat_id"] != "ALL") & (summary["benchmark_events"] > 0)].copy()
    per_sat["forward"] = per_sat["benchmark_matched"] / per_sat["benchmark_events"]
    per_sat["reverse"] = per_sat["madleo_matched"] / per_sat["madleo_events"]
    per_sat["reverse_in_span"] = (
        per_sat["madleo_matched_in_benchmark_span"] / per_sat["madleo_events_in_benchmark_span"]
    )
    per_sat = per_sat.sort_values("reverse")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, SINGLE * 1.05))
    y = np.arange(len(per_sat))
    ax1.hlines(y, per_sat["reverse"] * 100, per_sat["forward"] * 100, color="0.7", linewidth=0.8)
    ax1.plot(per_sat["forward"] * 100, y, "o", color=BLUE, markersize=5,
             label="forward (benchmark → MAD-LEO)")
    ax1.plot(per_sat["reverse"] * 100, y, "s", color=VERMILION, markersize=4.5,
             label="reverse (MAD-LEO → benchmark)")
    ax1.plot(per_sat["reverse_in_span"] * 100, y, "D", color=GREEN, markersize=4, markerfacecolor="none",
             label="reverse, within benchmark span")
    ax1.set_yticks(y, [short_target(s) for s in per_sat["sat_id"]])
    ax1.set_xlabel("match rate (%)")
    ax1.set_xlim(35, 104)
    ax1.legend(loc="upper left", fontsize=5.8)
    panel_label(ax1, "a")
    despine(ax1)

    matched = events[events["matched_within_1d"]]["abs_delta_sma_m"].dropna().to_numpy()
    unmatched = events[(events["benchmark_covered_target"]) & (~events["matched_within_1d"])][
        "abs_delta_sma_m"
    ].dropna().to_numpy()
    bp = ax2.boxplot([matched, unmatched], showfliers=False, widths=0.45, patch_artist=True,
                     boxprops=dict(linewidth=0.9), medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], (BLUE, VERMILION)):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax2.set_xticks([1, 2], [f"matched (n={len(matched)})", f"unmatched, out of span\n(n={len(unmatched)})"], fontsize=6.5)
    for xpos, values in enumerate((matched, unmatched), start=1):
        ax2.annotate(f"median {np.median(values):.1f} m", xy=(xpos, np.median(values)),
                     xytext=(8, 6), textcoords="offset points", fontsize=6)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$|\Delta\mathrm{sma}|$ across window (m)")
    panel_label(ax2, "b")
    despine(ax2)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "b_q1_external_validation", output)


def fig_evidence_distributions(tables: Path, output: Path) -> None:
    """Q6: evidence-snapshot distribution completeness (TLE elements + tier equivalence)."""
    tle = pd.read_csv(tables / "tle_element_distribution_summary.csv")
    equiv = pd.read_csv(tables / "tier_distribution_equivalence.csv")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DOUBLE, SINGLE * 0.95))
    for _, row in tle.iterrows():
        ax1.errorbar(
            row["inclination_median_deg"], row["sma_altitude_median_km"],
            xerr=max(row["inclination_iqr_deg"], 5e-4), yerr=max(row["sma_altitude_iqr_km"], 0.02),
            fmt="o", color=BLUE, markersize=4, elinewidth=0.6, capsize=1.5,
        )
    clusters: dict[tuple[float, float], list[tuple[str, float, float]]] = {}
    for _, row in tle.iterrows():
        key = (round(row["inclination_median_deg"]), round(row["sma_altitude_median_km"] / 30))
        clusters.setdefault(key, []).append(
            (short_target(row["sat_id"]), row["inclination_median_deg"], row["sma_altitude_median_km"])
        )
    for members in clusters.values():
        label = "/".join(m[0] for m in members)
        ax1.annotate(label, (members[0][1], members[0][2]), xytext=(6, 6),
                     textcoords="offset points", fontsize=5.5)
    ax1.set_xlabel("median inclination (deg)")
    ax1.set_ylabel("median SMA altitude (km)")
    panel_label(ax1, "a")
    despine(ax1)

    tle_sorted = tle.copy()
    tle_sorted["start"] = pd.to_datetime(tle_sorted["first_epoch_utc"], format="ISO8601")
    tle_sorted["end"] = pd.to_datetime(tle_sorted["last_epoch_utc"], format="ISO8601")
    tle_sorted = tle_sorted.sort_values("start")
    y = np.arange(len(tle_sorted))
    start_year = tle_sorted["start"].dt.year + tle_sorted["start"].dt.dayofyear / 365.25
    end_year = tle_sorted["end"].dt.year + tle_sorted["end"].dt.dayofyear / 365.25
    ax2.hlines(y, start_year, end_year, color=GREEN, linewidth=3)
    ax2.set_yticks(y, [short_target(s) for s in tle_sorted["sat_id"]])
    ax2.set_xlabel("TLE epoch coverage (year)")
    panel_label(ax2, "b")
    despine(ax2)

    eq = equiv[equiv["sat_id"] != "ALL_WITHIN_TARGET"]
    alt = eq[eq["element"] == "sma_altitude"].groupby("sat_id")["median_abs_delta"].max()
    inc = eq[eq["element"] == "inclination"].groupby("sat_id")["median_abs_delta"].max()
    label_offsets = {
        "jason-1": (6, 6),
        "cryosat-2": (-2, 10),
        "jason-3": (6, 6),
        "hy-2a": (6, 6),
        "saral": (6, -13),
        "topex-poseidon": (6, -13),
        "jason-2": (6, 6),
    }
    for sat in alt.index:
        if sat in inc.index:
            ax3.plot(alt[sat], inc[sat], "o", color=VERMILION, markersize=4.5)
            ax3.annotate(short_target(sat), (alt[sat], inc[sat]),
                         xytext=label_offsets.get(sat, (4, 4)),
                         textcoords="offset points", fontsize=5.5)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("within-target tier altitude shift (km)")
    ax3.set_ylabel("within-target tier inclination shift (deg)")
    panel_label(ax3, "c")
    despine(ax3)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "e_q6_tle_element_distributions", output)


def fig_coverage_matching(tables: Path, output: Path) -> None:
    """TV-4: per-target tier-A share, stable windows before vs after matching."""
    report = pd.read_csv(tables / "stable_windows_match_report.csv")
    report = report.dropna(subset=["matched_tier_a_fraction"])

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.9))
    # C14: no 1:1 diagonal here -- the x axis is categorical (events / stable
    # full / stable matched), so a diagonal carries no identity meaning.
    for _, row in report.iterrows():
        ax.plot([0.15, 0.85], [row["event_tier_a_fraction"], row["matched_tier_a_fraction"]],
                "-", color=GREY, linewidth=0.7, zorder=1)
    ax.plot(np.full(len(report), 0.15), report["event_tier_a_fraction"], "o", color=VERMILION,
            markersize=5, label="events (per target)", zorder=3)
    ax.plot(np.full(len(report), 0.15), report["event_tier_a_fraction"] * np.nan, "o")  # spacer
    full = pd.read_csv(tables / "stable_windows.csv")
    full_frac = full.groupby("sat_id")["confidence_tier"].apply(lambda s: (s == "A").mean())
    x0 = [0.35] * len(report)
    ax.plot(x0, [full_frac.get(s, np.nan) for s in report["sat_id"]], "o", color=GREY, markersize=5,
            label="stable, full set", zorder=3)
    ax.plot(np.full(len(report), 0.85), report["matched_tier_a_fraction"], "o", color=BLUE,
            markersize=5, label="stable, coverage-matched", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0.15, 0.35, 0.85], ["events", "stable full", "stable matched"])
    ax.set_ylabel("Tier-A share (per target)")
    ax.legend(loc="lower right", fontsize=6)
    despine(ax)
    save(fig, "c_q5_coverage_matching", output)


def _orbital_regime_points(sat_id: str, dataset_root: Path, rng) -> tuple[np.ndarray, np.ndarray] | None:
    """(inclination deg, altitude km) from <=400 sampled TLE epochs.

    C18: reads the released evidence snapshot
    (``dataset/mission_reported/evidence/tle/<sat>.parquet``) first; the
    legacy raw TLE day
    files under ``data/raw/`` are only a fallback for workspaces predating
    the snapshot (in the current checkout that archive was deleted and its
    symlinks dangle, so the fallback is effectively unreachable).
    """
    evidence_path = dataset_root / "mission_reported" / "evidence" / "tle" / f"{sat_id}.parquet"
    if evidence_path.exists():
        frame = pd.read_parquet(
            evidence_path, columns=["epoch", "mean_motion_rad_per_min", "inclination_rad"]
        )
        if frame.empty:
            return None
        take = frame.sample(n=min(400, len(frame)), random_state=rng)
        mean_motion_rad_s = take["mean_motion_rad_per_min"].to_numpy() / 60.0
        sma_km = (3.986004418e14 / mean_motion_rad_s**2) ** (1 / 3) / 1000.0 - EARTH_RADIUS_KM
        return np.degrees(take["inclination_rad"].to_numpy()), sma_km

    from benchmarking.normalization import parse_raw_tle_file

    tle_dir = LEGACY_RAW_ROOT / sat_id / "tle"
    if not tle_dir.is_dir():
        return None
    frames = []
    for path in sorted(tle_dir.iterdir()):
        if path.suffix.lower() in {".tle", ".txt"} and not path.name.startswith("."):
            frame = parse_raw_tle_file(path, sat_id=sat_id)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return None
    tle = pd.concat(frames, ignore_index=True)
    take = tle.sample(n=min(400, len(tle)), random_state=rng)
    mean_motion_rad_s = take["mean_motion"].to_numpy() / 60.0
    sma_km = (3.986004418e14 / mean_motion_rad_s**2) ** (1 / 3) / 1000.0 - EARTH_RADIUS_KM
    return np.degrees(take["inclination"].to_numpy()), sma_km


def fig_orbital_regimes(tables: Path, output: Path, dataset_root: Path = DATASET_ROOT) -> None:
    """Orbital-regime diversity from sampled TLE epochs (real catalog data)."""
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.9))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    oi_cycle = [BLUE, VERMILION, GREEN, ORANGE, SKY, PINK, GREY, "#000000"]
    for i, sat_id in enumerate(["sentinel-3a", "sentinel-3b", "jason-3", "sentinel-6a", "cryosat-2", "saral",
                                "jason-1", "jason-2", "topex-poseidon", "hy-2a", "swot"]):
        points = _orbital_regime_points(sat_id, dataset_root, rng)
        if points is None:
            continue
        incl_deg, sma_km = points
        ax.plot(incl_deg, sma_km, ".", markersize=1.2, alpha=0.35, rasterized=True,
                color=oi_cycle[i % len(oi_cycle)], label=short_target(sat_id))
    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlim(60, 108)
    ax.set_ylim(650, 1420)
    ax.legend(fontsize=5.5, markerscale=4, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=2,
              framealpha=0.9, columnspacing=0.8, handletextpad=0.4)
    ax.annotate("400 sampled TLE epochs per target;\ndiversity spans era, tracking systems, and processing chains",
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=5.8, style="italic")
    despine(ax)
    save(fig, "a_dataset_orbital_regimes", output)


def fig_maneuver_anatomy(tables: Path, output: Path, dataset_root: Path = DATASET_ROOT) -> None:
    """Anatomy of annotated maneuvers: TLE + POD + SLR around mission-reported t0.

    Fully dataset-sufficient: all three panels read the shipped evidence
    snapshots under ``dataset/mission_reported/evidence/`` (TLE element
    history, ECEF orbit
    state vectors, SLR normal points), so this figure regenerates from the
    released package alone -- the legacy raw archive is no longer needed.
    """
    from analyzers.event_response import _vis_viva_sma_m

    # (sat_id, event t0, row label)
    cases = [
        ("sentinel-3a", "2024-01-24T08:25:00Z", "Sentinel-3A routine station-keeping"),
        ("swot", "2023-07-11T03:00:04Z", "SWOT commissioning raise"),
        ("cryosat-2", "2023-10-18T15:37:17Z", "CryoSat-2 orbit maintenance"),
    ]

    # auto-pick actual annotated events nearest the requested dates
    def nearest_event(sat_id, t0_guess):
        ann_path = dataset_root / "mission_reported" / "annotations" / "maneuver_annotations.csv"
        ann = pd.read_csv(ann_path)
        if "sat_id" in ann.columns:
            ann = ann[ann["sat_id"].astype(str) == sat_id].reset_index(drop=True)
        times = pd.to_datetime(ann["event_time_utc"], utc=True, format="ISO8601")
        idx = (times - pd.Timestamp(t0_guess)).abs().idxmin()
        return times.iloc[idx], ann.iloc[idx]

    evidence = dataset_root / "mission_reported" / "evidence"
    for row, (sat_id, t0_guess, label) in enumerate(cases):
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, SINGLE * 0.75))
        t0, ann_row = nearest_event(sat_id, t0_guess)

        # TLE mean motion series (+/- 5 days)
        tle = pd.read_parquet(evidence / "tle" / f"{sat_id}.parquet",
                              columns=["epoch", "mean_motion_rad_per_min"])
        band = tle[(tle["epoch"] >= t0 - pd.Timedelta(days=5)) & (tle["epoch"] <= t0 + pd.Timedelta(days=5))].copy()
        band["days"] = (band["epoch"] - t0).dt.total_seconds() / 86400.0
        ax = axes[0]
        if len(band):
            rel = (band["mean_motion_rad_per_min"] - band["mean_motion_rad_per_min"].iloc[0]) * 1e6
            ax.plot(band["days"], rel, ".", color=BLUE, markersize=2.5)
        ax.axvline(0, color=VERMILION, linewidth=0.9)
        ax.axvspan(-0.25, 1.0, color=ORANGE, alpha=0.15, linewidth=0)
        ax.set_ylabel(r"TLE mean-motion change" + "\n" + r"($10^{-6}$ rad/min)")
        despine(ax)

        # POD period-averaged SMA (+/- 2 days) from shipped ECEF state vectors.
        # Released velocities are ECEF; rotating_frame=True applies the
        # omega x r correction (single shared implementation, schema B14).
        ax = axes[1]
        orbit_path = evidence / "orbit" / f"{sat_id}.parquet"
        plotted = False
        if orbit_path.exists():
            orb = pd.read_parquet(
                orbit_path,
                columns=["epoch", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"],
            )
            orb = orb[(orb["epoch"] >= t0 - pd.Timedelta(days=2)) & (orb["epoch"] <= t0 + pd.Timedelta(days=2))]
            if len(orb) >= 8:
                frame = orb.rename(columns={
                    "x_m": "x", "y_m": "y", "z_m": "z",
                    "vx_mps": "vx", "vy_mps": "vy", "vz_mps": "vz",
                })
                sma = _vis_viva_sma_m(frame, True).dropna()
                a = float(sma.median())
                period_s = 2.0 * np.pi * np.sqrt(a**3 / 3.986004418e14)
                times = pd.to_datetime(frame["epoch"], utc=True, format="ISO8601")
                step = float(np.median(np.diff(times.astype("int64") / 1e9)))
                window = max(2, int(round(period_s / max(step, 1e-6))))
                smooth = sma.rolling(window=window, center=True, min_periods=max(2, window // 3)).mean()
                edge = window // 2 + 1
                if len(smooth) > 2 * edge + 4:
                    smooth = smooth.iloc[edge:-edge]
                smooth = smooth.dropna()
                days = (times.loc[smooth.index] - t0).dt.total_seconds() / 86400.0
                step = max(1, len(days) // 400)
                ax.plot(days.iloc[::step], (smooth - a).iloc[::step], "-", color=GREEN, linewidth=0.8)
                plotted = True
        ax.axvline(0, color=VERMILION, linewidth=0.9)
        ax.axvspan(-0.25, 1.0, color=ORANGE, alpha=0.15, linewidth=0)
        ax.set_ylabel("POD SMA − band median (m)")
        if plotted and row == 0:
            ax.set_ylim(-400, 400)
        despine(ax)

        # SLR normal points (+/- 2 days)
        ax = axes[2]
        slr_path = evidence / "slr" / f"{sat_id}.parquet"
        if slr_path.exists():
            slr = pd.read_parquet(slr_path, columns=["epoch", "record_type", "sigma_m"])
            sb = slr[(slr["epoch"] >= t0 - pd.Timedelta(days=2)) & (slr["epoch"] <= t0 + pd.Timedelta(days=2))]
            if "record_type" in sb.columns:
                sb = sb[sb["record_type"] == "normal_point"]
            sb = sb[sb["sigma_m"].notna()]
            if len(sb):
                days = (sb["epoch"] - t0).dt.total_seconds() / 86400.0
                ax.plot(days, sb["sigma_m"].astype(float), ".", color=VERMILION, markersize=2.5)
        ax.axvline(0, color=VERMILION, linewidth=0.9)
        ax.axvspan(-0.25, 1.0, color=ORANGE, alpha=0.15, linewidth=0)
        ax.set_ylabel("SLR sigma (m)")
        despine(ax)

        for col, title in enumerate(("TLE evidence", "precise-orbit evidence", "SLR audit")):
            axes[col].set_title(title, fontsize=7.5)
        for ax in axes:
            ax.set_xlabel("days relative to event time")
        fig.suptitle(label, fontsize=8)
        fig.subplots_adjust(wspace=0.35, top=0.85)
        save(fig, f"c_q2_maneuver_anatomy_{chr(97 + row)}", output)
    plt.close("all")

if __name__ == "__main__":
    main()
