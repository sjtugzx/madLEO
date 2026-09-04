"""Generate six supplementary analysis figures for the MAD-LEO paper.

Single-panel, publication-quality figures (8x5 in, font >= 8 pt, tight
layout, no grid) covering analyses that supplement the technical
validation suite:

- ``g_slr_oc_residuals``        SLR O-C shift vs O-C RMS per event window
- ``g_sigma_calibration``      sigma-model nominal vs empirical coverage (1/2/3 sigma)
- ``g_tost_equivalence``       tier-equivalence TOST verdict counts per element
- ``g_kozai_comparison``       Kepler vs SGP4-Brouwer mean-element delta-a difference
- ``g_tolerance_sweep``        external-benchmark matched count vs match tolerance
- ``g_match_offsets``          external-benchmark signed match-offset histogram

Inputs are read from the published ``experiments/validation/`` tables first
and fall back to the regenerated ``results/tables/`` workspace (the SLR
O-C audit currently writes only to ``results/tables/``).

Example invocation (from the repository root)::

    python scripts/experiments/run_experiments.py new-analysis-figures
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

EXPERIMENTS_VALIDATION = REPO_ROOT / "experiments" / "validation"
TABLES = REPO_ROOT / "results" / "tables"
OUTPUT = REPO_ROOT / "results"

# Okabe-Ito colorblind-safe palette (matches generate_results_figures).
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
PINK = "#CC79A7"
GREY = "#7F7F7F"
SIGMA_COLORS = {1: BLUE, 2: ORANGE, 3: VERMILION}
VERDICT_COLORS = {"equivalent": GREEN, "exceeds_tolerance": VERMILION, "inconclusive": GREY}


def analysis_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "axes.grid": False,
        }
    )


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_table(name: str) -> pd.DataFrame:
    """Read a validation table preferring the shipped dataset snapshot."""
    for root in (EXPERIMENTS_VALIDATION, TABLES):
        path = root / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"{name} not found in experiments/validation/ or results/tables/")


def save(fig, name: str, output: Path) -> None:
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}", flush=True)


def short_target(sat_id: str) -> str:
    return (
        str(sat_id)
        .replace("sentinel-", "S")
        .replace("jason-", "J")
        .replace("topex-poseidon", "TOPEX")
        .replace("cryosat-2", "CS2")
        .replace("hy-2a", "HY2A")
        .replace("saral", "SARAL")
        .replace("swot", "SWOT")
    )


def fig_slr_oc_residuals(output: Path) -> None:
    """O-C shift vs O-C RMS per window; before/after shown as paired endpoints.

    Display limits follow the bulk of the distribution (shift within
    ±250 m, RMS within 2 km); out-of-range windows are drawn as edge
    triangles so the outlier at ~4 km shift does not stretch the axes.
    """
    data = read_table("slr_oc_residuals.csv").dropna(subset=["oc_shift_m", "oc_rms_before_m", "oc_rms_after_m"])
    sat_order = sorted(data["sat_id"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {sat: cmap(i % 20) for i, sat in enumerate(sat_order)}
    x_lim, y_lim = (-250.0, 250.0), (8.0, 2000.0)

    def clip_x(v):
        return np.clip(v, x_lim[0], x_lim[1])

    def clip_y(v):
        return np.clip(v, y_lim[0], y_lim[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    # before -> after RMS comparison: one light vertical segment per window
    ax.vlines(
        clip_x(data["oc_shift_m"]),
        clip_y(data["oc_rms_before_m"]),
        clip_y(data["oc_rms_after_m"]),
        color="0.80",
        linewidth=0.6,
        zorder=1,
        rasterized=True,
    )
    ax.plot(clip_x(data["oc_shift_m"]), clip_y(data["oc_rms_before_m"]), "o", markerfacecolor="none",
            markeredgecolor="0.45", markersize=2.2, zorder=2, rasterized=True, label="before")
    for sat in sat_order:
        group = data[data["sat_id"] == sat]
        inside = group[
            group["oc_shift_m"].between(*x_lim) & group["oc_rms_after_m"].between(*y_lim)
        ]
        outside = group.drop(inside.index)
        ax.plot(inside["oc_shift_m"], inside["oc_rms_after_m"], "o", color=colors[sat],
                markersize=2.6, alpha=0.85, zorder=3, rasterized=True, label=short_target(sat))
        if len(outside):
            ax.plot(clip_x(outside["oc_shift_m"]), clip_y(outside["oc_rms_after_m"]), "^",
                    color=colors[sat], markersize=3.2, alpha=0.85, zorder=3, rasterized=True)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("log")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("O-C bias shift across event window (m)")
    ax.set_ylabel("SLR O-C RMS per window (m)")
    n_out = int((~(data["oc_shift_m"].between(*x_lim) & data["oc_rms_after_m"].between(*y_lim))).sum())
    ax.annotate(
        f"n={len(data)} windows; grey = before→after RMS change;\n"
        f"median RMS {data['oc_rms_before_m'].median():.1f} m → {data['oc_rms_after_m'].median():.1f} m"
        + (f"; triangles = {n_out} beyond axis range" if n_out else ""),
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
        ha="left", va="bottom",
    )
    ax.legend(loc="upper right", ncol=2, fontsize=7, markerscale=2.2, handletextpad=0.3, columnspacing=0.8)
    despine(ax)
    fig.tight_layout()
    save(fig, "g_slr_oc_residuals", output)


def fig_sigma_calibration(output: Path) -> None:
    """Nominal vs empirical coverage at 1/2/3 sigma per evaluation group."""
    data = read_table("sigma_calibration_curve.csv")
    group_order = [g for g in ("all", "fit", "eval_contemporary", "eval_historical") if g in set(data["sigma_eval_group"])]
    label_map = {"all": "all", "fit": "fit", "eval_contemporary": "contemporary", "eval_historical": "historical"}

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.13
    x_group = np.arange(len(group_order))
    for ki, k in enumerate((1, 2, 3)):
        nominal, empirical, counts = [], [], []
        for group in group_order:
            row = data[(data["sigma_eval_group"] == group) & (data["k_sigma"] == k)]
            nominal.append(float(row["nominal_coverage"].iloc[0]) * 100)
            empirical.append(float(row["empirical_coverage"].iloc[0]) * 100)
            counts.append(int(row["sample_count"].iloc[0]))
        base = x_group + (ki - 1) * 3 * width
        ax.bar(base - width, nominal, width=width, color=SIGMA_COLORS[k], alpha=0.35,
               label=f"{k}σ nominal")
        ax.bar(base, empirical, width=width, color=SIGMA_COLORS[k],
               label=f"{k}σ empirical")
        for xi, val in zip(base, empirical):
            ax.text(xi, val + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x_group)
    ax.set_xticklabels([f"{label_map[g]}\n(n={data[data['sigma_eval_group'] == g]['sample_count'].iloc[0]})" for g in group_order])
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 108)
    ax.axhline(68.27, color="0.6", linewidth=0.6, linestyle=":")
    ax.axhline(95.45, color="0.6", linewidth=0.6, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    order = [i for pair in ([labels.index(f"{k}σ nominal"), labels.index(f"{k}σ empirical")] for k in (1, 2, 3)) for i in pair]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3, fontsize=7, loc="upper left")
    despine(ax)
    fig.tight_layout()
    save(fig, "g_sigma_calibration", output)


def fig_tost_equivalence(output: Path) -> None:
    """Stacked TOST verdict counts per orbital element."""
    data = read_table("tier_equivalence_tost.csv")
    element_order = [e for e in ("inclination_deg", "eccentricity", "sma_altitude_km") if e in set(data["element"])]
    label_map = {"inclination_deg": "inclination", "eccentricity": "eccentricity", "sma_altitude_km": "SMA altitude"}
    verdict_order = ["equivalent", "exceeds_tolerance", "inconclusive"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(element_order))
    bottom = np.zeros(len(element_order))
    for verdict in verdict_order:
        counts = [int((data[data["element"] == e]["verdict"] == verdict).sum()) for e in element_order]
        bars = ax.bar(x, counts, width=0.55, bottom=bottom, color=VERDICT_COLORS[verdict], label=verdict)
        for xi, (count, base) in enumerate(zip(counts, bottom)):
            if count > 0:
                ax.text(xi, base + count / 2, str(count), ha="center", va="center", fontsize=8,
                        color="white" if verdict != "inconclusive" else "black")
        bottom += np.asarray(counts, dtype=float)
    for xi, total in enumerate(bottom):
        ax.text(xi, total + 0.4, f"n={int(total)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[e] for e in element_order])
    ax.set_ylabel("tier-pair comparisons")
    ax.set_ylim(0, bottom.max() * 1.28)
    ax.legend(
        loc="upper left", fontsize=8, title="TOST verdict", title_fontsize=8,
        labels=["equivalent", "exceeds tolerance", "inconclusive"],
    )
    despine(ax)
    fig.tight_layout()
    save(fig, "g_tost_equivalence", output)


def fig_kozai_comparison(output: Path) -> None:
    """|Kepler - SGP4-Brouwer| delta-a difference distribution per event."""
    data = read_table("kozai_comparison_per_event.csv")
    diff = data["difference_kepler_vs_sgp4_m"].abs().dropna()
    diff = diff[diff > 0]
    median_v = float(diff.median())
    p94 = float(diff.quantile(0.94))

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(diff.min()), np.log10(diff.max()), 48)
    ax.hist(diff, bins=bins, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|Kepler − SGP4-Brouwer| Δa difference (m)")
    ax.set_ylabel("event windows")
    ax.axvline(median_v, color=VERMILION, linewidth=1.2, linestyle="--",
               label=f"median = {median_v:.3g} m")
    ax.axvline(p94, color=GREEN, linewidth=1.2, linestyle="--",
               label=f"94th percentile = {p94:.3g} m")
    ax.legend(loc="upper left", fontsize=8)
    ax.annotate(f"n={len(diff)} compared events", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=8)
    despine(ax)
    fig.tight_layout()
    save(fig, "g_kozai_comparison", output)


def fig_tolerance_sweep(output: Path) -> None:
    """Matched count vs match tolerance per target, total highlighted."""
    data = read_table("external_benchmark_tolerance_sweep.csv")
    tolerances = sorted(data["tolerance_days"].unique())
    x = np.arange(len(tolerances))
    per_target = data[data["sat_id"] != "ALL"]
    total = data[data["sat_id"] == "ALL"].set_index("tolerance_days")

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab20")
    targets = sorted(per_target["sat_id"].unique())
    for i, sat in enumerate(targets):
        group = per_target[per_target["sat_id"] == sat].set_index("tolerance_days")
        counts = [int(group["madleo_matched"].get(t, np.nan)) for t in tolerances]
        ax.plot(x, counts, "o-", color=cmap(i % 20), markersize=4, linewidth=1.0, alpha=0.85,
                label=short_target(sat))
    total_counts = [int(total["madleo_matched"].get(t, np.nan)) for t in tolerances]
    ax.plot(x, total_counts, "s-", color="black", markersize=7, linewidth=2.2,
            label=f"ALL (total): {' / '.join(str(c) for c in total_counts)}")
    for xi, count in zip(x, total_counts):
        ax.annotate(str(count), (xi, count), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g} d" for t in tolerances])
    ax.set_xlabel("match tolerance (days)")
    ax.set_ylabel("matched event pairs")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    despine(ax)
    fig.tight_layout()
    save(fig, "g_tolerance_sweep", output)


def fig_match_offsets(output: Path) -> None:
    """Signed match-offset histogram (all benchmark-covered targets)."""
    data = read_table("external_benchmark_match_offsets.csv")
    agg = data.groupby(["offset_bin_start_min", "offset_bin_end_min"], as_index=False)["matched_count"].sum()
    agg["center"] = (agg["offset_bin_start_min"] + agg["offset_bin_end_min"]) / 2.0
    agg["width"] = (agg["offset_bin_end_min"] - agg["offset_bin_start_min"]).clip(lower=30.0)
    total = int(agg["matched_count"].sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    mask_near = agg["center"].abs() <= 150
    near = agg[mask_near]
    far = agg[~mask_near]
    ax.bar(near["center"], near["matched_count"], width=near["width"] * 0.9, color=BLUE, alpha=0.85,
           label="near-zero offsets")
    ax.bar(far["center"], far["matched_count"], width=far["width"] * 0.9, color=ORANGE, alpha=0.85,
           label="±1-day offsets")
    ax.axvline(1440, color=VERMILION, linewidth=1.4, linestyle="--")
    ymax = float(agg["matched_count"].max())
    ax.set_ylim(0, ymax * 1.35)
    ax.annotate(
        "+1440 min (+1 day):\nbenchmark day-of-year\nparsing offset",
        xy=(1440, ymax * 1.22),
        xytext=(-16, 0), textcoords="offset points",
        fontsize=8, ha="right", va="center", color=VERMILION,
    )
    ax.set_xlabel("signed match offset (min), benchmark − MAD-LEO")
    ax.set_ylabel("matched event pairs")
    ax.set_xlim(-1700, 1700)
    ax.annotate(f"n={total} matched pairs", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8)
    ax.legend(loc="upper center", fontsize=8)
    despine(ax)
    fig.tight_layout()
    save(fig, "g_match_offsets", output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args()
    output = ensure_directory(resolve_repo_path(args.output_dir))
    analysis_style()
    fig_slr_oc_residuals(output)
    fig_sigma_calibration(output)
    fig_tost_equivalence(output)
    fig_kozai_comparison(output)
    fig_tolerance_sweep(output)
    fig_match_offsets(output)
    print(f"all new-analysis figures -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
