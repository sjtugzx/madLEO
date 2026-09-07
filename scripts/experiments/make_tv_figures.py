#!/usr/bin/env python3
"""Build the technical-validation figures for the MAD-LEO paper.

Every panel is drawn directly from the published technical-validation tables
under ``experiments/validation/`` (and the released
``dataset/operational/starlink/`` subset, with the Starlink analysis tables
under ``experiments/starlink/``)
with matplotlib, except the two standalone event-anatomy figures
(``tv_anatomy_a.pdf`` / ``tv_anatomy_b.pdf``), which embed the component PDFs
kept in ``arxiv/MAD_LEO_paper/images/`` (``c_q2_maneuver_anatomy_{a,b}.pdf``) via
PyMuPDF as rasterized images.

Style follows Nature figure conventions: Arial (sans-serif), 5--7 pt text at
the 183 mm double-column width, bold lowercase panel letters (descriptions
live in the caption, not on the figure), a single muted colourblind-safe
palette (Okabe-Ito), no gridlines, and outward ticks with top/right spines
removed.

Layout (panels are grouped thematically):
- ``tv_dataset.pdf`` (2 x 3): mission-reported subset composition, evidence
  coverage, tier structure, stable-window controls, SLR precision, and the
  tier A/B equivalence checks.
- ``tv_external_validation.pdf`` (1 x 3): benchmark match rates, match
  offsets, and response by match status.
- ``tv_event_response.pdf`` (2 x 3): response distribution, TLE-vs-orbit
  agreement, specificity, sign agreement, Kozai convention check, SLR O-C
  audit.
- ``tv_consistency.pdf`` (2 x 3): SGP4 propagation residuals, sigma-model
  calibration, TLE-POD biases (reference subset) plus the Starlink
  shell/TLE-ephemeris audits (operational subset).
- ``tv_anatomy_a.pdf`` / ``tv_anatomy_b.pdf``: full-width single-panel
  event anatomies (legibility at final size).

Usage (from the repository root, via the experiment dispatcher):

    python3 scripts/experiments/run_experiments.py paper-tv-figures
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from benchmarking.experiment_params import (
    DEFAULT_SIGMA_MODEL,
    EARTH_RADIUS_KM,
    SMA_SHIFT_NOISE_FLOOR_M,
    SHELL_TOLERANCE_DEG,
)

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
})

# Nature double-column width is 183 mm = 7.2 in.
PANEL_H = 2.4   # inches per panel row

# Muted, colourblind-safe palette (Okabe-Ito). One palette everywhere.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
BROWN = "#8C6D5D"
GREY = "#7F7F7F"

TIER_COLORS = {"A": GREEN, "B": ORANGE, "C": PURPLE}
# 8-colour sequence for per-satellite series (Starlink consistency panels).
SAT8 = [BLUE, ORANGE, GREEN, VERMILION, PURPLE, BROWN, SKY, GREY]

REPO = Path(__file__).resolve().parents[2]
VAL = REPO / "experiments" / "validation"
OPS = REPO / "dataset" / "operational" / "starlink"
# Starlink analysis tables (Q4 distribution/consistency) live with the code,
# not the dataset: the released subset carries only the data + QC audits.
OPS_TABLES = REPO / "experiments" / "starlink"
OUT = REPO / "arxiv" / "MAD_LEO_paper" / "images"
OUT.mkdir(parents=True, exist_ok=True)

SAT_NAMES = {
    "cryosat-2": "CryoSat-2", "hy-2a": "HY-2A", "jason-1": "Jason-1",
    "jason-2": "Jason-2", "jason-3": "Jason-3", "saral": "SARAL",
    "sentinel-3a": "Sentinel-3A", "sentinel-3b": "Sentinel-3B",
    "sentinel-6a": "Sentinel-6A", "swot": "SWOT", "topex-poseidon": "TOPEX/Poseidon",
}
# Consistent per-target palette (tab20 dark colors + one extra).
_PALETTE = [plt.matplotlib.colormaps["tab20"](i) for i in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 5)]


def sat_order(by_events: pd.Series) -> list[str]:
    return list(by_events.sort_values(ascending=False).index)


def sat_colors(order: list[str]) -> dict[str, tuple]:
    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(order)}


def new_figure_2x3():
    fig = plt.figure(figsize=(7.2, PANEL_H * 2 + 0.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.62, hspace=0.60)
    return fig, gs


def new_figure_1x3():
    fig = plt.figure(figsize=(7.2, PANEL_H + 0.3))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.50)
    return fig, gs


def panel(ax, letter: str, title: str | None = None):
    """Nature-style panel tag: bold lowercase letter only (descriptions live
    in the caption). ``title`` is accepted for call-site documentation."""
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=9, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.0, width=0.6, pad=1.5)


def save(fig, name: str):
    path = OUT / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"saved {path}  ({path.stat().st_size/1024:.0f} KiB)")


def log_violin(ax, groups: list[np.ndarray], labels: list[str], colors: list[str]):
    """Violin plots with the KDE computed in log10 space, linear log axis."""
    logs = [np.log10(g[g > 0]) for g in groups]
    parts = ax.violinplot(logs, showmedians=False, showextrema=False, widths=0.7)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.8)
        body.set_edgecolor("none")
    for i, g in enumerate(groups, start=1):
        # Marker and label use the plain median of the FULL group (dropna
        # only, zeros included) so the annotated value matches the published
        # tables; the KDE above is computed in log10 space (positive values
        # only) because the axis is logarithmic.
        med = float(np.median(g[~np.isnan(g)]))
        ax.scatter([i], [np.log10(med)], marker="_", s=34, color="black", zorder=5,
                   linewidth=1.2)
        ax.text(i + 0.30, np.log10(med), f"{med:.1f} m",
                va="center", fontsize=6)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ticks = [-2, -1, 0, 1, 2, 3]
    ax.set_yticks(ticks, [rf"$10^{{{t}}}$" for t in ticks])
    ax.set_ylabel(r"$|\Delta a|$ (m)")


# ================================================================ FIGURE 2 ==
def fig_dataset():
    """Mission-reported subset: composition, coverage, tiers, controls."""
    ann = pd.read_csv(VAL / "maneuver_annotation_summary.csv")
    ann = ann[ann.sat_id != "ALL"]
    ev = pd.read_csv(VAL / "maneuver_evidence_alignment_summary.csv")
    ev = ev[ev.sat_id != "ALL"]
    tier = pd.read_csv(VAL / "maneuver_confidence_tier_summary.csv")

    order = sat_order(ann.set_index("sat_id")["mission_reported_event_count"])
    y = np.arange(len(order))[::-1]

    fig, gs = new_figure_2x3()

    # (a) event count per satellite, with all-source-aligned subset
    ax = fig.add_subplot(gs[0, 0])
    total = ann.set_index("sat_id")["mission_reported_event_count"].reindex(order)
    aligned = ev.set_index("sat_id")["all_source_aligned_count"].reindex(order)
    ax.barh(y, total, height=0.62, color=SKY, label="mission-reported")
    ax.barh(y, aligned, height=0.62, color=BLUE, label="three-source aligned")
    for yi, v in zip(y, total):
        ax.text(v + 4, yi, str(int(v)), va="center", fontsize=6)
    ax.set_yticks(y, [SAT_NAMES[s] for s in order])
    ax.set_xlabel("Event windows")
    ax.set_xlim(0, 275)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), frameon=False,
              ncol=1, fontsize=5.5, handletextpad=0.3)
    panel(ax, "a", "Event windows per target")

    # (b) evidence coverage per satellite
    ax = fig.add_subplot(gs[0, 1])
    h = 0.26
    for k, (col, lab, c) in enumerate([
        ("tle_covered_count", "TLE", BLUE),
        ("orbit_covered_count", "POD", ORANGE),
        ("slr_covered_count", "SLR", GREEN),
    ]):
        vals = ev.set_index("sat_id")[col].reindex(order)
        ax.barh(y + (1 - k) * h, vals, height=h * 0.9, color=c, label=lab)
    ax.set_yticks(y, [SAT_NAMES[s] for s in order])
    ax.set_xlabel("Windows covered")
    ax.set_xlim(0, 275)
    ax.legend(loc="lower right", bbox_to_anchor=(0.97, 0.03), frameon=False,
              ncol=1, title=None, handletextpad=0.3, labelspacing=0.3)
    panel(ax, "b", "Evidence coverage per target")

    # (c) tier composition
    ax = fig.add_subplot(gs[0, 2])
    tp = (tier[tier.sat_id != "ALL"]
          .pivot_table(index="sat_id", columns="confidence_tier",
                       values="event_count", aggfunc="sum")
          .reindex(order)[["A", "B", "C"]].fillna(0))
    left = np.zeros(len(order))
    for t in ["A", "B", "C"]:
        vals = tp[t].to_numpy()
        ax.barh(y, vals, left=left, height=0.62, color=TIER_COLORS[t], label=f"Tier {t}")
        for yi, v, l in zip(y, vals, left):
            if v >= 24:
                ax.text(l + v / 2, yi, str(int(v)), va="center", ha="center",
                        fontsize=6, color="white")
        left += vals
    for yi, v in zip(y, left):
        ax.text(v + 4, yi, str(int(v)), va="center", fontsize=6)
    ax.set_yticks(y, [SAT_NAMES[s] for s in order])
    ax.set_xlabel("Event windows")
    ax.set_xlim(0, 275)
    ax.legend(loc="lower right", frameon=False, ncol=1)
    panel(ax, "c", "Confidence-tier composition")

    # (d) stable no-event control windows per satellite
    ax = fig.add_subplot(gs[1, 0])
    st = pd.read_csv(VAL / "stable_windows_summary.csv")
    st_total = st.set_index("sat_id")["stable_window_count"].reindex(order)
    st_aligned = st.set_index("sat_id")["aligned_window_count"].reindex(order)
    ax.barh(y, st_total, height=0.62, color=SKY, label="stable windows")
    ax.barh(y, st_aligned, height=0.62, color=BLUE, label="three-source aligned")
    for yi, v in zip(y, st_total):
        ax.text(v + 4, yi, str(int(v)), va="center", fontsize=6)
    ax.set_yticks(y, [SAT_NAMES[s] for s in order])
    ax.set_xlabel("Stable windows")
    ax.set_xlim(0, 275)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), frameon=False,
              ncol=1, fontsize=5.5, handletextpad=0.3)
    panel(ax, "d", "Stable control windows per target")

    # (e) SLR NP precision per target
    ax = fig.add_subplot(gs[1, 1])
    slr = pd.read_csv(VAL / "slr_distribution_summary.csv").sort_values("sigma_median_mm")
    yy = np.arange(len(slr))
    ax.barh(yy, slr.sigma_median_mm, height=0.62, color=BLUE,
            xerr=slr.sigma_iqr_mm / 2, error_kw=dict(elinewidth=0.6, capsize=1.5))
    for yi, (_, row) in zip(yy, slr.iterrows()):
        ax.text(row.sigma_median_mm + row.sigma_iqr_mm / 2 + 0.45, yi,
                f"{row.sigma_median_mm:.1f}", va="center", fontsize=6)
    ax.set_yticks(yy, [SAT_NAMES.get(s, s) for s in slr.sat_id])
    ax.set_xlabel("Normal-point sigma, median (mm)")
    ax.set_xlim(0, 22)
    panel(ax, "e", "SLR normal-point precision")

    # (f) tier comparison: 3-sigma coverage + sign agreement
    ax = fig.add_subplot(gs[1, 2])
    tcov = pd.read_csv(VAL / "tier_sigma_coverage.csv").set_index("confidence_tier")
    tsgn = pd.read_csv(VAL / "tier_sign_agreement.csv").set_index("confidence_tier")
    metrics = [
        ("3$\\sigma$ coverage", tcov.within_3sigma_fraction, tcov.sample_count),
        ("sign agreement", tsgn.sign_agreement, tsgn.sample_count),
    ]
    xx = np.arange(len(metrics))
    w = 0.32
    # Point estimates, not bars: on the zoomed (0.85-1.04) axis bar lengths
    # would visually exaggerate the tier differences.
    for k, (tier_name, c) in enumerate([("A", TIER_COLORS["A"]), ("B", TIER_COLORS["B"])]):
        vals = [m[1].get(tier_name, np.nan) for m in metrics]
        ax.plot(xx + (k - 0.5) * w, vals, "o", color=c, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5,
                label=f"Tier {tier_name}")
        ha, dx = ("right", -0.045) if k == 0 else ("left", 0.045)
        for xi, v in zip(xx + (k - 0.5) * w, vals):
            ax.text(xi + dx, v, f"{v*100:.1f}%", ha=ha, va="center", fontsize=5.5)
    labels = [f"{m[0]}\n(n={m[2].get('A', 0)}/{m[2].get('B', 0)})" for m in metrics]
    ax.axhline(1.0, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_xticks(xx, labels)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.85, 1.04)
    ax.set_ylabel("Fraction")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    panel(ax, "f", "Tier A vs tier B checks")

    save(fig, "tv_dataset")


# ================================================================ FIGURE 3 ==
def fig_external():
    """External benchmark cross-validation (tolerance sweep is text-only)."""
    xv = pd.read_csv(VAL / "external_benchmark_crossvalidation.csv")
    xv = xv[xv.sat_id != "ALL"].copy()
    # The benchmark predates SWOT (zero benchmark events), so its match rates
    # are undefined rather than zero — exclude it from the panel.
    xv = xv[xv.benchmark_events > 0]
    xv["forward"] = xv.benchmark_matched / xv.benchmark_events
    xv["reverse"] = xv.madleo_matched / xv.madleo_events
    xv["in_span"] = xv.madleo_matched_in_benchmark_span / xv.madleo_events_in_benchmark_span
    order = sat_order(xv.set_index("sat_id")["madleo_events"])
    y = np.arange(len(order))[::-1]

    fig, gs = new_figure_1x3()

    # (a) match rates
    ax = fig.add_subplot(gs[0, 0])
    h = 0.26
    series = [("forward", "forward", BLUE),
              ("reverse", "reverse (all)", ORANGE),
              ("in_span", "reverse (in-span)", GREEN)]
    for k, (col, lab, c) in enumerate(series):
        vals = xv.set_index("sat_id")[col].reindex(order).fillna(0.0)
        ax.barh(y + (1 - k) * h, vals, height=h * 0.9, color=c, label=lab)
    ax.set_yticks(y, [SAT_NAMES[s] for s in order])
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Match rate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, fontsize=5.5, handletextpad=0.25, columnspacing=0.4,
              borderpad=0.1)
    panel(ax, "a", "Benchmark match rates per target")

    # (b) signed match-offset histogram (pre-binned table)
    ax = fig.add_subplot(gs[0, 1])
    off = pd.read_csv(VAL / "external_benchmark_match_offsets.csv")
    off = off[off.sat_id == "ALL"]
    widths = off.offset_bin_end_min - off.offset_bin_start_min
    centers = (off.offset_bin_start_min + off.offset_bin_end_min) / 2
    cap = 60.0
    ax.bar(centers, np.minimum(off.matched_count, cap), width=widths * 0.92,
           color=BLUE, edgecolor="none")
    tall = off[off.matched_count > cap]
    for _, r in tall.iterrows():
        cx = (r.offset_bin_start_min + r.offset_bin_end_min) / 2
        ax.text(cx, cap + 1.5, f"{int(r.matched_count)}", ha="center", va="bottom",
                fontsize=6, color="black")
    ax.axvline(1440, color=VERMILION, linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 68)
    n_day = off[off.offset_bin_start_min >= 1380].matched_count.sum()
    ax.annotate(f"+1 d benchmark offset:\n{n_day} of {int(off.matched_count.sum())} pairs "
                f"within 1 h", xy=(0.03, 0.97), xycoords="axes fraction",
                fontsize=6, color="black", ha="left", va="top")
    ax.set_xlabel("Signed offset, benchmark $-$ MAD-LEO (min)")
    ax.set_ylabel("Matched pairs")
    panel(ax, "b", "Match offsets (+1 d clustering)")

    # (c) response comparison by match/span status
    ax = fig.add_subplot(gs[0, 2])
    pe = pd.read_csv(VAL / "external_benchmark_crossvalidation_per_event.csv")
    pe["grp"] = np.where(~pe.within_benchmark_span, "out-of-span",
                         np.where(pe.matched_within_1d.fillna(False),
                                  "matched\nin-span", "unmatched\nin-span"))
    names = ["matched", "unmatched", "out-of-span"]
    keys = ["matched\nin-span", "unmatched\nin-span", "out-of-span"]
    groups, counts = [], []
    for key in keys:
        g = pe[pe.grp == key].abs_delta_sma_m.dropna().to_numpy()
        groups.append(g)
        counts.append(len(g))
    log_violin(ax, groups, [f"{lab}\n(n={c})" for lab, c in zip(names, counts)],
               [GREEN, ORANGE, GREY])
    ax.set_xlim(0.4, 4.05)
    ax.set_ylim(-2.2, 3.6)
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", fontsize=6)
    panel(ax, "c", "Response by match status")

    save(fig, "tv_external_validation")


# ================================================================ FIGURE 4 ==
def _deming(x, y, sx, sy):
    sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    lam = (sy / sx) ** 2
    a = syy - lam * sxx
    root = np.sqrt(a**2 + 4 * lam * sxy**2)
    b = (a + root) / (2 * sxy) if a >= 0 else 2 * lam * sxy / (root - a)
    return b


def fig_event_response():
    """Event response plus the convention (Kozai) and SLR O-C audits."""
    ev = pd.read_csv(VAL / "maneuver_event_response_validation.csv")
    st = pd.read_csv(VAL / "stable_window_response_validation.csv")
    sg = pd.read_csv(VAL / "sign_agreement_by_magnitude.csv")

    ev_abs = (ev.delta_sma_km * 1000.0).abs().dropna().to_numpy()
    st_abs = (st.delta_sma_km * 1000.0).abs().dropna().to_numpy()

    fig, gs = new_figure_2x3()

    # (a) TLE |da| histogram + ECDF, log x
    ax = fig.add_subplot(gs[0, 0])
    v = ev_abs[ev_abs > 0]
    bins = np.logspace(-2, np.log10(v.max() * 1.05), 45)
    ax.hist(v, bins=bins, color=BLUE, alpha=0.75, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\Delta a|$ TLE (m)")
    ax.set_ylabel("Event windows")
    axt = ax.twinx()
    xs = np.sort(v)
    axt.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="0.25", linewidth=1.0)
    axt.set_ylim(0, 1.02)
    axt.set_ylabel("ECDF", color="0.25")
    axt.tick_params(axis="y", colors="0.25", labelsize=6, length=2.0, width=0.6)
    axt.spines["top"].set_visible(False)
    med = np.median(v)
    frac_below_500 = float((v < 500.0).mean())
    ax.axvline(med, color="black", linestyle=":", linewidth=0.7)
    ax.annotate(f"median {med:.1f} m\n{frac_below_500 * 100:.0f}% < 500 m", xy=(0.04, 0.96),
                xycoords="axes fraction", ha="left", va="top", fontsize=6)
    panel(ax, "a", "TLE response distribution")

    # (b) TLE vs POD scatter
    ax = fig.add_subplot(gs[0, 1])
    dual = ev[(ev.tle_status == "computed") & ev.delta_sma_km.notna()
              & ev.orbit_mean_sma_shift_m.notna()]
    x = dual.delta_sma_km.to_numpy() * 1000.0
    y = dual.orbit_mean_sma_shift_m.to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    ols = np.polyfit(x, y, 1)
    dem = _deming(x, y, SMA_SHIFT_NOISE_FLOOR_M["tle_median"], SMA_SHIFT_NOISE_FLOOR_M["orbit_median"])
    lim = 1500.0
    inside = (np.abs(x) <= lim) & (np.abs(y) <= lim)
    outside = ~inside
    ax.scatter(x[inside], y[inside], s=6, color=BLUE, alpha=0.6,
               edgecolor="none", rasterized=True)
    ax.scatter(np.clip(x[outside], -lim, lim), np.clip(y[outside], -lim, lim),
               s=10, facecolor="none", edgecolor=BLUE, linewidth=0.6,
               marker="^", label=f"beyond $\\pm${lim/1000:.1f} km (n={outside.sum()})")
    xx = np.array([-lim, lim])
    ax.plot(xx, xx, color="0.6", linewidth=0.6, linestyle=":", label="1:1")
    ax.plot(xx, np.polyval(ols, xx), color="black", linewidth=0.8, linestyle="--",
            label=f"OLS {ols[0]:.2f}")
    ax.plot(xx, dem * xx + (y.mean() - dem * x.mean()), color=VERMILION, linewidth=0.8,
            linestyle="-", label=f"Deming {dem:.2f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta a$ TLE (m)")
    ax.set_ylabel(r"$\Delta \bar a$ orbit (m)")
    ax.annotate(f"n = {len(x)}\nr = {r:.2f}", xy=(0.04, 0.96),
                xycoords="axes fraction", va="top", fontsize=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              frameon=False, fontsize=5.5, columnspacing=0.6)
    panel(ax, "b", "TLE vs precise-orbit response")

    # (c) specificity violin
    ax = fig.add_subplot(gs[0, 2])
    log_violin(ax, [ev_abs, st_abs],
               [f"events\n(n={len(ev_abs)})", f"stable\n(n={len(st_abs)})"],
               [BLUE, GREY])
    ax.set_xlim(0.4, 3.15)
    ax.set_ylim(-2.2, 3.9)
    panel(ax, "c", "Specificity: events vs stable")

    # (d) sign agreement by magnitude band
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(range(len(sg)), sg["mean"], color=BLUE, width=0.62)
    for i, (_, row) in enumerate(sg.iterrows()):
        ax.text(i, row["mean"] + 0.015,
                f"{row['mean']*100:.0f}%\nn={int(row['count'])}",
                ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(sg)), sg.response_band)
    ax.set_ylim(0, 1.22)
    ax.set_xlabel(r"TLE $|\Delta a|$ band")
    ax.set_ylabel("TLE-orbit sign agreement")
    panel(ax, "d", "Sign agreement by magnitude")

    # (e) Kozai convention comparison
    ax = fig.add_subplot(gs[1, 1])
    kz = pd.read_csv(VAL / "kozai_comparison_per_event.csv")
    d = kz.difference_kepler_vs_sgp4_m.abs().dropna().to_numpy()
    d = d[d > 0]
    bins = np.logspace(-3, np.log10(d.max() * 1.05), 45)
    ax.hist(d, bins=bins, color=BLUE, alpha=0.75, edgecolor="none")
    ax.set_xscale("log")
    med, sub = np.median(d), (d < 1).mean()
    ax.axvline(med, color="black", linestyle=":", linewidth=0.7)
    ax.annotate(f"median {med*100:.1f} cm\n{sub*100:.0f}% sub-meter",
                xy=(0.97, 0.72), xycoords="axes fraction", ha="right", fontsize=6)
    ax.set_xlabel(r"$|$Kepler $-$ SGP4/Brouwer $\Delta a|$ (m)")
    ax.set_ylabel("Event windows")
    panel(ax, "e", "Kozai convention comparison")

    # (f) SLR O-C shift vs RMS; extreme outliers drawn as edge markers
    ax = fig.add_subplot(gs[1, 2])
    oc = pd.read_csv(VAL / "slr_oc_residuals.csv")
    oc = oc[oc.oc_shift_m.notna() & oc.oc_rms_after_m.notna()]
    order = list(oc.groupby("sat_id").size().sort_values(ascending=False).index)
    colors = sat_colors(order)
    top = oc.loc[oc.oc_shift_m <= 1000, "oc_shift_m"].max()
    ylim_hi = max(top * 1.30, 50.0)
    ylim_lo = min(oc.oc_shift_m.min() * 1.3, -10.0)
    for s in order:
        g = oc[oc.sat_id == s]
        inside = g[g.oc_shift_m <= 1000]
        ax.scatter(inside.oc_rms_after_m, inside.oc_shift_m, s=9, color=colors[s],
                   alpha=0.8, edgecolor="none", label=SAT_NAMES.get(s, s),
                   rasterized=True)
        out = g[g.oc_shift_m > 1000]
        if len(out):
            ax.scatter(out.oc_rms_after_m, np.full(len(out), ylim_hi * 0.94),
                       marker="^", s=16, facecolor="none", edgecolor=colors[s],
                       linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="-")
    big = oc[oc.oc_shift_m > 1000]
    for _, out_row in big.iterrows():
        ax.annotate(f"Jason-3: {out_row.oc_shift_m/1000:.1f} km (off scale)",
                    xy=(out_row.oc_rms_after_m, ylim_hi * 0.94),
                    xytext=(-6, 2), textcoords="offset points", fontsize=6,
                    ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_xlabel("Post-event O-C RMS (m)")
    ax.set_ylabel("O-C median shift (m)")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.42), frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.9,
              fontsize=5, ncol=1, handletextpad=0.2, labelspacing=0.3,
              borderpad=0.25)
    panel(ax, "f", "SLR O-C shift vs RMS")

    save(fig, "tv_event_response")
    return r, ols[0], dem, len(x)


# ============================================================ FIGURES 5/6 ==
def _embed_pdf(ax, path: Path, letter: str | None = None, zoom: float = 4.0):
    import pymupdf
    page = pymupdf.open(path)[0]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    ax.imshow(img)
    ax.set_axis_off()
    if letter:
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=9, pad=2)


def fig_anatomy_a():
    """Standalone full-width Sentinel-3A event anatomy."""
    fig = plt.figure(figsize=(7.2, 3.0))
    _embed_pdf(fig.add_subplot(111), REPO / "arxiv" / "MAD_LEO_paper" / "images" / "c_q2_maneuver_anatomy_a.pdf")
    save(fig, "tv_anatomy_a")


def fig_anatomy_b():
    """Standalone full-width SWOT event anatomy."""
    fig = plt.figure(figsize=(7.2, 3.0))
    _embed_pdf(fig.add_subplot(111), REPO / "arxiv" / "MAD_LEO_paper" / "images" / "c_q2_maneuver_anatomy_b.pdf")
    save(fig, "tv_anatomy_b")


# ================================================================ FIGURE 7 ==
def fig_consistency():
    """Cross-source absolute consistency: harmonization (reference subset)
    plus the Starlink operational-slice audits."""
    fig, gs = new_figure_2x3()

    # (a) SGP4 propagation residual curve + sigma model
    # Tail bins (24-48 h, n=23; 48-96 h, n=5) are pooled into a single
    # (24,96] bin recomputed from the per-window state estimates, because
    # the small per-bin samples are unstable (reviewer note); the first
    # three bins reproduce sgp4_propagation_residuals_clean.csv exactly.
    ax = fig.add_subplot(gs[0, 0])
    sg = pd.read_csv(VAL / "sgp4_propagation_residuals_clean.csv")
    mids, med, p75, ns = [], [], [], []
    for _, row in sg.iterrows():
        lo, hi = map(float, re.findall(r"[\d.]+", row.propagation_bin_hours))
        if lo >= 24.0:
            continue
        mids.append((lo + hi) / 2)
        med.append(row["median"])
        p75.append(row.p75)
        ns.append(int(row["count"]))
    se = pd.read_csv(VAL / "maneuver_event_state_estimates.csv")
    ok = se[(se.tle_state_status == "computed") & (se.orbit_state_status == "computed")
            & se.orbit_extrapolation_seconds.isna()]
    res = np.sqrt((ok.itrs_x_m - ok.orbit_x_m) ** 2
                  + (ok.itrs_y_m - ok.orbit_y_m) ** 2
                  + (ok.itrs_z_m - ok.orbit_z_m) ** 2) / 1000.0
    tail = (ok.tle_propagation_seconds / 3600.0 > 24.0) & (ok.tle_propagation_seconds / 3600.0 <= 96.0)
    mids.append(60.0)
    med = np.append(med, res[tail.to_numpy()].median())
    p75 = np.append(p75, res[tail.to_numpy()].quantile(0.75))
    ns.append(int(tail.sum()))
    med, p75 = np.array(med), np.array(p75)
    ax.axvspan(48, 82, color="0.94", zorder=0)
    ax.text(65, 0.92, "beyond validated\nrange ($>$48 h)", fontsize=6,
            ha="center", va="top", color="0.35")
    ax.errorbar(mids, med, yerr=[np.zeros_like(p75), p75 - med], fmt="o-",
                color=BLUE, markersize=3.5, capsize=2, zorder=3,
                elinewidth=0.7, label="median (bar: p75)")
    h = np.linspace(0, 80, 100)
    sigma_base = DEFAULT_SIGMA_MODEL["base_km"]
    sigma_growth = DEFAULT_SIGMA_MODEL["growth_km_per_24h"]
    ax.plot(h, sigma_base + sigma_growth * h / 24, color="black", linewidth=1.0,
            linestyle="--", zorder=4,
            label=rf"$\sigma = {sigma_base} + {sigma_growth}\,h/24$")
    for i, (m, v, n) in enumerate(zip(mids, med, ns)):
        if i == 0:
            ax.annotate(f"n={n}", (m, v), textcoords="offset points",
                        xytext=(2, -13), ha="left", fontsize=6)
            continue
        xytext = (0, 6) if i == 1 else (0, -13)
        ax.annotate(f"n={n}", (m, v), textcoords="offset points", xytext=xytext,
                    ha="center", fontsize=6)
    ax.set_xlabel("Propagation age bin midpoint (h)")
    ax.set_ylabel("Residual vs POD (km)")
    ax.set_xlim(0, 82)
    ax.set_ylim(0.35, 1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False,
              ncol=1, fontsize=6)
    panel(ax, "a", "SGP4 propagation residuals")

    # (b) sigma calibration curve
    ax = fig.add_subplot(gs[0, 1])
    sc = pd.read_csv(VAL / "sigma_calibration_curve.csv")
    sc = sc[sc.sigma_eval_group == "all"].sort_values("k_sigma")
    xx = np.arange(3)
    w = 0.36
    ax.bar(xx - w / 2, sc.nominal_coverage, width=w, color="0.75",
           label="nominal (Gaussian)")
    ax.bar(xx + w / 2, sc.empirical_coverage, width=w, color=BLUE,
           label="empirical")
    for xi, nom, emp in zip(xx, sc.nominal_coverage, sc.empirical_coverage):
        ax.text(xi - w / 2, nom + 0.012, f"{nom*100:.0f}%", ha="center", fontsize=6)
        ax.text(xi + w / 2, emp + 0.012, f"{emp*100:.0f}%", ha="center", fontsize=6)
    ax.set_xticks(xx, [r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Coverage fraction")
    ax.text(0.03, 0.97, f"n = {int(sc.sample_count.iloc[0])} windows",
            transform=ax.transAxes, fontsize=6, ha="left", va="top")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False,
              ncol=2, fontsize=6, columnspacing=0.6, handletextpad=0.3)
    panel(ax, "b", "Sigma-model calibration")

    # (c) bias calibration R/A/C
    ax = fig.add_subplot(gs[0, 2])
    b = pd.read_csv(VAL / "source_bias_calibration.csv")
    g = b.groupby("sat_id")[["radial_bias_m", "along_track_bias_m",
                             "cross_track_bias_m"]].median()
    g = g.reindex(["radial_bias_m", "along_track_bias_m", "cross_track_bias_m"], axis=1)
    g = g.loc[g.along_track_bias_m.abs().sort_values().index]
    yy = np.arange(len(g))
    h = 0.26
    for k, (col, lab, c) in enumerate([
        ("radial_bias_m", "radial", BLUE),
        ("along_track_bias_m", "along-track", VERMILION),
        ("cross_track_bias_m", "cross-track", GREEN),
    ]):
        ax.barh(yy + (1 - k) * h, g[col], height=h * 0.9, color=c, label=lab)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(yy, [SAT_NAMES.get(s, s) for s in g.index])
    ax.set_xlabel("TLE $-$ POD bias, median (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False,
              ncol=3, fontsize=6, columnspacing=0.5, handletextpad=0.3)
    panel(ax, "c", "Per-target TLE-POD bias")

    # (d) Starlink plane-family scatter + reference missions
    ax = fig.add_subplot(gs[1, 0])
    tel = pd.read_parquet(OPS / "tle_elements.parquet",
                          columns=["inclination_rad", "mean_motion_rad_per_min"])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tel), size=min(9000, len(tel)), replace=False)
    tel = tel.iloc[np.sort(idx)]
    inc = np.degrees(tel.inclination_rad.to_numpy())
    mu = 398600.4418
    n_s = tel.mean_motion_rad_per_min.to_numpy() / 60.0
    alt = (mu / n_s**2) ** (1 / 3.0) - EARTH_RADIUS_KM
    # Color by orbital-plane family (inclination band). The sub-480 km
    # population at 43/53.2 deg is deployment/transfer traffic; the altitude
    # spread itself is shown by the scatter.
    shells = [43.0, 53.0, 53.2, 70.0, 97.6]
    shell_cols = [BLUE, PURPLE, GREEN, VERMILION, ORANGE]
    assigned = np.zeros(len(inc), dtype=bool)
    for sh, c in zip(shells, shell_cols):
        m = np.abs(inc - sh) <= SHELL_TOLERANCE_DEG
        assigned |= m
        ax.scatter(inc[m], alt[m], s=1.8, color=c, alpha=0.45, edgecolor="none",
                   label=f"{sh:g}$^\\circ$", rasterized=True)
    if (~assigned).any():
        ax.scatter(inc[~assigned], alt[~assigned], s=1.8, color="0.6", alpha=0.45,
                   edgecolor="none", rasterized=True)
    ax.annotate("sub-480 km: deployment / transfer",
                xy=(47.0, 350), xytext=(56, 215), fontsize=6, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", linewidth=0.4, color="0.4"))
    tle = pd.read_csv(VAL / "tle_element_distribution_summary.csv")
    ax.scatter(tle.inclination_median_deg, tle.sma_altitude_median_km, s=11,
               marker="D", color="black", edgecolors="white", linewidths=0.4,
               zorder=6, label="reference (11)")
    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlim(38, 106)
    ax.set_ylim(150, 1520)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
                    frameon=False, fontsize=5.5, handletextpad=0.3,
                    borderpad=0.2, labelspacing=0.3, columnspacing=0.6)
    for handle in leg.legend_handles:
        handle.set_sizes([10.0])
        handle.set_alpha(1.0)
    panel(ax, "d", "Starlink planes vs reference missions")

    # (e) TLE-ephemeris residual ECDF per satellite
    ax = fig.add_subplot(gs[1, 1])
    smp = pd.read_csv(OPS_TABLES / "starlink_tle_ephemeris_consistency_samples.csv")
    sats = sorted(smp.sat_id.unique())
    cols = SAT8[:len(sats)]
    for c, s in zip(cols, sats):
        v = np.sort(smp[smp.sat_id == s].residual_km.to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), linewidth=0.9, color=c,
                label=str(s))
    ax.set_xscale("log")
    ax.set_xlabel("TLE-ephemeris position residual (km)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="upper left", frameon=False, fontsize=6, ncol=1,
              title="NORAD ID", title_fontsize=6, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.35, handlelength=1.6)
    panel(ax, "e", "TLE-ephemeris residual ECDF")

    # (f) per-satellite median residual with age
    ax = fig.add_subplot(gs[1, 2])
    cons = pd.read_csv(OPS_TABLES / "starlink_tle_ephemeris_consistency.csv")
    cons = cons.sort_values("residual_median_km")
    yy = np.arange(len(cons))
    ax.barh(yy, cons.residual_median_km, height=0.62,
            color=[cols[sats.index(s)] for s in cons.sat_id])
    for yi, (_, row) in zip(yy, cons.iterrows()):
        ax.text(row.residual_median_km + 0.3, yi,
                f"{row.residual_median_km:.1f} km, {row.propagation_age_median_hours:.1f} h",
                va="center", fontsize=6)
    ax.set_yticks(yy, cons.sat_id)
    ax.set_xlim(0, 20.0)
    ax.set_xlabel("Median TLE-ephemeris residual (km)")
    panel(ax, "f", "Starlink residual per satellite")

    save(fig, "tv_consistency")


# ==================================================================== main ==
def main(argv: list[str] | None = None):
    # argparse guard: without it, `--help` (or any stray argument) silently
    # triggered a full figure build instead of printing usage.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    fig_dataset()
    fig_external()
    stats = fig_event_response()
    print(f"event_response scatter: n={stats[3]}, r={stats[0]:.3f}, "
          f"OLS={stats[1]:.3f}, Deming={stats[2]:.3f}")
    fig_anatomy_a()
    fig_anatomy_b()
    fig_consistency()


if __name__ == "__main__":
    main()
