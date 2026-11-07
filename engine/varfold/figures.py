"""Publication figures (matplotlib). Every figure is written as PDF (vector) and PNG (300 dpi).

Colour roles follow one validated palette (shared with the web app):
benign = blue, pathogenic = orange, VUS = aqua — all-pairs CVD-safe for up to
three classes. Cancer-context variants take the palette's fourth hue (yellow) and
population variants neutral grey (background, like the WT). Aqua and yellow sit
below 3:1 contrast on white, so classes are always also named on an axis or legend. Structure confidence uses the AlphaFold pLDDT
convention so structural biologists read it without a legend lookup.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .stats import roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CLASS_COLORS = {"benign": "#2a78d6", "pathogenic": "#eb6834", "vus": "#1baf7a", "cancer": "#eda100",
                "population": "#898781"}
CLASS_NAMES = {"benign": "Benign / LB", "pathogenic": "Pathogenic / LP", "vus": "VUS",
               "cancer": "Cancer context", "population": "gnomAD carriers"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ROC_COLORS = {"delta_local_plddt": SERIES[0], "llr": SERIES[1], "blosum62": SERIES[2], "wt_site_plddt": SERIES[3],
              "alphamissense": INK2}
PLDDT_BANDS = [(90, 100, "#0053d6", "Very high (>90)"), (70, 90, "#65cbf3", "Confident (70–90)"),
               (50, 70, "#ffdb13", "Low (50–70)"), (0, 50, "#ff7d45", "Very low (<50)")]
DIVERGING = LinearSegmentedColormap.from_list("delta", ["#e34948", "#f0efec", "#256abf"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42,
})


def _save(fig, out_dir: Path, name: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        files.append(path.name)
    plt.close(fig)
    return files


def wt_profile(wt_plddt: np.ndarray, start: int, metrics: pd.DataFrame, annotations: list[dict],
               out_dir: Path) -> list[str]:
    x = np.arange(start, start + len(wt_plddt))
    fig, (ax, rug) = plt.subplots(2, 1, figsize=(7.2, 3.0), sharex=True,
                                  gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08})
    for lo, hi, color, name in PLDDT_BANDS:
        ax.axhspan(lo, hi, color=color, alpha=0.08, lw=0)
        ax.text(x[-1] + 2, (lo + hi) / 2, name.split(" (")[0], color=MUTED, fontsize=6.5, va="center")
    ax.plot(x, wt_plddt, color=INK, lw=1.6)
    ax.set_ylim(0, 108)
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.set_ylabel("pLDDT")
    ax.set_title("Wild-type ESMFold confidence per residue")
    helices = [a for a in annotations if a["type"] == "Helix"]
    for a in helices:
        ax.plot([a["start"], a["end"]], [104, 104], color=INK2, lw=3, solid_capstyle="butt")
    if helices:
        ax.text(x[0], 106.5, "α-helices (PDB-derived)", color=INK2, fontsize=7, va="bottom")
    # Clinical variant positions as a labelled rug strip under the profile.
    rows = [c for c in ("pathogenic", "benign") if (metrics["label"] == c).any()]
    for y, cls in enumerate(rows):
        pos = metrics.loc[metrics["label"] == cls, "position"].unique()
        rug.scatter(pos, np.full(len(pos), y), marker="|", s=60, lw=1.2, color=CLASS_COLORS[cls])
    rug.set_yticks(range(len(rows)), [CLASS_NAMES[c] for c in rows], fontsize=7)
    rug.set_ylim(-0.7, len(rows) - 0.3)
    rug.grid(False)
    rug.tick_params(axis="y", length=0)
    rug.spines["left"].set_visible(False)
    rug.set_xlabel("Residue (UniProt numbering)")
    return _save(fig, out_dir, "fig1_wt_plddt_profile")


def delta_by_class(metrics: pd.DataFrame, null_floor: float, out_dir: Path, seed: int) -> list[str]:
    classes = [c for c in ("benign", "population", "pathogenic", "cancer", "vus") if (metrics["label"] == c).any()]
    if not classes:
        return []
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for k, cls in enumerate(classes):
        vals = metrics.loc[metrics["label"] == cls, "delta_local_plddt"].to_numpy()
        ax.boxplot(vals, positions=[k], widths=0.5, showfliers=False, medianprops={"color": INK, "lw": 1.4},
                   boxprops={"color": INK2, "lw": 0.8}, whiskerprops={"color": INK2, "lw": 0.8},
                   capprops={"color": INK2, "lw": 0.8})
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(vals)), vals, s=14, color=CLASS_COLORS[cls],
                   edgecolors=SURFACE, linewidths=0.8, zorder=3, alpha=0.9)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axhline(null_floor, color=INK2, lw=0.8, zorder=1)
    ax.text(len(classes) - 0.5, null_floor, "noise floor (1st pct)", ha="right", va="bottom",
            fontsize=7, color=INK2)
    ax.set_xticks(range(len(classes)),
                  [f"{CLASS_NAMES[c]}\nn={int((metrics['label'] == c).sum())}" for c in classes])
    ax.set_ylabel(f"Δ local pLDDT (mutant − WT)")
    ax.set_title("Local confidence change by clinical class")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "fig2_delta_plddt_by_class")


MECHANISM_ORDER = ("core", "ligand", "coactivator", "dimer", "surface")


def delta_by_mechanism(metrics: pd.DataFrame, null_floor: float, out_dir: Path, seed: int) -> list[str]:
    """Δ local pLDDT of pathogenic (orange) and tolerated (blue) variants per site mechanism."""
    m = metrics[metrics["label"].isin(["pathogenic", "benign", "population"])]
    classes = [c for c in MECHANISM_ORDER if (m["mechanism"] == c).any()]
    if not classes:
        return []
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for k, cls in enumerate(classes):
        for offset, labels, color in ((-0.17, ("pathogenic",), CLASS_COLORS["pathogenic"]),
                                      (0.17, ("benign", "population"), CLASS_COLORS["benign"])):
            vals = m.loc[(m["mechanism"] == cls) & m["label"].isin(labels), "delta_local_plddt"].to_numpy()
            if not len(vals):
                continue
            if len(vals) >= 5:
                ax.boxplot(vals, positions=[k + offset], widths=0.28, showfliers=False,
                           medianprops={"color": INK, "lw": 1.4}, boxprops={"color": INK2, "lw": 0.8},
                           whiskerprops={"color": INK2, "lw": 0.8}, capprops={"color": INK2, "lw": 0.8})
            ax.scatter(k + offset + rng.uniform(-0.09, 0.09, len(vals)), vals, s=10, color=color,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3, alpha=0.9)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axhline(null_floor, color=INK2, lw=0.8, zorder=1)
    ax.set_xticks(range(len(classes)), [
        f"{c}\nP {int(((m['mechanism'] == c) & (m['label'] == 'pathogenic')).sum())} · "
        f"T {int(((m['mechanism'] == c) & (m['label'] != 'pathogenic')).sum())}" for c in classes], fontsize=7)
    ax.scatter([], [], s=10, color=CLASS_COLORS["pathogenic"], label="Pathogenic / LP")
    ax.scatter([], [], s=10, color=CLASS_COLORS["benign"], label="Tolerated (benign + gnomAD carriers)")
    ax.legend(loc="lower left", fontsize=6.5)
    ax.set_ylabel("Δ local pLDDT (mutant − WT)")
    ax.set_title("Local confidence change by what the mutated site does")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "fig7_delta_by_mechanism")


def roc_figure(curves: list[dict], out_dir: Path) -> list[str]:
    """``curves``: dicts with key, label, scores (oriented), positives, auc, ci.

    Colours are fixed per metric (not per list position) so a metric keeps its
    colour when another curve is absent.
    """
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
    for c in curves:
        fpr, tpr = roc_curve(c["scores"], c["positives"])
        lo, hi = c["ci"]
        ax.plot(fpr, tpr, color=ROC_COLORS.get(c["key"], SERIES[3]), lw=2,
                label=f"{c['label']}  AUC {c['auc']:.2f} [{lo:.2f}–{hi:.2f}]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate (benign called damaging)")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Pathogenic vs benign discrimination")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.2)
    return _save(fig, out_dir, "fig3_roc")


def position_profile(positions: pd.DataFrame, start: int, end: int, out_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    p = positions.dropna(subset=["min_delta_local_plddt"])
    flagged = p["n_breaking_points"] > 0
    ax.vlines(p.loc[~flagged, "position"], 0, p.loc[~flagged, "min_delta_local_plddt"], color=AXIS, lw=1.6)
    ax.vlines(p.loc[flagged, "position"], 0, p.loc[flagged, "min_delta_local_plddt"], color="#e34948", lw=1.6,
              label="Breaking point (FDR < threshold)")
    path = p[p["n_pathogenic"] > 0]
    top = max(1.0, float(p["min_delta_local_plddt"].max()) + 1.0)
    ax.scatter(path["position"], np.full(len(path), top), marker="v", s=12, color=CLASS_COLORS["pathogenic"],
               edgecolors=SURFACE, linewidths=0.5, label="Has pathogenic variant", zorder=3)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlim(start - 1, end + 1)
    ax.set_xlabel("Residue (UniProt numbering)")
    ax.set_ylabel("Most negative Δ local pLDDT")
    ax.set_title("Structural breaking points along the domain")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    return _save(fig, out_dir, "fig4_position_profile")


def structure_vs_sequence(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    m = metrics.dropna(subset=["llr"])
    if m.empty:
        return []
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for cls in ("vus", "population", "benign", "pathogenic", "cancer"):  # pathogenic and cancer drawn on top
        sub = m[m["label"] == cls]
        if len(sub):
            ax.scatter(sub["llr"], sub["delta_local_plddt"], s=14, color=CLASS_COLORS[cls], edgecolors=SURFACE,
                       linewidths=0.8, alpha=0.9, label=f"{CLASS_NAMES[cls]} (n={len(sub)})")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("ESM-2 log-likelihood ratio (sequence)")
    ax.set_ylabel("Δ local pLDDT (structure)")
    ax.set_title("Sequence vs structural signal")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=7)
    return _save(fig, out_dir, "fig5_structure_vs_sequence")


def saturation_heatmap(metrics: pd.DataFrame, start: int, end: int, out_dir: Path) -> list[str]:
    scan = metrics[metrics["in_scan"]]
    if scan["mt"].nunique() < 19:
        return []
    order = list("DEKRHNQSTGAPCVILMFYW")  # charged → polar → small → hydrophobic → aromatic
    grid = scan.pivot_table(index="mt", columns="position", values="delta_local_plddt").reindex(order)
    grid = grid.reindex(columns=range(start, end + 1))
    vmax = np.nanpercentile(np.abs(grid.to_numpy()), 99) or 1.0
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=DIVERGING, interpolation="nearest",
                   norm=TwoSlopeNorm(0, -vmax, vmax), extent=(start - 0.5, end + 0.5, len(order) - 0.5, -0.5))
    ax.set_yticks(range(len(order)), order, fontsize=6)
    ax.set_xlabel("Residue (UniProt numbering)")
    ax.set_title("In-silico saturation scan: Δ local pLDDT")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Δ local pLDDT", color=INK2)
    cb.outline.set_visible(False)
    return _save(fig, out_dir, "fig6_saturation_heatmap")
