"""Cross-reference pLDDT/structural change with clinical labels and find breaking points."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import figures
from .config import TargetConfig
from .curate import load_meta
from .metrics import METRIC_BY_KEY, METRICS, compute_metrics
from .stats import (benjamini_hochberg, compare_groups, empirical_lower_p, fisher_enrichment,
                    paired_auc_difference)

log = logging.getLogger(__name__)
PRIMARY = "delta_local_plddt"
BENIGN = ("benign",)
# Secondary negative set: clinical benign plus variants carried by people in gnomAD.
TOLERATED = ("benign", "population")


def _clean(obj):
    """Make numpy/NaN values JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        # Significant figures, not decimals: p-values like 1e-9 must survive.
        # Non-finite values (e.g. an infinite odds ratio) become null: strict JSON has no inf.
        return float(f"{float(obj):.6g}") if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def discrimination(metrics: pd.DataFrame, n_boot: int, seed: int, negatives: tuple[str, ...] = BENIGN) -> list[dict]:
    labeled = metrics[metrics["label"].isin(["pathogenic", *negatives])]
    y = (labeled["label"] == "pathogenic").to_numpy()
    rows = []
    for m in METRICS:
        if labeled[m.key].isna().all():
            continue
        res = compare_groups(labeled[m.key].to_numpy(float), y, m.direction, n_boot, seed)
        rows.append({"metric": m.key, "label": m.label, "kind": m.kind, **res})
    q = benjamini_hochberg([r.get("p", np.nan) for r in rows])
    for r, qv in zip(rows, q):
        r["q_bh"] = qv
    return rows


def versus_baseline(metrics: pd.DataFrame, baseline: str, n_perm: int, seed: int,
                    negatives: tuple[str, ...] = BENIGN) -> list[dict]:
    if metrics[baseline].isna().all():
        return []
    labeled = metrics[metrics["label"].isin(["pathogenic", *negatives])].dropna(subset=[baseline])
    y = (labeled["label"] == "pathogenic").to_numpy()
    if y.sum() < 2 or (~y).sum() < 2:
        return []
    b = labeled[baseline].to_numpy(float) * METRIC_BY_KEY[baseline].direction
    out = []
    for m in METRICS:
        if m.kind != "structural":
            continue
        a = labeled[m.key].to_numpy(float) * m.direction
        out.append({"metric": m.key, "baseline": baseline,
                    **paired_auc_difference(a, b, y, n_perm=n_perm, seed=seed)})
    return out


def correlations(metrics: pd.DataFrame) -> list[dict]:
    out = []
    for a in ("delta_local_plddt", "delta_mean_plddt", "rmsd_ca", "lddt_local"):
        for b in ("llr", "alphamissense", "blosum62", "wt_site_plddt", "wt_site_burial"):
            sub = metrics[[a, b]].dropna()
            if len(sub) < 5:
                continue
            rho, p = sps.spearmanr(sub[a], sub[b])
            out.append({"x": a, "y": b, "n": len(sub), "spearman_rho": rho, "p": p})
    return out


def position_table(metrics: pd.DataFrame, wt_plddt: list[float], burial: list[int], construct_seq: str,
                   start: int) -> pd.DataFrame:
    base = pd.DataFrame({
        "position": np.arange(start, start + len(construct_seq)),
        "wt": list(construct_seq),
        "wt_plddt": wt_plddt,
        "wt_burial": burial,
    })
    if metrics.empty:
        return base
    g = metrics.groupby("position")
    agg = pd.DataFrame({
        "n_variants": g.size(),
        "n_pathogenic": g["label"].apply(lambda s: int((s == "pathogenic").sum())),
        "n_benign": g["label"].apply(lambda s: int((s == "benign").sum())),
        "n_vus": g["label"].apply(lambda s: int((s == "vus").sum())),
        "n_cancer": g["label"].apply(lambda s: int((s == "cancer").sum())),
        "n_population": g["label"].apply(lambda s: int((s == "population").sum())),
        "mean_delta_local_plddt": g["delta_local_plddt"].mean(),
        "min_delta_local_plddt": g["delta_local_plddt"].min(),
        "max_rmsd_ca": g["rmsd_ca"].max(),
        "mean_llr": g["llr"].mean(),
        "n_breaking_points": g["breaking_point"].sum().astype(int),
    }).reset_index()
    out = base.merge(agg, on="position", how="left")
    for c in ("n_variants", "n_pathogenic", "n_benign", "n_vus", "n_cancer", "n_population", "n_breaking_points"):
        out[c] = out[c].fillna(0).astype(int)
    return out


def _confidence_bands(plddt: np.ndarray) -> dict:
    return {"very_high_gt90": float((plddt > 90).mean()), "confident_70_90": float(((plddt > 70) & (plddt <= 90)).mean()),
            "low_50_70": float(((plddt > 50) & (plddt <= 70)).mean()), "very_low_le50": float((plddt <= 50).mean())}


def _low_confidence_segments(plddt: np.ndarray, start: int, cutoff: float = 70.0, min_len: int = 3) -> list[list[int]]:
    segs, run = [], None
    for i, v in enumerate(plddt):
        if v < cutoff and run is None:
            run = i
        elif v >= cutoff and run is not None:
            if i - run >= min_len:
                segs.append([start + run, start + i - 1])
            run = None
    if run is not None and len(plddt) - run >= min_len:
        segs.append([start + run, start + len(plddt) - 1])
    return segs


def run_analysis(cfg: TargetConfig) -> dict:
    paths = cfg.paths
    A = cfg.analysis
    paths.analysis.mkdir(parents=True, exist_ok=True)
    meta = load_meta(cfg)
    metrics, null, info = compute_metrics(cfg)
    if metrics.empty:
        raise RuntimeError("no folded variants to analyse yet")
    if len(null) == 0:
        raise RuntimeError("construct too small to build a distant-residue noise floor")

    metrics["local_p"] = empirical_lower_p(metrics[PRIMARY].to_numpy(), null)
    metrics["local_q"] = benjamini_hochberg(metrics["local_p"].to_numpy())
    metrics["breaking_point"] = metrics["local_q"] < A.breaking_point_fdr
    metrics.sort_values(["position", "mt"]).to_csv(paths.analysis / "metrics.csv", index=False)

    wt_plddt = np.asarray(info["wt_plddt"])
    start, end = cfg.construct.start, cfg.construct.end
    positions = position_table(metrics, info["wt_plddt"], info["wt_burial"], meta["construct"]["sequence"], start)
    positions.to_csv(paths.analysis / "positions.csv", index=False)

    disc = discrimination(metrics, A.bootstrap, A.seed)
    high_conf = metrics[(metrics["review_stars"].fillna(0) >= 1) | metrics["uniprot_curated"]]
    disc_hc = discrimination(high_conf, A.bootstrap, A.seed)
    vs_llr = versus_baseline(metrics, "llr", A.permutations, A.seed)
    vs_am = versus_baseline(metrics, "alphamissense", A.permutations, A.seed)
    vs_site = versus_baseline(metrics, "wt_site_plddt", A.permutations, A.seed)
    vs_blosum = versus_baseline(metrics, "blosum62", A.permutations, A.seed)
    # The same questions against the larger negative set, when population variants exist.
    has_population = bool((metrics["label"] == "population").any())
    disc_tol = discrimination(metrics, A.bootstrap, A.seed, TOLERATED) if has_population else []
    vs_llr_tol = versus_baseline(metrics, "llr", A.permutations, A.seed, TOLERATED) if has_population else []
    vs_am_tol = versus_baseline(metrics, "alphamissense", A.permutations, A.seed, TOLERATED) if has_population else []

    # Variant-level: is a pathogenic variant more often a breaking point than a benign one?
    # (A residue-level test would be confounded by how many variants each residue carries.)
    pb = metrics[metrics["label"].isin(["pathogenic", "benign"])]
    enrichment = fisher_enrichment((pb["label"] == "pathogenic").to_numpy(), pb["breaking_point"].to_numpy(bool))
    # Negative control: cancer-context variants are mostly gain-of-function and should stay folded,
    # so germline pathogenic variants should be breaking points more often than they are.
    pc = metrics[metrics["label"].isin(["pathogenic", "cancer"])]
    control = (fisher_enrichment((pc["label"] == "pathogenic").to_numpy(), pc["breaking_point"].to_numpy(bool))
               if (pc["label"] == "cancer").any() else None)
    bp = metrics[metrics["breaking_point"]]
    by_class = {c: {"n": int((metrics["label"] == c).sum()),
                    "n_breaking_points": int((bp["label"] == c).sum())}
                for c in sorted(metrics["label"].unique())}

    validation = json.loads(paths.validation.read_text()) if paths.validation.exists() else None
    if validation:
        validation = {k: v for k, v in validation.items() if k != "per_residue"}

    null_floor = float(np.percentile(null, 1))
    figs = []
    figs += figures.wt_profile(wt_plddt, start, metrics, meta.get("annotations", []), paths.figures)
    figs += figures.delta_by_class(metrics, null_floor, paths.figures, A.seed)
    labeled = metrics[metrics["label"].isin(["pathogenic", "benign"])]
    y = (labeled["label"] == "pathogenic").to_numpy()
    curves = []
    for key in (PRIMARY, "alphamissense", "llr", "blosum62", "wt_site_plddt"):
        row = next((r for r in disc if r["metric"] == key and r.get("auc") is not None), None)
        if row is None:
            continue
        s = labeled[key].to_numpy(float) * METRIC_BY_KEY[key].direction
        ok = ~np.isnan(s)
        curves.append({"key": key, "label": METRIC_BY_KEY[key].label, "scores": s[ok], "positives": y[ok],
                       "auc": row["auc"], "ci": row["auc_ci95"]})
    if curves:
        figs += figures.roc_figure(curves, paths.figures)
    figs += figures.position_profile(positions, start, end, paths.figures)
    figs += figures.structure_vs_sequence(metrics, paths.figures)
    figs += figures.saturation_heatmap(metrics, start, end, paths.figures)

    result = _clean({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": meta["target"],
        "construct": {"start": start, "end": end, "length": end - start + 1},
        "model": {"fingerprint": info["fingerprint"], "folding": cfg.folding.model,
                  "num_recycles": cfg.folding.num_recycles, "llr": cfg.llr.model,
                  "alphamissense": info["alphamissense"]},
        "dataset": {"counts": metrics["label"].value_counts().to_dict(), "n_folded": len(metrics),
                    "n_missing": info["n_missing"], "n_other_settings": info["n_other_settings"],
                    "n_stale": info["n_stale"]},
        "wild_type": {"mean_plddt": info["wt_mean_plddt"], "ptm": info["wt_ptm"],
                      "confidence_bands": _confidence_bands(wt_plddt),
                      "low_confidence_segments": _low_confidence_segments(wt_plddt, start),
                      "validation": validation},
        "noise_floor": {"definition": "window-mean ΔpLDDT at residues >15 Å (WT Cα) from the mutation",
                        "n": len(null), "p01": null_floor, "p05": float(np.percentile(null, 5)),
                        "median": float(np.median(null))},
        "breaking_points": {"fdr": A.breaking_point_fdr, "local_window": A.local_window,
                            "n_variants": len(bp), "n_positions": int(bp["position"].nunique()),
                            "by_class": by_class, "pathogenic_enrichment": enrichment,
                            "pathogenic_vs_cancer": control,
                            "top": bp.nsmallest(25, PRIMARY)[["variant_id", "label", PRIMARY, "local_q",
                                                              "rmsd_ca", "llr"]].to_dict("records")},
        "discrimination": disc,
        "discrimination_high_confidence_labels": disc_hc,
        "versus_llr": vs_llr,
        "versus_alphamissense": vs_am,
        "versus_blosum62": vs_blosum,
        "versus_wt_site_plddt": vs_site,
        "discrimination_tolerated": disc_tol,
        "versus_llr_tolerated": vs_llr_tol,
        "versus_alphamissense_tolerated": vs_am_tol,
        "correlations": correlations(metrics),
        "figures": figs,
    })
    (paths.analysis / "stats.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (paths.analysis / "report.md").write_text(render_report(result))
    log.info("analysis written to %s", paths.analysis)
    return result


def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def _odds(en: dict) -> str:
    return {"infinite": "∞", "undefined": "undefined (no breaking points)"}.get(en.get("odds_ratio_kind"), _f(en["odds_ratio"], 2))


def _p(v):
    if v is None:
        return "—"
    return f"{v:.1e}" if v < 1e-3 else f"{v:.3f}"


def render_report(r: dict) -> str:
    wt = r["wild_type"]
    lines = [
        f"# {r['target']['title']} — structural confidence analysis",
        "",
        f"_Generated {r['generated_at']} · {r['model']['fingerprint']}_",
        "",
        "## Dataset",
        "",
        "| Class | Folded | Breaking points |",
        "|---|---:|---:|",
    ]
    for cls, v in r["breaking_points"]["by_class"].items():
        lines.append(f"| {cls} | {v['n']} | {v['n_breaking_points']} |")
    ds = r["dataset"]
    if ds["n_missing"] or ds["n_other_settings"] or ds["n_stale"]:
        lines += ["", f"> Incomplete: {ds['n_missing']} not folded, {ds['n_other_settings']} folded under "
                      f"different settings, {ds['n_stale']} stale. Results below are provisional."]
    lines += [
        "", "## Wild-type prediction", "",
        f"- Mean pLDDT **{_f(wt['mean_plddt'], 1)}**, pTM **{_f(wt['ptm'])}**",
        "- Residues by confidence: " + ", ".join(f"{k} {v:.0%}" for k, v in wt["confidence_bands"].items()),
        "- Low-confidence segments (pLDDT < 70): "
        + (", ".join(f"{a}–{b}" for a, b in wt["low_confidence_segments"]) or "none"),
    ]
    if wt["validation"]:
        v = wt["validation"]
        lines.append(f"- Versus experimental {v['pdb_id']} chain {v['chain']} ({v['n_aligned']} residues, "
                     f"{v['coverage']:.0%} coverage): Cα RMSD **{v['ca_rmsd']:.2f} Å**, lDDT-Cα "
                     f"**{v['lddt_ca']:.3f}**; pLDDT–lDDT Spearman ρ = {v['plddt_vs_lddt_spearman']['rho']:.2f}")
    nf = r["noise_floor"]
    bp = r["breaking_points"]
    en = bp["pathogenic_enrichment"]
    lines += [
        "", "## Breaking points", "",
        f"Noise floor ({nf['definition']}; n = {nf['n']:,}): 1st percentile {_f(nf['p01'], 2)}, "
        f"5th percentile {_f(nf['p05'], 2)}.",
        f"A variant is a breaking point when its Δ local pLDDT (±{bp['local_window']} residues) falls "
        f"below that floor at BH-FDR < {bp['fdr']}.",
        "",
        f"- **{bp['n_variants']}** breaking-point variants at **{bp['n_positions']}** positions.",
        f"- Breaking-point rate, pathogenic vs benign variants: odds ratio "
        f"{_odds(en)}, one-sided Fisher p = {_p(en['p_one_sided'])} "
        f"(table [[P breaking, P not], [B breaking, B not]] = {en['table']}).",
    ]
    if ctl := bp.get("pathogenic_vs_cancer"):
        lines.append(f"- Negative control, germline pathogenic vs cancer-context variants (mostly gain-of-function, "
                     f"expected to stay folded): odds ratio {_odds(ctl)}, one-sided Fisher p = {_p(ctl['p_one_sided'])} "
                     f"(table [[P breaking, P not], [cancer breaking, cancer not]] = {ctl['table']}).")
    lines += [
        "", "| Variant | Class | Δ local pLDDT | q | Cα RMSD (Å) | ESM-2 LLR |", "|---|---|---:|---:|---:|---:|",
    ]
    for t in bp["top"][:15]:
        lines.append(f"| {t['variant_id']} | {t['label']} | {_f(t['delta_local_plddt'], 2)} | {_p(t['local_q'])} | "
                     f"{_f(t['rmsd_ca'], 2)} | {_f(t['llr'], 2)} |")

    def comparison_table(rows):
        out = ["| Metric | ΔAUC [95% CI] | p (permutation) | p (DeLong) |", "|---|---|---:|---:|"]
        for c in rows:
            if c["difference"] is None:
                continue
            out.append(f"| {c['metric']} | {c['difference']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
                       f"| {_p(c['p'])} | {_p(c['p_delong'])} |")
        return out

    def disc_table(rows):
        out = ["| Metric | Kind | n (P/B) | AUC [95% CI] | Cliff's δ | p | q (BH) |",
               "|---|---|---|---|---:|---:|---:|"]
        for d in rows:
            if d.get("auc") is None:
                continue
            lo, hi = d["auc_ci95"]
            out.append(f"| {d['label']} | {d['kind']} | {d['n_pos']}/{d['n_neg']} | {d['auc']:.3f} "
                       f"[{lo:.3f}–{hi:.3f}] | {d['cliffs_delta']:.3f} | {_p(d['p'])} | {_p(d['q_bh'])} |")
        return out

    lines += ["", "## Pathogenic vs benign", "",
              "AUC > 0.5 means the metric ranks pathogenic variants as more damaging. "
              "CIs: stratified bootstrap. Positional rows are site priors that ignore the substituted residue.", ""]
    lines += disc_table(r["discrimination"])
    lines += ["", "High-confidence labels only (ClinVar ≥1★ or UniProt-curated):", ""]
    lines += disc_table(r["discrimination_high_confidence_labels"])
    for key, title in (("versus_alphamissense", "AlphaMissense"), ("versus_llr", "ESM-2 LLR"),
                       ("versus_blosum62", "BLOSUM62"), ("versus_wt_site_plddt", "WT site pLDDT")):
        if r[key]:
            lines += ["", f"### Paired comparison against {title}", "",
                      "ΔAUC with DeLong 95% CI; p from a paired permutation test (DeLong p for reference).", ""]
            lines += comparison_table(r[key])
    if r["discrimination_tolerated"]:
        lines += ["", "## Pathogenic vs tolerated (benign + population)", "",
                  "Secondary analysis with a larger negative set: clinical benign variants plus clinically "
                  "unclassified variants carried by people in gnomAD (see `population` in the config). Carriers can "
                  "have mild disease, so this set is noisier than the benign one.", ""]
        lines += disc_table(r["discrimination_tolerated"])
        for key, title in (("versus_alphamissense_tolerated", "AlphaMissense"), ("versus_llr_tolerated", "ESM-2 LLR")):
            if r[key]:
                lines += ["", f"### Paired comparison against {title} (tolerated negatives)", ""]
                lines += comparison_table(r[key])
    lines += ["", "## Correlations (Spearman, all folded variants)", "", "| x | y | n | ρ | p |", "|---|---|---:|---:|---:|"]
    for c in r["correlations"]:
        lines.append(f"| {c['x']} | {c['y']} | {c['n']} | {_f(c['spearman_rho'])} | {_p(c['p'])} |")
    lines += ["", "## Figures", ""] + [f"- `figures/{f}`" for f in r["figures"] if f.endswith(".pdf")]
    return "\n".join(lines) + "\n"
