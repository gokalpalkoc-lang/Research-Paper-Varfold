"""Two robustness checks the paper needs before it can claim structural confidence "works".

1. **Edge effect.** The folded construct is a cut-out fragment, and the ends of any fragment
   are floppy. If the pathogenic-vs-tolerated signal came from the ends, it would vanish
   once variants near both ends are dropped. ``edge_trim`` repeats the comparison while
   trimming 0, 10, 20 and 30 residues from each end.
2. **Burial.** How buried a position is needs no folding at all (it comes from the wild-type
   structure). ``feature_sets`` asks whether the mutant-versus-wild-type structural change
   adds anything once burial is known, and whether burial adds anything to AlphaMissense.
   Scores are compared as cross-validated AUCs of a small L2-regularised logistic model;
   gains are tested on the pooled out-of-fold predictions (paired, DeLong CI + permutation p).

Burial here is the relative solvent accessibility of the wild-type site (``rsasa``, smaller =
more buried, so it is oriented with direction -1).

Status: these checks were chosen after a first look at the AR data, so they are exploratory
for that target. The design (which comparisons, which feature sets) is fixed here in code,
so a new target is analysed with exactly the same choices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit

from .metrics import METRIC_BY_KEY
from .stats import auc, bootstrap_auc, fisher_enrichment, paired_auc_difference

BENIGN = ("benign",)
TOLERATED = ("benign", "population")
TRIMS = (0, 10, 20, 30)
EDGE_WIDTH = 25
BURIAL = "rsasa"
# Metrics shown in the edge-trim table (burial is included as the no-folding reference).
EDGE_KEYS = ("delta_local_plddt", "delta_mean_plddt", "rmsd_ca", BURIAL, "alphamissense")
STRUCTURE = ("delta_local_plddt", "delta_mean_plddt", "rmsd_ca")
# name -> metric keys. The first two entries drive the headline comparison.
FEATURE_SETS = {
    "burial": (BURIAL,),
    "burial + structure": (BURIAL, *STRUCTURE),
    "structure (Δ local pLDDT)": ("delta_local_plddt",),
    "structure (Δ mean pLDDT)": ("delta_mean_plddt",),
    "structure (Cα RMSD)": ("rmsd_ca",),
    "structure (all three)": STRUCTURE,
    "AlphaMissense": ("alphamissense",),
    "AlphaMissense + burial": ("alphamissense", BURIAL),
    "AlphaMissense + burial + structure": ("alphamissense", BURIAL, *STRUCTURE),
}


def direction(key: str) -> int:
    """+1 when a larger value means more damaging (burial: smaller rSASA = more buried)."""
    return -1 if key == BURIAL else METRIC_BY_KEY[key].direction


def oriented(df: pd.DataFrame, key: str) -> np.ndarray:
    return df[key].to_numpy(float) * direction(key)


# ---------------------------------------------------------------- edge effect

def edge_trim(metrics: pd.DataFrame, start: int, end: int, negatives: tuple[str, ...],
              n_boot: int, seed: int, trims: tuple[int, ...] = TRIMS) -> list[dict]:
    """Pathogenic vs ``negatives`` after dropping variants within ``trim`` residues of either end."""
    rows = []
    for t in trims:
        sub = metrics[(metrics["position"] >= start + t) & (metrics["position"] <= end - t)]
        lab = sub[sub["label"].isin(["pathogenic", *negatives])]
        y = (lab["label"] == "pathogenic").to_numpy()
        row = {"trim": t, "n_variants": len(sub), "n_pathogenic": int(y.sum()), "n_negative": int((~y).sum()),
               "auc": {}, "breaking_point_enrichment": None,
               "pathogenic_breaking_rate": None, "negative_breaking_rate": None}
        for key in EDGE_KEYS:
            if key not in lab or lab[key].isna().all():
                continue
            ok = lab[key].notna().to_numpy()
            s, yy = oriented(lab, key)[ok], y[ok]
            if yy.sum() < 2 or (~yy).sum() < 2:
                row["auc"][key] = None
                continue
            lo, hi = bootstrap_auc(s, yy, n_boot, seed)
            row["auc"][key] = {"auc": auc(s, yy), "ci95": [lo, hi]}
        if y.sum() and (~y).sum():
            bp = lab["breaking_point"].to_numpy(bool)
            row["breaking_point_enrichment"] = fisher_enrichment(y, bp)
            row["pathogenic_breaking_rate"] = float(bp[y].mean())
            row["negative_breaking_rate"] = float(bp[~y].mean())
        rows.append(row)
    return rows


def edge_share(metrics: pd.DataFrame, start: int, end: int, width: int = EDGE_WIDTH) -> dict:
    """How many breaking points sit in the outer ``width`` residues, and how that differs by class."""
    edge = ((metrics["position"] < start + width) | (metrics["position"] > end - width)).to_numpy()
    bp = metrics["breaking_point"].to_numpy(bool)
    by_label = {}
    metrics = metrics.reset_index(drop=True)  # row labels double as positions into ``edge``/``bp``
    for label, g in metrics.groupby("label"):
        e, b = edge[g.index.to_numpy()], bp[g.index.to_numpy()]
        by_label[label] = {"n_edge": int(e.sum()), "edge_breaking_rate": float(b[e].mean()) if e.any() else None,
                           "n_middle": int((~e).sum()),
                           "middle_breaking_rate": float(b[~e].mean()) if (~e).any() else None}
    return {"width": width, "variants": len(metrics), "variants_in_edges": int(edge.sum()),
            "breaking_points": int(bp.sum()), "breaking_points_in_edges": int((bp & edge).sum()),
            "by_label": by_label}


# ----------------------------------------------------------------- burial

def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    X1 = np.c_[np.ones(len(X)), X]

    def loss(w):
        z = X1 @ w
        grad = X1.T @ (expit(z) - y)
        grad[1:] += 2 * l2 * w[1:]
        return float(np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[1:] ** 2)), grad

    return optimize.minimize(loss, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B").x


def cross_validated(X: np.ndarray, y: np.ndarray, reps: int, folds: int, seed: int,
                    l2: float = 1.0) -> tuple[float, float, np.ndarray]:
    """Repeated stratified k-fold. Returns (mean AUC, sd over repeats, mean out-of-fold score).

    Features are standardised inside each training fold, so nothing about the held-out
    variants leaks into the fit.
    """
    y = np.asarray(y, bool)
    rng = np.random.default_rng(seed)
    groups = (np.where(y)[0], np.where(~y)[0])
    oof = np.zeros((reps, len(y)))
    aucs = []
    for r in range(reps):
        fold_of = np.empty(len(y), int)
        for idx in groups:
            fold_of[rng.permutation(idx)] = np.arange(len(idx)) % folds
        for k in range(folds):
            te = fold_of == k
            tr = ~te
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            w = _fit_logistic((X[tr] - mu) / sd, y[tr].astype(float), l2)
            oof[r, te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ w
        aucs.append(auc(oof[r], y))
    return float(np.mean(aucs)), float(np.std(aucs)), oof.mean(0)


def feature_sets(metrics: pd.DataFrame, negatives: tuple[str, ...], reps: int, seed: int,
                 n_perm: int, folds: int = 5) -> dict | None:
    """Cross-validated AUC of each feature set on the complete cases, plus paired gains."""
    lab = metrics[metrics["label"].isin(["pathogenic", *negatives])]
    avail = {name: keys for name, keys in FEATURE_SETS.items()
             if all(k in lab and lab[k].notna().any() for k in keys)}
    if "burial" not in avail:
        return None
    need = sorted({k for keys in avail.values() for k in keys})
    lab = lab.dropna(subset=need)
    y = (lab["label"] == "pathogenic").to_numpy()
    if min(y.sum(), (~y).sum()) < 2 * folds:
        return None
    results, oof = [], {}
    for name, keys in avail.items():
        X = np.column_stack([oriented(lab, k) for k in keys])
        mean, sd, oof[name] = cross_validated(X, y, reps, folds, seed)
        results.append({"name": name, "features": list(keys), "cv_auc": mean, "cv_auc_sd": sd})

    def gain(new: str, base: str) -> dict | None:
        if new not in oof or base not in oof:
            return None
        return {"new": new, "base": base, **paired_auc_difference(oof[new], oof[base], y, n_perm=n_perm, seed=seed)}

    gains = [g for g in (gain("burial + structure", "burial"),
                         gain("AlphaMissense + burial", "AlphaMissense"),
                         gain("AlphaMissense + burial + structure", "AlphaMissense + burial"),
                         gain("burial", "structure (Δ mean pLDDT)")) if g]
    return {"negatives": list(negatives), "n_pathogenic": int(y.sum()), "n_negative": int((~y).sum()),
            "folds": folds, "repeats": reps, "sets": results, "gains": gains,
            "strata": burial_strata(lab, y)}


def burial_strata(lab: pd.DataFrame, y: np.ndarray) -> dict:
    """Is the structural change useful once burial is held fixed? Split at the median rSASA."""
    cut = float(lab[BURIAL].median())
    out = {"rsasa_cut": cut, "groups": []}
    for name, mask in (("buried (rSASA ≤ median)", (lab[BURIAL] <= cut).to_numpy()),
                       ("exposed (rSASA > median)", (lab[BURIAL] > cut).to_numpy())):
        yy = y[mask]
        row = {"name": name, "n_pathogenic": int(yy.sum()), "n_negative": int((~yy).sum()), "auc": {}}
        if yy.sum() >= 2 and (~yy).sum() >= 2:
            for key in (*STRUCTURE, "alphamissense"):
                if key in lab and lab[key].notna().all():
                    row["auc"][key] = auc(oriented(lab[mask], key), yy)
        out["groups"].append(row)
    return out


def run(metrics: pd.DataFrame, start: int, end: int, *, n_boot: int, n_perm: int, seed: int,
        reps: int = 30) -> dict | None:
    """All robustness checks, or None when burial (``rsasa``) is not available for this target."""
    if BURIAL not in metrics or metrics[BURIAL].isna().all():
        return None
    has_population = bool((metrics["label"] == "population").any())
    negatives = TOLERATED if has_population else BENIGN
    return {
        "burial_definition": "relative solvent accessibility of the wild-type site (smaller = more buried)",
        "edge_trim_benign": edge_trim(metrics, start, end, BENIGN, n_boot, seed),
        "edge_trim_tolerated": edge_trim(metrics, start, end, TOLERATED, n_boot, seed) if has_population else [],
        "edge_share": edge_share(metrics, start, end),
        "feature_sets": feature_sets(metrics, negatives, reps, seed, n_perm),
    }
