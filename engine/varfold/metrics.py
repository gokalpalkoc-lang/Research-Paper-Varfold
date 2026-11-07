"""Per-variant structural metrics relative to the wild-type prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d

from .alphamissense import load_scores
from .config import TargetConfig
from .curate import load_dataset, load_meta
from .folding import read_predictions, structure_file
from .sequences import sequence_hash
from .structure import lddt_per_residue, neighbour_count, read_ca, rmsd

log = logging.getLogger(__name__)

DISTANT_CUTOFF = 15.0  # Å from the mutated residue (WT Cα) — beyond lDDT's inclusion radius

# BLOSUM62 (Henikoff & Henikoff, 1992): the trivial substitution-chemistry baseline.
_B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_B62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
BLOSUM62 = {(a, b): int(v) for a, row in zip(_B62_ORDER, _B62_ROWS.split("\n")[1:])
            for b, v in zip(_B62_ORDER, row.split())}


@dataclass
class Metric:
    key: str
    label: str
    # +1: larger value = more damaging; -1: smaller value = more damaging.
    direction: int
    kind: str  # "structural" (mutation-specific), "baseline" (sequence), "positional" (site prior)


METRICS = [
    Metric("delta_local_plddt", "Δ local pLDDT", -1, "structural"),
    Metric("delta_site_plddt", "Δ site pLDDT", -1, "structural"),
    Metric("delta_mean_plddt", "Δ mean pLDDT", -1, "structural"),
    Metric("delta_ptm", "Δ pTM", -1, "structural"),
    Metric("lddt_local", "local lDDT-Cα vs WT", -1, "structural"),
    Metric("rmsd_ca", "Cα RMSD vs WT (Å)", +1, "structural"),
    Metric("llr", "ESM-2 LLR", -1, "baseline"),
    Metric("blosum62", "BLOSUM62 score", -1, "baseline"),
    Metric("alphamissense", "AlphaMissense", +1, "baseline"),
    Metric("wt_site_plddt", "WT site pLDDT", +1, "positional"),
    Metric("wt_site_burial", "WT site burial (Cα within 10 Å)", +1, "positional"),
]
METRIC_BY_KEY = {m.key: m for m in METRICS}


@dataclass
class WildType:
    plddt: np.ndarray
    ca: np.ndarray
    mean_plddt: float
    ptm: float
    fingerprint: str
    distances: np.ndarray
    burial: np.ndarray


def window_mean(values: np.ndarray, w: int) -> np.ndarray:
    """Mean over [i-w, i+w] clipped to the chain ends, for every i."""
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(len(values))
    lo, hi = np.maximum(0, idx - w), np.minimum(len(values), idx + w + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def load_wild_type(cfg: TargetConfig, predictions: dict[str, dict]) -> WildType:
    rec = predictions.get("WT")
    if rec is None:
        raise RuntimeError("no WT prediction — run `varfold fold` first")
    ca = read_ca(structure_file(cfg.paths, rec).read_text()).coords
    plddt = np.asarray(rec["plddt"], dtype=float)
    if len(ca) != len(plddt):
        raise RuntimeError("WT structure and pLDDT lengths differ")
    dist = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    return WildType(plddt, ca, rec["mean_plddt"], rec["ptm"], rec["fingerprint"], dist, neighbour_count(ca))


def compute_metrics(cfg: TargetConfig) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return (per-variant metrics, null distribution of local ΔpLDDT, provenance info).

    The null is built from the *same statistic* evaluated at windows whose residues
    are all > 15 Å from each mutation: how much local pLDDT moves where the
    substitution should have no direct effect. It is the noise floor that
    "breaking point" calls are tested against.
    """
    paths = cfg.paths
    load_meta(cfg)  # refuses to run if the config's construct no longer matches the dataset
    df = load_dataset(cfg)
    latest = read_predictions(paths.predictions)
    if "WT" not in latest:
        raise RuntimeError("no WT prediction — run `varfold fold` first")
    fingerprint = latest["WT"]["fingerprint"]
    preds = read_predictions(paths.predictions, fingerprint)
    wt = load_wild_type(cfg, preds)
    w = cfg.analysis.local_window
    start = cfg.construct.start

    rows, null_parts = [], []
    missing, stale, other_fp = [], [], []
    for r in df[df["variant_id"] != "WT"].itertuples(index=False):
        rec = preds.get(r.variant_id)
        if rec is None:
            (other_fp if r.variant_id in latest else missing).append(r.variant_id)
            continue
        if rec["sequence_hash"] != sequence_hash(r.sequence):
            stale.append(r.variant_id)
            continue
        i = int(r.construct_index) - 1
        plddt = np.asarray(rec["plddt"], dtype=float)
        delta = plddt - wt.plddt
        local = window_mean(delta, w)
        ca = read_ca(structure_file(paths, rec).read_text()).coords
        per_res_lddt = lddt_per_residue(ca, wt.ca)
        lo, hi = max(0, i - w), min(len(delta), i + w + 1)

        # A null window counts only if *every* residue in it is >15 Å from the mutation;
        # testing just the centre would let the substitution's own effect into the null.
        distant = minimum_filter1d(wt.distances[i], size=2 * w + 1, mode="nearest") > DISTANT_CUTOFF
        null_parts.append(local[distant])

        rows.append({
            "variant_id": r.variant_id, "position": int(r.position), "wt": r.wt, "mt": r.mt,
            "label": r.label, "in_scan": bool(r.in_scan),
            "review_stars": r.review_stars, "uniprot_curated": bool(r.uniprot_curated),
            # Datasets curated before population counts existed have no such column.
            "gnomad_carriers": getattr(r, "gnomad_carriers", pd.NA),
            "mean_plddt": rec["mean_plddt"], "ptm": rec["ptm"],
            "delta_mean_plddt": rec["mean_plddt"] - wt.mean_plddt,
            "delta_ptm": rec["ptm"] - wt.ptm,
            "site_plddt": plddt[i], "delta_site_plddt": delta[i],
            "delta_local_plddt": local[i],
            "min_delta_plddt": float(delta.min()),
            "min_delta_position": start + int(delta.argmin()),
            "n_residues_drop5": int((delta < -5).sum()),
            "rmsd_ca": rmsd(ca, wt.ca),
            "lddt_ca": float(np.nanmean(per_res_lddt)),
            "lddt_local": float(np.nanmean(per_res_lddt[lo:hi])),
            "wt_site_plddt": wt.plddt[i], "wt_site_burial": int(wt.burial[i]),
            "blosum62": BLOSUM62[(r.wt, r.mt)],
        })

    if missing:
        log.warning("%d variants have no prediction yet (e.g. %s)", len(missing), missing[:5])
    if other_fp:
        log.warning("%d variants were folded under different settings than WT (%s) and are excluded; "
                    "re-run `varfold fold` to refold them", len(other_fp), fingerprint)
    if stale:
        log.warning("%d predictions are for an outdated sequence and are excluded", len(stale))

    out = pd.DataFrame(rows)
    if paths.llr.exists() and len(out):
        llr = pd.read_csv(paths.llr)[["variant_id", "llr"]]
        out = out.merge(llr, on="variant_id", how="left")
    else:
        out["llr"] = np.nan
    am, am_source = load_scores(cfg)
    out = out.merge(am, on="variant_id", how="left") if am is not None and len(out) else out.assign(alphamissense=np.nan)
    null = np.concatenate(null_parts) if null_parts else np.array([])
    info = {"fingerprint": fingerprint, "n_missing": len(missing), "n_other_settings": len(other_fp),
            "n_stale": len(stale), "alphamissense": am_source, "wt_mean_plddt": wt.mean_plddt, "wt_ptm": wt.ptm,
            "wt_plddt": wt.plddt.tolist(), "wt_burial": wt.burial.tolist()}
    return out, null, info
