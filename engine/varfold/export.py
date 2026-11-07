"""Package results into a single read-only SQLite file for the showcase web app."""

from __future__ import annotations

import gzip
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import TargetConfig
from .curate import load_dataset, load_meta
from .folding import read_predictions, structure_file
from .structure import read_ca, superpose_pdb

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1

VARIANT_COLUMNS = {
    "variant_id": "TEXT PRIMARY KEY", "position": "INTEGER", "wt": "TEXT", "mt": "TEXT",
    "label": "TEXT NOT NULL", "in_scan": "INTEGER", "review_stars": "INTEGER", "uniprot_curated": "INTEGER",
    "clinvar_ids": "TEXT", "dbsnp_ids": "TEXT", "diseases": "TEXT", "assertions": "TEXT",
    "max_gnomad_af": "REAL", "mean_plddt": "REAL", "ptm": "REAL", "delta_mean_plddt": "REAL",
    "delta_ptm": "REAL", "site_plddt": "REAL", "delta_site_plddt": "REAL", "delta_local_plddt": "REAL",
    "min_delta_plddt": "REAL", "min_delta_position": "INTEGER", "n_residues_drop5": "INTEGER",
    "rmsd_ca": "REAL", "lddt_ca": "REAL", "lddt_local": "REAL", "llr": "REAL", "alphamissense": "REAL",
    "blosum62": "INTEGER", "wt_site_plddt": "REAL", "gnomad_carriers": "INTEGER",
    "wt_site_burial": "INTEGER", "local_p": "REAL", "local_q": "REAL", "breaking_point": "INTEGER",
    "sequence": "TEXT NOT NULL", "plddt": "TEXT NOT NULL", "pdb_gz": "BLOB NOT NULL",
}
POSITION_COLUMNS = {
    "position": "INTEGER PRIMARY KEY", "wt": "TEXT", "wt_plddt": "REAL", "wt_burial": "INTEGER",
    "n_variants": "INTEGER", "n_pathogenic": "INTEGER", "n_benign": "INTEGER", "n_vus": "INTEGER",
    "n_cancer": "INTEGER", "n_population": "INTEGER",
    "mean_delta_local_plddt": "REAL", "min_delta_local_plddt": "REAL", "max_rmsd_ca": "REAL",
    "mean_llr": "REAL", "n_breaking_points": "INTEGER",
}


def _value(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.bool_, bool)):
        return int(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return v


def export_web_db(cfg: TargetConfig) -> str:
    paths = cfg.paths
    stats_path = paths.analysis / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError("analysis missing — run `varfold analyze` first")
    stats = json.loads(stats_path.read_text())
    meta = load_meta(cfg)
    dataset = load_dataset(cfg).set_index("variant_id")
    metrics = pd.read_csv(paths.analysis / "metrics.csv").set_index("variant_id")
    positions = pd.read_csv(paths.analysis / "positions.csv")
    preds = read_predictions(paths.predictions, stats["model"]["fingerprint"])
    validation = json.loads(paths.validation.read_text()) if paths.validation.exists() else None

    wt_pdb = structure_file(paths, preds["WT"]).read_text()
    wt_ca = read_ca(wt_pdb).coords

    out = paths.web_db
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute(f"CREATE TABLE variants ({', '.join(f'{k} {t}' for k, t in VARIANT_COLUMNS.items())})")
        con.execute(f"CREATE TABLE positions ({', '.join(f'{k} {t}' for k, t in POSITION_COLUMNS.items())})")

        ids = ["WT"] + [v for v in metrics.index if v in preds]
        rows = []
        for vid in ids:
            d = dataset.loc[vid]
            m = metrics.loc[vid] if vid in metrics.index else None
            pdb = wt_pdb if vid == "WT" else superpose_pdb(structure_file(paths, preds[vid]).read_text(), wt_ca)
            row = {}
            for col in VARIANT_COLUMNS:
                if col == "variant_id":
                    row[col] = vid
                elif col == "plddt":
                    row[col] = json.dumps(preds[vid]["plddt"])
                elif col == "pdb_gz":
                    row[col] = gzip.compress(pdb.encode(), mtime=0)
                elif m is not None and col in m.index:
                    row[col] = _value(m[col])
                elif col in d.index:
                    row[col] = _value(d[col])
                elif col in ("mean_plddt", "ptm"):
                    row[col] = preds[vid][col]
                else:
                    row[col] = None
            rows.append(row)
        cols = list(VARIANT_COLUMNS)
        con.executemany(f"INSERT INTO variants ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [tuple(r[c] for c in cols) for r in rows])
        pcols = [c for c in POSITION_COLUMNS if c in positions.columns]
        con.executemany(f"INSERT INTO positions ({', '.join(pcols)}) VALUES ({', '.join('?' * len(pcols))})",
                        [tuple(_value(v) for v in rec) for rec in positions[pcols].itertuples(index=False)])
        con.execute("CREATE INDEX idx_variants_position ON variants(position)")
        con.execute("CREATE INDEX idx_variants_label ON variants(label)")
        con.execute("CREATE UNIQUE INDEX idx_variants_sequence ON variants(sequence)")

        summary = {k: stats[k] for k in ("dataset", "wild_type", "noise_floor", "breaking_points",
                                         "discrimination", "discrimination_high_confidence_labels",
                                         "versus_llr", "versus_alphamissense", "versus_blosum62", "versus_wt_site_plddt",
                                         "discrimination_tolerated", "correlations", "generated_at")
                   if k in stats}
        wt_rec = preds["WT"]
        meta_rows = {
            "schema_version": SCHEMA_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "target": stats["target"],
            "construct": {**stats["construct"], "sequence": meta["construct"]["sequence"]},
            "model": {**stats["model"], "device": wt_rec.get("device"), "precision": wt_rec.get("precision"),
                      "versions": wt_rec.get("versions")},
            "provenance": {"retrieved_at": meta["retrieved_at"], "sources": meta["sources"],
                           "uniprot_entry_version": meta.get("uniprot_entry_version"),
                           "filters": meta["filters"], "dropped": meta["dropped"]},
            "annotations": meta.get("annotations", []),
            "stats": summary,
            "validation": validation,
        }
        con.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                        [(k, json.dumps(v, allow_nan=False)) for k, v in meta_rows.items()])
        con.commit()
        con.execute("VACUUM")
    finally:
        con.close()
    os.replace(tmp, out)
    log.info("exported %d structures to %s (%.1f MB)", len(rows), out, out.stat().st_size / 1e6)
    return str(out)
