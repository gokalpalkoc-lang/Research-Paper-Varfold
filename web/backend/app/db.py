"""Read-only access to the SQLite bundle produced by ``varfold export``."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SUPPORTED_SCHEMA = 1

SUMMARY_COLUMNS = [
    "variant_id", "position", "wt", "mt", "label", "in_scan", "review_stars", "mean_plddt", "ptm",
    "delta_mean_plddt", "delta_site_plddt", "delta_local_plddt", "rmsd_ca", "lddt_local", "llr",
    "local_q", "breaking_point",
]
DETAIL_COLUMNS = SUMMARY_COLUMNS + [
    "uniprot_curated", "clinvar_ids", "dbsnp_ids", "diseases", "assertions", "max_gnomad_af",
    "delta_ptm", "site_plddt", "min_delta_plddt", "min_delta_position", "n_residues_drop5", "lddt_ca",
    "wt_site_plddt", "wt_site_burial", "blosum62", "local_p", "sequence", "plddt",
]
SORTABLE = {"position", "variant_id", "label", "mean_plddt", "delta_mean_plddt", "delta_site_plddt",
            "delta_local_plddt", "rmsd_ca", "lddt_local", "llr", "local_q"}
LABELS = {"wild_type", "pathogenic", "benign", "vus", "conflicting", "cancer", "population", "unlabeled"}


class Dataset:
    """Thread-safe, read-only view of the bundle; metadata is cached at open."""

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"dataset bundle not found at {path}")
        self.path = path
        self._local = threading.local()
        self.meta = {k: json.loads(v) for k, v in self._conn().execute("SELECT key, value FROM meta")}
        if self.meta.get("schema_version") != SUPPORTED_SCHEMA:
            raise RuntimeError(f"bundle schema {self.meta.get('schema_version')} != {SUPPORTED_SCHEMA}")
        wt = self._conn().execute("SELECT sequence, plddt FROM variants WHERE variant_id = 'WT'").fetchone()
        self.wt_sequence: str = wt["sequence"]
        self.wt_plddt: list[float] = json.loads(wt["plddt"])
        self.etag = f'"{self.meta["exported_at"]}"'

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def list_variants(self, labels: list[str] | None, query: str | None, breaking_only: bool,
                      sort: str, descending: bool, limit: int, offset: int) -> tuple[int, list[dict]]:
        if sort not in SORTABLE:  # interpolated into SQL below, so whitelist strictly
            raise ValueError(f"cannot sort by {sort!r}")
        where, params = ["variant_id != 'WT'"], []
        if labels:
            where.append(f"label IN ({', '.join('?' * len(labels))})")
            params += labels
        if query:
            q = query.strip().upper()
            if q.isascii() and q.isdigit():
                where.append("position = ?")
                params.append(int(q))
            else:
                where.append("variant_id LIKE ? ESCAPE '\\'")
                params.append(q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        if breaking_only:
            where.append("breaking_point = 1")
        clause = " AND ".join(where)
        conn = self._conn()
        total = conn.execute(f"SELECT COUNT(*) FROM variants WHERE {clause}", params).fetchone()[0]
        order = f"{sort} IS NULL, {sort} {'DESC' if descending else 'ASC'}, position, mt"
        rows = conn.execute(
            f"SELECT {', '.join(SUMMARY_COLUMNS)} FROM variants WHERE {clause} ORDER BY {order} LIMIT ? OFFSET ?",
            params + [limit, offset]).fetchall()
        return total, [dict(r) for r in rows]

    def variant(self, variant_id: str) -> dict | None:
        row = self._conn().execute(f"SELECT {', '.join(DETAIL_COLUMNS)} FROM variants WHERE variant_id = ?",
                                   (variant_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["plddt"] = json.loads(out["plddt"])
        if out["position"] is not None:
            out["siblings"] = [dict(r) for r in self._conn().execute(
                "SELECT variant_id, mt, label, delta_local_plddt, breaking_point FROM variants "
                "WHERE position = ? AND variant_id != ? ORDER BY mt", (out["position"], variant_id))]
        return out

    def structure_gz(self, variant_id: str) -> bytes | None:
        row = self._conn().execute("SELECT pdb_gz FROM variants WHERE variant_id = ?", (variant_id,)).fetchone()
        return row[0] if row else None

    def find_sequence(self, sequence: str) -> str | None:
        row = self._conn().execute("SELECT variant_id FROM variants WHERE sequence = ?", (sequence,)).fetchone()
        return row[0] if row else None

    def positions(self) -> list[dict]:
        return [dict(r) for r in self._conn().execute("SELECT * FROM positions ORDER BY position")]
