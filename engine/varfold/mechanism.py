"""Where each residue of the construct sits, from experimental structures.

A substitution can cause disease by unfolding the domain or by breaking one of its
interactions while the fold stays intact. ESMFold sees a single chain with no ligand
and no partner, so it can only notice the first kind. This module tags every position
with the mechanism it most plausibly serves, so analysis can ask where structural
confidence tracks pathogenicity and where it is blind by construction:

  ligand       heavy atom within ``contact_cutoff`` Å of the bound hormone
  coactivator  within the cutoff of a coactivator / N-terminal peptide in the AF-2 groove
  dimer        within the cutoff of the other domain of the LBD homodimer
  core         relative solvent accessibility below ``buried_rsasa`` (protein chain alone)
  surface      everything else that is resolved
  unresolved   not modelled in the burial structure and in no contact set

A residue in several sets takes the first class in that order. Residue numbers go
through SIFTS and every mapped residue's amino acid is checked against UniProt, as
in validation; structures and mappings are snapshotted in ``raw/`` with their SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import TargetConfig
from .curate import fetch_json, fetch_text, load_meta
from .validate import PDB_URL, SIFTS_URL, THREE_TO_ONE, author_to_uniprot

log = logging.getLogger(__name__)

CLASSES = ("ligand", "coactivator", "dimer", "core", "surface", "unresolved")
INTERACTION_CLASSES = ("ligand", "coactivator", "dimer")
ASSEMBLY_URL = "https://files.rcsb.org/download/{pdb}.pdb{n}"
PROBE = 1.4
RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90}
# Theoretical maximum accessible surface area per residue (Tien et al., PLoS One 2013), Å².
MAX_ASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
           "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174}


@dataclass
class Atoms:
    chain: np.ndarray    # (N,) chain id
    resnum: np.ndarray   # (N,) author residue number
    resname: np.ndarray  # (N,)
    element: np.ndarray  # (N,)
    hetero: np.ndarray   # (N,) bool: HETATM record
    xyz: np.ndarray      # (N, 3)

    def select(self, mask: np.ndarray) -> "Atoms":
        return Atoms(*(getattr(self, f)[mask] for f in ("chain", "resnum", "resname", "element", "hetero", "xyz")))


def read_atoms(pdb_text: str, model: int | None = None) -> Atoms:
    """Heavy atoms of one model (first altloc, no insertion codes, no water).

    ``model=None`` takes the first model. Biological-assembly files (``.pdb1``) put each
    symmetry copy in its own MODEL, with the same chain ids, so ask for it by number.
    """
    rows = []
    current = None
    for line in pdb_text.splitlines():
        if line.startswith("MODEL"):
            current = int(line[10:14])
            continue
        if line.startswith("ENDMDL"):
            if model is None or current == model:
                break
            continue
        if model is not None and current != model:
            continue
        if not line.startswith(("ATOM", "HETATM")) or line[16] not in " A" or line[26].strip():
            continue
        resname = line[17:20].strip()
        element = (line[76:78].strip() if len(line) >= 78 else "") or line[12:16].strip()[0]
        if element.upper() in ("H", "D") or resname in ("HOH", "WAT", "DOD"):
            continue
        rows.append((line[21], int(line[22:26]), resname, element.upper(), line.startswith("HETATM"),
                     float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not rows:
        raise ValueError("no atoms found")
    cols = list(zip(*rows))
    return Atoms(np.array(cols[0]), np.array(cols[1]), np.array(cols[2]), np.array(cols[3]), np.array(cols[4]),
                 np.array(cols[5:8], dtype=float).T)


def protein(atoms: Atoms, chain: str) -> Atoms:
    return atoms.select((atoms.chain == chain) & ~atoms.hetero)


def contact_residues(site: Atoms, partner: Atoms, cutoff: float) -> set[int]:
    """Author residue numbers of ``site`` with any atom within ``cutoff`` Å of ``partner``."""
    if not len(partner.xyz):
        raise ValueError("partner has no atoms")
    dist, _ = cKDTree(partner.xyz).query(site.xyz, distance_upper_bound=cutoff)
    return {int(n) for n in site.resnum[np.isfinite(dist)]}


def _sphere(n: int) -> np.ndarray:
    """``n`` near-uniform unit vectors (golden-section spiral)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def atom_sasa(atoms: Atoms, n_points: int = 200) -> np.ndarray:
    """Solvent-accessible surface area per atom (Shrake–Rupley), Å²."""
    radii = np.array([RADII.get(e, 1.80) for e in atoms.element]) + PROBE
    tree = cKDTree(atoms.xyz)
    dots = _sphere(n_points)
    area = np.empty(len(radii))
    for i, (x, r) in enumerate(zip(atoms.xyz, radii)):
        near = [j for j in tree.query_ball_point(x, r + radii.max()) if j != i]
        points = x + r * dots
        exposed = np.ones(n_points, bool)
        for j in near:
            exposed &= np.sum((points - atoms.xyz[j]) ** 2, axis=1) >= radii[j] ** 2
        area[i] = 4 * np.pi * r ** 2 * exposed.mean()
    return area


def relative_sasa(chain_atoms: Atoms) -> dict[int, float]:
    """Residue SASA / theoretical maximum, per author residue number."""
    area = atom_sasa(chain_atoms)
    out = {}
    for n in np.unique(chain_atoms.resnum):
        sel = chain_atoms.resnum == n
        aa = THREE_TO_ONE.get(str(chain_atoms.resname[sel][0]))
        if aa:
            out[int(n)] = float(area[sel].sum() / MAX_ASA[aa])
    return out


class _Structure:
    """One PDB entry: its atoms and a checked author→UniProt mapping for one chain."""

    def __init__(self, cfg: TargetConfig, pdb_id: str, chain: str, refresh: bool, provenance: dict):
        paths = cfg.paths
        pdb_id = pdb_id.upper()
        pdb_path, sifts_path = paths.raw / f"{pdb_id}.pdb", paths.raw / f"sifts_{pdb_id}.json"
        self.atoms = read_atoms(fetch_text(PDB_URL.format(pdb=pdb_id), pdb_path, refresh))
        sifts = fetch_json(SIFTS_URL.format(pdb=pdb_id.lower()), sifts_path, refresh)
        mapping = author_to_uniprot(sifts, pdb_id, cfg.uniprot, chain)
        sequence = load_meta(cfg)["full_sequence"]
        chain_atoms = protein(self.atoms, chain)
        self.to_uniprot: dict[int, int] = {}
        mismatches = 0
        for n in np.unique(chain_atoms.resnum):
            unp = mapping.get(int(n))
            if unp is None:
                continue
            aa = THREE_TO_ONE.get(str(chain_atoms.resname[chain_atoms.resnum == n][0]))
            if aa != sequence[unp - 1]:
                mismatches += 1  # engineered mutation in the crystal construct
                continue
            self.to_uniprot[int(n)] = unp
        if mismatches > 0.1 * (len(self.to_uniprot) + mismatches):
            raise ValueError(f"{mismatches} residues of {pdb_id}:{chain} disagree with UniProt; mapping is wrong")
        self.chain, self.pdb_id = chain, pdb_id
        provenance[pdb_id] = {"file": pdb_path.name, "sha256": hashlib.sha256(pdb_path.read_bytes()).hexdigest(),
                              "sifts_sha256": hashlib.sha256(sifts_path.read_bytes()).hexdigest()}

    def uniprot(self, author_residues) -> set[int]:
        return {self.to_uniprot[n] for n in author_residues if n in self.to_uniprot}


def classify(position: int, sets: dict[str, set[int]], rsasa: dict[int, float], buried: float) -> str:
    for cls in INTERACTION_CLASSES:
        if position in sets[cls]:
            return cls
    if position not in rsasa:
        return "unresolved"
    return "core" if rsasa[position] < buried else "surface"


def annotate(cfg: TargetConfig, refresh: bool = False) -> pd.DataFrame | None:
    """Per-position mechanism table for the construct, or None if not configured."""
    m = cfg.mechanism
    if not (m.ligand or m.coactivator or m.dimer or m.burial):
        return None
    provenance: dict = {}
    sets: dict[str, set[int]] = {cls: set() for cls in INTERACTION_CLASSES}

    if m.ligand:
        s = _Structure(cfg, m.ligand["pdb_id"], m.ligand["chain"], refresh, provenance)
        hormone = s.atoms.select(s.atoms.hetero & (s.atoms.resname == m.ligand["ligand"]))
        if not len(hormone.xyz):
            raise ValueError(f"{s.pdb_id} has no {m.ligand['ligand']} ligand")
        sets["ligand"] |= s.uniprot(contact_residues(protein(s.atoms, s.chain), hormone, m.contact_cutoff))
    for cls, entries in (("coactivator", m.coactivator), ("dimer", m.dimer)):
        for e in entries:
            s = _Structure(cfg, e["pdb_id"], e["chain"], refresh, provenance)
            if "assembly" in e:
                # The partner is a symmetry copy: another MODEL of the biological-assembly file,
                # with the same author numbering as the deposited chain.
                n, path = e["assembly"], cfg.paths.raw / f"{s.pdb_id}.pdb{e['assembly']}"
                text = fetch_text(ASSEMBLY_URL.format(pdb=s.pdb_id, n=n), path, refresh)
                provenance[f"{s.pdb_id} assembly {n}"] = {"file": path.name,
                                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                site = protein(read_atoms(text, model=1), s.chain)
                partner = protein(read_atoms(text, model=e.get("partner_model", 2)), e.get("partner", s.chain))
            else:
                site, partner = protein(s.atoms, s.chain), protein(s.atoms, e["partner"])
            sets[cls] |= s.uniprot(contact_residues(site, partner, m.contact_cutoff))

    rsasa: dict[int, float] = {}
    if m.burial:
        s = _Structure(cfg, m.burial["pdb_id"], m.burial["chain"], refresh, provenance)
        rsasa = {s.to_uniprot[n]: v for n, v in relative_sasa(protein(s.atoms, s.chain)).items() if n in s.to_uniprot}

    start, end = cfg.construct.start, cfg.construct.end
    table = pd.DataFrame({"position": np.arange(start, end + 1)})
    table["rsasa"] = table["position"].map(rsasa)
    for cls in INTERACTION_CLASSES:
        table[f"in_{cls}_site"] = table["position"].isin(sets[cls])
    table["mechanism"] = [classify(int(p), sets, rsasa, m.buried_rsasa) for p in table["position"]]

    paths = cfg.paths
    paths.analysis.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.analysis / "mechanism.csv", index=False)
    (paths.analysis / "mechanism.json").write_text(json.dumps({
        "contact_cutoff": m.contact_cutoff, "buried_rsasa": m.buried_rsasa,
        "ligand": m.ligand, "coactivator": m.coactivator, "dimer": m.dimer, "burial": m.burial,
        "structures": provenance, "counts": table["mechanism"].value_counts().to_dict(),
    }, indent=2))
    log.info("mechanism classes: %s", table["mechanism"].value_counts().to_dict())
    return table
