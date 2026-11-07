"""AlphaMissense scores (Cheng et al., Science 2023): the published state-of-the-art missense
pathogenicity predictor, and the strongest baseline a structural metric has to beat.

The scores are not downloaded automatically. Save the per-protein file from the AlphaFold
Database (the ``amAnnotationsUrl`` listed by https://alphafold.ebi.ac.uk/api/prediction/<UniProt>,
named ``AF-<UniProt>-F1-aa-substitutions.csv``) into the run's ``raw/`` folder. The matching
rows of the genome-wide release (``AlphaMissense_aa_substitutions.tsv.gz``) work too. Both use
UniProt canonical numbering, the same as the dataset.
"""

from __future__ import annotations

import hashlib
import io
import logging

import pandas as pd

from .config import TargetConfig

log = logging.getLogger(__name__)


def load_scores(cfg: TargetConfig) -> tuple[pd.DataFrame | None, dict | None]:
    """(variant_id, alphamissense) per substitution plus the file's provenance, or (None, None)."""
    path = cfg.paths.alphamissense(cfg.uniprot)
    if not path.exists():
        log.warning("no AlphaMissense scores at %s; that baseline is skipped (see varfold/alphamissense.py)", path)
        return None, None
    # The genome-wide release is tab-separated under '#' licence lines (its column header may
    # carry a '#' too); the per-protein file is a plain CSV.
    lines = [line.lstrip("#") if "protein_variant" in line else line for line in path.read_text().splitlines()
             if "protein_variant" in line or not line.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=None, engine="python")
    if "uniprot_id" in df.columns:
        df = df[df["uniprot_id"] == cfg.uniprot]
    missing = {"protein_variant", "am_pathogenicity"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} lacks column(s) {sorted(missing)}")
    scores = df.rename(columns={"protein_variant": "variant_id", "am_pathogenicity": "alphamissense"})
    provenance = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "n_scores": len(scores)}
    return scores[["variant_id", "alphamissense"]], provenance
