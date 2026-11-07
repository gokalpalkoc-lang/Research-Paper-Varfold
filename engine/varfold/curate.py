"""Build the variant dataset: canonical sequence + clinically classified missense variants.

Sources
-------
* UniProtKB REST     — canonical sequence and positional annotations.
* EBI Proteins API   — variants aggregated from ClinVar, UniProt curation,
                       Ensembl/dbSNP and gnomAD, already mapped to UniProt numbering.
* gnomAD GraphQL API — carrier counts (optional, ``population:`` in the config), the
                       source of population-tolerated negatives.

Raw API responses are snapshotted under ``raw/`` with their SHA-256 in
``dataset.meta.json`` so the dataset in the paper can be reproduced exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import TargetConfig
from .sequences import AMINO_ACIDS, Mutation, SequenceError, apply_mutations, is_residue

log = logging.getLogger(__name__)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"
VARIATION_URL = "https://www.ebi.ac.uk/proteins/api/variation/{acc}"
GNOMAD_URL = "https://gnomad.broadinstitute.org/api"
GNOMAD_QUERY = """query VarfoldPopulation($gene: String!, $dataset: DatasetId!) {
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    gene_id canonical_transcript_id
    variants(dataset: $dataset) {
      variant_id consequence hgvsp transcript_id
      exome { ac an ac_hemi ac_hom } genome { ac an ac_hemi ac_hom }
    }
  }
}"""

# ClinVar review status -> star rating.
REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
}
SOMATIC_ONLY_SOURCES = {"NCI-TCGA", "COSMIC"}
# Disease names that place an assertion in a cancer context. A substitution that is
# pathogenic *only* for such diseases is relabelled "cancer": for AR these are the
# prostate-cancer resistance mutations (T878A, L702H, H875Y...), gain-of-function changes
# that keep the receptor folded, the opposite of the loss of function behind androgen
# insensitivity, so pooling them with the germline pathogenic class would blur the test.
CANCER_TERMS = re.compile(r"cancer|carcinoma|neoplasm|tumou?r|adenoma|melanoma|glioma|leuka?emia|lymphoma|sarcoma",
                          re.IGNORECASE)
# UniProt feature types worth showing alongside the structure.
ANNOTATION_TYPES = {"Domain", "Region", "Binding site", "Site", "Motif", "Helix", "Beta strand", "Turn"}

DATASET_COLUMNS = [
    "variant_id", "position", "wt", "mt", "construct_index", "label", "assertions",
    "review_stars", "uniprot_curated", "clinvar_ids", "dbsnp_ids", "max_gnomad_af",
    "gnomad_carriers", "diseases", "in_scan", "sequence",
]


def fetch_text(url: str, dest: Path, refresh: bool = False, accept: str = "*/*", post: dict | None = None) -> str:
    """GET ``url`` (or POST ``post`` as JSON), caching the raw body at ``dest`` (the provenance snapshot)."""
    if dest.exists() and not refresh:
        return dest.read_text()
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            if post is None:
                resp = requests.get(url, headers={"Accept": accept}, timeout=120)
            else:
                resp = requests.post(url, json=post, headers={"Accept": accept}, timeout=300)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(resp.text)
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning("fetch %s failed (%s); retrying in %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not fetch {url}") from last_exc


def fetch_json(url: str, dest: Path, refresh: bool = False, post: dict | None = None) -> dict:
    return json.loads(fetch_text(url, dest, refresh, accept="application/json", post=post))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(terms: set[str]) -> str:
    """Collapse all clinical assertions for one substitution into a single class.

    Rules (documented in the paper's methods):
      * any explicit "conflicting" assertion, or both P/LP and B/LB -> conflicting
      * P/LP (VUS assertions alongside are tolerated)                 -> pathogenic
      * B/LB (VUS assertions alongside are tolerated)                 -> benign
      * only VUS                                                      -> vus
      * anything else (risk factor, drug response, ...)               -> other
      * no assertion at all                                           -> unlabeled
    ``aggregate_variants`` then moves a pathogenic substitution whose diseases are all
    cancers (``CANCER_TERMS``) to "cancer", and ``build_dataset`` makes an unlabeled one
    carried by enough people in gnomAD "population".
    """
    has_p = any(t in ("pathogenic", "likely pathogenic") for t in terms)
    has_b = any(t in ("benign", "likely benign") for t in terms)
    if any("conflicting" in t for t in terms) or (has_p and has_b):
        return "conflicting"
    if has_p:
        return "pathogenic"
    if has_b:
        return "benign"
    if any("uncertain" in t for t in terms):
        return "vus"
    return "other" if terms else "unlabeled"


def _abbreviate(term: str) -> str:
    return {
        "pathogenic": "P", "likely pathogenic": "LP", "benign": "B", "likely benign": "LB",
        "variant of uncertain significance": "VUS",
    }.get(term, term)


def aggregate_variants(features: list[dict], sequence: str) -> tuple[list[dict], Counter]:
    """Group EBI variant records by substitution and validate them against ``sequence``."""
    grouped: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    dropped: Counter = Counter()
    for feat in features:
        if feat.get("type") != "VARIANT" or feat.get("consequenceType") != "missense":
            continue
        wt, mt = feat.get("wildType"), feat.get("mutatedType") or feat.get("alternativeSequence")
        try:
            pos = int(feat["begin"])
        except (KeyError, TypeError, ValueError):
            dropped["bad_position"] += 1
            continue
        if not (is_residue(wt) and is_residue(mt)) or wt == mt:
            dropped["nonsense_or_non_canonical"] += 1
            continue
        # Records mapped through a different isoform/numbering disagree with the
        # canonical residue; they must not be silently relabelled.
        if pos > len(sequence) or sequence[pos - 1] != wt:
            dropped["reference_mismatch"] += 1
            continue
        grouped[(pos, wt, mt)].append(feat)

    variants = []
    for (pos, wt, mt), records in sorted(grouped.items()):
        terms: list[str] = []
        starred: list[tuple[str, int]] = []  # (term, ClinVar review stars) per assertion
        clinvar, dbsnp, diseases = set(), set(), set()
        af = 0.0
        curated = False
        for rec in records:
            curated |= rec.get("sourceType") in ("uniprot", "mixed")
            for cs in rec.get("clinicalSignificances") or []:
                sources = set(cs.get("sources") or [])
                if sources and sources <= SOMATIC_ONLY_SOURCES:
                    continue
                cs_terms = [t.strip().lower() for t in (cs.get("type") or "").split(",") if t.strip()]
                terms.extend(cs_terms)
                if "ClinVar" in sources and cs.get("reviewStatus") in REVIEW_STARS:
                    starred.extend((t, REVIEW_STARS[cs["reviewStatus"]]) for t in cs_terms)
            for x in rec.get("xrefs") or []:
                if x.get("name") == "ClinVar" and x.get("id", "").startswith(("RCV", "VCV")):
                    clinvar.add(x["id"])
                elif x.get("name") == "dbSNP":
                    dbsnp.add(x["id"])
            for pf in rec.get("populationFrequencies") or []:
                if str(pf.get("source", "")).startswith("gnomAD"):
                    af = max(af, float(pf.get("frequency") or 0.0))
            for assoc in rec.get("association") or []:
                if assoc.get("disease") and assoc.get("name"):
                    diseases.add(assoc["name"])
        counts = Counter(_abbreviate(t) for t in terms)
        label = classify(set(terms))
        # Review stars describe the assertions *behind the label*, not the best assertion of any kind.
        stars = [s for t, s in starred if label == "conflicting" or classify({t}) == label]
        if label == "pathogenic" and diseases and all(CANCER_TERMS.search(d) for d in diseases):
            label = "cancer"
        variants.append({
            "variant_id": f"{wt}{pos}{mt}",
            "position": pos, "wt": wt, "mt": mt,
            "label": label,
            "assertions": ";".join(f"{k}:{v}" for k, v in sorted(counts.items())),
            "review_stars": max(stars) if stars else None,
            "uniprot_curated": curated,
            "clinvar_ids": ";".join(sorted(clinvar)),
            "dbsnp_ids": ";".join(sorted(dbsnp)),
            "max_gnomad_af": af or None,
            "diseases": ";".join(sorted(diseases)),
        })
    return variants, dropped


def population_counts(payload: dict, sequence: str, count: str) -> tuple[dict[str, int], Counter]:
    """Carriers per substitution (UniProt numbering) from a gnomAD gene query.

    Only the canonical transcript counts: the same genomic change has a different protein
    position on every other transcript. Exome and genome carriers are summed (separate
    cohorts), as are different codon changes that give the same substitution.
    """
    if payload.get("errors"):
        raise RuntimeError(f"gnomAD query failed: {payload['errors'][0].get('message')}")
    gene = payload["data"]["gene"]
    carriers: dict[str, int] = defaultdict(int)
    dropped: Counter = Counter()
    for v in gene["variants"]:
        if v.get("consequence") != "missense_variant" or v.get("transcript_id") != gene["canonical_transcript_id"]:
            continue
        try:
            mut = Mutation.parse(v.get("hgvsp") or "")
        except SequenceError:
            dropped["gnomad_unparsed"] += 1
            continue
        if mut.position > len(sequence) or sequence[mut.position - 1] != mut.wt:
            dropped["gnomad_reference_mismatch"] += 1
            continue
        carriers[mut.id] += sum((v.get(cohort) or {}).get(count) or 0 for cohort in ("exome", "genome"))
    return dict(carriers), dropped


def scan_mutations(construct_seq: str, start: int, mode: str) -> list[Mutation]:
    if mode == "none":
        return []
    muts = []
    for i, wt in enumerate(construct_seq):
        targets = "A" if mode == "alanine" else AMINO_ACIDS
        muts.extend(Mutation(start + i, wt, mt) for mt in targets if mt != wt)
    return muts


def slim_annotations(entry: dict, start: int, end: int) -> list[dict]:
    """UniProt features overlapping the construct, for figures and the web viewer."""
    out = []
    for f in entry.get("features", []):
        if f.get("type") not in ANNOTATION_TYPES:
            continue
        try:
            a = int(f["location"]["start"]["value"])
            b = int(f["location"]["end"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if b < start or a > end:
            continue
        ligand = (f.get("ligand") or {}).get("name")
        out.append({"type": f["type"], "start": a, "end": b,
                    "description": f.get("description") or ligand or ""})
    return out


def build_dataset(cfg: TargetConfig, refresh: bool = False) -> pd.DataFrame:
    paths = cfg.paths
    paths.raw.mkdir(parents=True, exist_ok=True)
    uniprot_raw = paths.raw / f"uniprot_{cfg.uniprot}.json"
    variation_raw = paths.raw / f"variation_{cfg.uniprot}.json"

    entry = fetch_json(UNIPROT_URL.format(acc=cfg.uniprot), uniprot_raw, refresh)
    variation = fetch_json(VARIATION_URL.format(acc=cfg.uniprot), variation_raw, refresh)

    full_seq = entry["sequence"]["value"]
    if variation.get("sequence") and variation["sequence"] != full_seq:
        raise RuntimeError("UniProt and EBI Proteins API disagree on the canonical sequence; "
                           "re-run with --refresh so both snapshots are from the same release")
    start, end = cfg.construct.start, cfg.construct.end
    if end > len(full_seq):
        raise ValueError(f"construct end {end} exceeds sequence length {len(full_seq)}")
    construct_seq = full_seq[start - 1:end]

    variants, dropped = aggregate_variants(variation.get("features", []), full_seq)
    in_construct = [v for v in variants if start <= v["position"] <= end]
    dropped["outside_construct"] = len(variants) - len(in_construct)

    pop = cfg.population
    carriers: dict[str, int] | None = None
    gnomad_raw = paths.raw / f"{pop.dataset}_{cfg.gene}.json"
    if pop.dataset != "none":
        payload = fetch_json(GNOMAD_URL, gnomad_raw, refresh,
                             post={"query": GNOMAD_QUERY, "variables": {"gene": cfg.gene, "dataset": pop.dataset}})
        carriers, gnomad_dropped = population_counts(payload, full_seq, pop.count)
        dropped.update(gnomad_dropped)

    def n_carriers(vid: str) -> int | None:
        return None if carriers is None else carriers.get(vid, 0)

    keep_classes = set(cfg.variants.include_classes)
    min_stars = cfg.variants.min_review_stars
    rows: dict[str, dict] = {}
    for v in in_construct:
        if v["label"] not in keep_classes:
            continue
        stars = v["review_stars"]
        if min_stars > 0 and not v["uniprot_curated"] and (stars is None or stars < min_stars):
            dropped["below_review_stars"] += 1
            continue
        rows[v["variant_id"]] = {**v, "gnomad_carriers": n_carriers(v["variant_id"]), "in_scan": False}

    by_id = {v["variant_id"]: v for v in in_construct}

    def unlabeled_row(mut: Mutation, label: str, in_scan: bool) -> dict:
        base = by_id.get(mut.id, {})
        return {
            "variant_id": mut.id, "position": mut.position, "wt": mut.wt, "mt": mut.mt,
            "label": label, "assertions": base.get("assertions", ""),
            "review_stars": base.get("review_stars"), "uniprot_curated": base.get("uniprot_curated", False),
            "clinvar_ids": base.get("clinvar_ids", ""), "dbsnp_ids": base.get("dbsnp_ids", ""),
            "max_gnomad_af": base.get("max_gnomad_af"), "gnomad_carriers": n_carriers(mut.id),
            "diseases": base.get("diseases", ""), "in_scan": in_scan,
        }

    # Population-tolerated negatives: only substitutions nobody has classified clinically
    # (a clinical label always wins, so pathogenic variants seen in carriers stay pathogenic).
    if carriers is not None and "population" in keep_classes:
        for vid, n in sorted(carriers.items()):
            mut = Mutation.parse(vid)
            if (n >= pop.min_carriers and start <= mut.position <= end and vid not in rows
                    and by_id.get(vid, {}).get("label", "unlabeled") == "unlabeled"):
                rows[vid] = unlabeled_row(mut, "population", in_scan=False)

    for mut in scan_mutations(construct_seq, start, cfg.variants.scan):
        if mut.id in rows:
            rows[mut.id]["in_scan"] = True
        else:
            rows[mut.id] = unlabeled_row(mut, "unlabeled", in_scan=True)

    records = [{
        "variant_id": "WT", "position": None, "wt": None, "mt": None, "construct_index": None,
        "label": "wild_type", "assertions": "", "review_stars": None, "uniprot_curated": False,
        "clinvar_ids": "", "dbsnp_ids": "", "max_gnomad_af": None, "gnomad_carriers": None, "diseases": "",
        "in_scan": False, "sequence": construct_seq,
    }]
    for row in sorted(rows.values(), key=lambda r: (r["position"], r["mt"])):
        mut = Mutation(row["position"], row["wt"], row["mt"])
        row["construct_index"] = row["position"] - start + 1
        row["sequence"] = apply_mutations(construct_seq, [mut], offset=start)
        records.append(row)

    df = pd.DataFrame.from_records(records, columns=DATASET_COLUMNS)
    df["position"] = df["position"].astype("Int64")
    df["construct_index"] = df["construct_index"].astype("Int64")
    df["review_stars"] = df["review_stars"].astype("Int64")
    df["gnomad_carriers"] = df["gnomad_carriers"].astype("Int64")
    paths.root.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.dataset, index=False)

    meta = {
        "target": {"name": cfg.name, "title": cfg.title, "gene": cfg.gene, "uniprot": cfg.uniprot,
                   "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {})
                   .get("fullName", {}).get("value")},
        "uniprot_entry_version": entry.get("entryAudit", {}).get("entryVersion"),
        "uniprot_sequence_version": entry.get("entryAudit", {}).get("sequenceVersion"),
        "retrieved_at": datetime.fromtimestamp(uniprot_raw.stat().st_mtime, timezone.utc).isoformat(),
        "sources": {
            "uniprot": {"url": UNIPROT_URL.format(acc=cfg.uniprot), "file": uniprot_raw.name,
                        "sha256": _sha256(uniprot_raw)},
            "variation": {"url": VARIATION_URL.format(acc=cfg.uniprot), "file": variation_raw.name,
                          "sha256": _sha256(variation_raw)},
        },
        "construct": {"start": start, "end": end, "length": len(construct_seq), "sequence": construct_seq},
        "full_length": len(full_seq),
        "full_sequence": full_seq,
        "filters": {"include_classes": sorted(keep_classes), "min_review_stars": min_stars,
                    "scan": cfg.variants.scan,
                    "population": {"dataset": pop.dataset, "count": pop.count, "min_carriers": pop.min_carriers}},
        "dropped": dict(dropped),
        "counts": df["label"].value_counts().to_dict(),
        "n_structures": len(df),
        "annotations": slim_annotations(entry, start, end),
    }
    if carriers is not None:
        meta["sources"]["gnomad"] = {"url": GNOMAD_URL, "dataset": pop.dataset, "gene": cfg.gene,
                                     "query": GNOMAD_QUERY, "file": gnomad_raw.name, "sha256": _sha256(gnomad_raw)}
    paths.dataset_meta.write_text(json.dumps(meta, indent=2))
    log.info("dataset: %d sequences (%s); dropped %s", len(df), meta["counts"], dict(dropped))
    return df


def load_dataset(cfg: TargetConfig) -> pd.DataFrame:
    if not cfg.paths.dataset.exists():
        raise FileNotFoundError(f"{cfg.paths.dataset} missing — run `varfold curate` first")
    return pd.read_csv(cfg.paths.dataset, dtype={"review_stars": "Int64", "position": "Int64",
                                                  "construct_index": "Int64", "gnomad_carriers": "Int64"},
                       keep_default_na=True, na_values=[""])


def load_meta(cfg: TargetConfig) -> dict:
    """Dataset metadata; refuses to continue if the config's construct has changed since curation."""
    meta = json.loads(cfg.paths.dataset_meta.read_text())
    c = meta["construct"]
    if (c["start"], c["end"]) != (cfg.construct.start, cfg.construct.end):
        raise RuntimeError(f"config construct {cfg.construct.start}-{cfg.construct.end} differs from the curated "
                           f"dataset ({c['start']}-{c['end']}); re-run `varfold curate` (and fold) first")
    return meta
