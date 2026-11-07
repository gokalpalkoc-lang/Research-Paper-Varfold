"""Target configuration (YAML) and the on-disk layout of a run."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# "cancer": pathogenic only in a cancer context (see curate.CANCER_TERMS).
CLINICAL_CLASSES = ("pathogenic", "benign", "vus", "conflicting", "cancer")
# "population": clinically unclassified, but carried by people in gnomAD (see PopulationOptions).
VARIANT_CLASSES = CLINICAL_CLASSES + ("population",)
SCAN_MODES = ("none", "alanine", "saturation")
POPULATION_DATASETS = ("none", "gnomad_r4")
CARRIER_COUNTS = ("ac_hemi", "ac_hom")


@dataclass
class Construct:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class ReferenceStructure:
    pdb_id: str | None = None
    chain: str = "A"


@dataclass
class VariantOptions:
    include_classes: list[str] = field(default_factory=lambda: list(CLINICAL_CLASSES))
    min_review_stars: int = 0
    scan: str = "none"


@dataclass
class PopulationOptions:
    # gnomAD release to take carrier counts from; "none" turns population variants off.
    dataset: str = "none"
    # Who counts as a carrier: hemizygous XY people (ac_hemi) for X-linked genes, homozygous
    # people (ac_hom) for autosomal ones. Carrying the variant in one copy only says little.
    count: str = "ac_hemi"
    # An unclassified variant seen in at least this many carriers becomes "population".
    min_carriers: int = 1


@dataclass
class MechanismOptions:
    """Experimental structures that tag each residue with a mechanism (see mechanism.py)."""
    ligand: dict | None = None             # {pdb_id, chain, ligand}: hormone-contact residues
    coactivator: list = field(default_factory=list)  # [{pdb_id, chain, partner}]: AF-2 groove
    # [{pdb_id, chain, partner} or {pdb_id, chain, assembly, partner_model}]: dimer interface
    dimer: list = field(default_factory=list)
    burial: dict | None = None             # {pdb_id, chain}: relative solvent accessibility
    contact_cutoff: float = 4.0            # Å between heavy atoms
    buried_rsasa: float = 0.20             # below this relative accessibility a residue is core


@dataclass
class FoldingOptions:
    model: str = "facebook/esmfold_v1"
    num_recycles: int = 4
    chunk_size: int | None = 64
    precision: str = "fp16"
    device: str = "auto"
    save_pae: bool = True


@dataclass
class LLROptions:
    model: str = "facebook/esm2_t33_650M_UR50D"
    context: str = "full"


@dataclass
class AnalysisOptions:
    local_window: int = 5
    breaking_point_fdr: float = 0.05
    bootstrap: int = 2000
    permutations: int = 5000
    seed: int = 0


@dataclass
class TargetConfig:
    name: str
    title: str
    gene: str
    uniprot: str
    construct: Construct
    reference_structure: ReferenceStructure = field(default_factory=ReferenceStructure)
    variants: VariantOptions = field(default_factory=VariantOptions)
    population: PopulationOptions = field(default_factory=PopulationOptions)
    mechanism: MechanismOptions = field(default_factory=MechanismOptions)
    folding: FoldingOptions = field(default_factory=FoldingOptions)
    llr: LLROptions = field(default_factory=LLROptions)
    analysis: AnalysisOptions = field(default_factory=AnalysisOptions)
    output_dir: str | None = None
    source_path: str | None = None

    @property
    def paths(self) -> "RunPaths":
        root = Path(self.output_dir) if self.output_dir else REPO_ROOT / "data" / self.name
        return RunPaths(root)

    def validate(self) -> None:
        if self.construct.start < 1 or self.construct.end < self.construct.start:
            raise ValueError(f"invalid construct range {self.construct}")
        bad = set(self.variants.include_classes) - set(VARIANT_CLASSES)
        if bad:
            raise ValueError(f"unknown variant classes {sorted(bad)}; choose from {VARIANT_CLASSES}")
        if self.variants.scan not in SCAN_MODES:
            raise ValueError(f"variants.scan must be one of {SCAN_MODES}")
        if self.population.dataset not in POPULATION_DATASETS:
            raise ValueError(f"population.dataset must be one of {POPULATION_DATASETS}")
        if self.population.count not in CARRIER_COUNTS:
            raise ValueError(f"population.count must be one of {CARRIER_COUNTS}")
        if self.population.min_carriers < 1:
            raise ValueError("population.min_carriers must be at least 1")
        if self.folding.precision not in ("fp16", "fp32", "cpu-lm"):
            raise ValueError("folding.precision must be fp16, fp32 or cpu-lm")
        if self.llr.context not in ("full", "construct"):
            raise ValueError("llr.context must be full or construct")


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset.csv"

    @property
    def dataset_meta(self) -> Path:
        return self.root / "dataset.meta.json"

    @property
    def structures(self) -> Path:
        return self.root / "structures"

    @property
    def pae(self) -> Path:
        return self.root / "pae"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.jsonl"

    @property
    def failures(self) -> Path:
        return self.root / "failures.jsonl"

    @property
    def llr(self) -> Path:
        return self.root / "llr.csv"

    def alphamissense(self, uniprot: str) -> Path:
        """Per-protein AlphaMissense scores, saved by hand (see alphamissense.py)."""
        return self.raw / f"AF-{uniprot}-F1-aa-substitutions.csv"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    @property
    def figures(self) -> Path:
        return self.analysis / "figures"

    @property
    def web_db(self) -> Path:
        return self.root / "web" / "varfold.sqlite"


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    data = dict(data or {})
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data.pop(f.name)
        sub = f.type if isinstance(f.type, type) else globals().get(str(f.type))
        if sub is not None and is_dataclass(sub) and isinstance(value, dict):
            value = _build(sub, value)
        kwargs[f.name] = value
    if data:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(data)}")
    return cls(**kwargs)


def load_config(path: str | Path) -> TargetConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cfg = _build(TargetConfig, raw)
    cfg.source_path = str(path.resolve())
    if cfg.output_dir and not Path(cfg.output_dir).is_absolute():
        cfg.output_dir = str((REPO_ROOT / cfg.output_dir).resolve())
    cfg.validate()
    return cfg
