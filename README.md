# VarFold

**Predicted 3D structures and structural-confidence analysis of clinical missense variants.**

VarFold folds every clinically classified missense variant in a protein domain with
[ESMFold](https://github.com/facebookresearch/esm), measures where the model loses
confidence relative to the wild type, and tests whether those "structural breaking
points" track pathogenicity.

This branch (`active`) is the research copy used for the paper: everything runs on your
own machine, and nothing here deploys to a server. The public website is the **`demo`**
branch (local copy: `~/Desktop/VarFold/varfold-demo`, live at <https://varfold.devtga.com>),
which only serves a bundle exported from here. The two share history, so a fix made on one
can be moved to the other with `git cherry-pick`.

```
 ┌──────────── this branch: research PC (GPU) ────────────┐         ┌──── demo branch: devtga.com ────┐
 │ varfold curate ─► fold ─► llr ─► validate ─► analyze ─► │ publish │  FastAPI + React (3Dmol)        │
 │ export ──► data/ar_lbd/web/varfold.sqlite ──────────────┼────────►│  read-only, no PyTorch, no GPU  │
 └─────────────────────────────────────────────────────────┘         └─────────────────────────────────┘
```

![Overview page](docs/overview.png)

## Run it on your computer

You need Python 3.10+ (3.12 recommended), Node 20+, `git` and `make`.

```bash
git clone -b active https://github.com/gokalpalkoc-lang/Research-Paper-Varfold.git varfold-active
cd varfold-active
```

**1. Just look at it (2 minutes, no GPU, no model download).** Uses the bundled demo
dataset: the wild type plus 9 clinical variants, folded with ESMFold.

```bash
make setup-web
make demo                  # open http://127.0.0.1:8000
```

**2. Live folding on your own machine.** Installs PyTorch and downloads the ESMFold
weights (~8 GB, once). Then type or mutate any sequence on the *Live fold* page and
your computer folds it. An NVIDIA GPU makes this take seconds; on a CPU expect a few
minutes for ~250 residues, and you'll need about 20 GB of free RAM.

```bash
make setup
make live                  # open http://127.0.0.1:8000 → Live fold
```

**3. Reproduce the research dataset.** Curates the variants, folds all of them,
computes the statistics and figures, and serves your own results.

```bash
.venv/bin/varfold all      # resumable; ~442 structures (hours on a GPU, ~1 day on a CPU)
make live                  # the site now shows your data instead of the demo
```

Tip for CPUs with performance and efficiency cores (e.g. Intel Core Ultra):
`export VARFOLD_THREADS=<number of performance cores>` is about 2× faster than the default.

![Variant page](docs/variant.png)

Default target: **androgen receptor ligand-binding domain** (AR, UniProt P10275,
residues 669–919). It has by far the most labelled missense variants of the
hormone receptors checked (androgen insensitivity syndrome), crystal structures to
validate against, and fits ESMFold on an 8 GB GPU. The curated dataset is
**441 variants plus the wild type**: 294 pathogenic/likely pathogenic, 21 benign/likely
benign, 79 VUS, 11 pathogenic only in a cancer context (prostate-cancer resistance
mutations, kept apart as a negative control) and 36 clinically unclassified variants
carried by hemizygous men in gnomAD v4 (extra negatives for a secondary analysis).
Any other target is one YAML file away (see below).

---

## 1. Setup

### NVIDIA driver (for GPU inference)

Without CUDA everything runs on the CPU, correctly but slowly (~4 min per 251-residue
structure on a Core Ultra 9 185H). The pinned PyTorch build (CUDA 13.0) needs NVIDIA
driver **≥ 580**; Blackwell cards (RTX 50xx) need the *open* kernel modules. On Ubuntu:

```bash
nvidia-smi                          # already working? skip this section
sudo ubuntu-drivers install         # or: sudo apt install nvidia-driver-580-open
sudo reboot                         # with Secure Boot on, enrol the MOK key when prompted
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

`precision: cpu-lm` (the AR config's setting) keeps ESMFold's 3B language model on
the CPU in fp32 and runs the folding trunk on the GPU, which needs ~3 GB of VRAM.
`precision: fp16` puts the language model on the GPU in half precision instead
(faster, ~8.4 GB, too much for 8 GB laptop cards that expose 7.5 GB).

### Python + Node

```bash
make setup     # Python 3.12 venv with engine (ml, serve, dev) + backend deps, and npm ci
make test      # engine + API tests and the frontend type-check (no model needed)
cd engine && ../.venv/bin/python -m pytest -m model   # integration tests with the real weights
```

Model weights (~11 GB: ESMFold 8.4 GB, ESM-2 650M 2.5 GB) are downloaded to the
Hugging Face cache on first use. After that, `export HF_HUB_OFFLINE=1` makes runs
fully offline.

---

## 2. Research workflow

Every step is resumable and writes into `data/<target>/`.

```bash
source .venv/bin/activate
varfold curate      # UniProt + EBI Proteins API + gnomAD → dataset.csv (raw API snapshots kept with SHA-256)
varfold fold        # batch ESMFold; Ctrl-C/kill any time, re-run to resume
varfold status      # progress + time-remaining estimate
varfold llr         # ESM-2 masked-marginal baseline scores
                    # AlphaMissense baseline: save its scores by hand once (see below)
varfold validate    # WT prediction vs PDB 2AM9 (SIFTS residue mapping)
varfold analyze     # metrics, statistics, figures, report.md
varfold export      # web bundle: data/ar_lbd/web/varfold.sqlite
```

`varfold all` runs the whole chain. For the overnight batch, run it detached and stop
the live worker first (they share one GPU):

```bash
systemctl --user stop varfold-worker 2>/dev/null
systemd-run --user --unit varfold-batch --working-directory="$PWD" .venv/bin/varfold all
journalctl --user -fu varfold-batch          # or: tail -f data/ar_lbd/varfold.log
```

**Runtime.**

- **Measured on CPU** (Core Ultra 9 185H, fp32, 4 recycles): **~4 min** per
  251-residue structure, so about 26 h for the 442-structure set. That's fine
  for testing; production runs need the GPU.
- **CPU threads:** this hybrid CPU is fastest with `VARFOLD_THREADS=6` (one per P-core).
  Using all 22 logical CPUs is about 2× slower than PyTorch's default.
- **Measured on GPU** (RTX 5070 Laptop, `precision: cpu-lm`): **~12 s** per structure,
  so about 1.5 h for the 442-structure set; the `saturation` scan (~4,800 structures)
  is an overnight run. The card exposes 7.5 GB, which is too little for `fp16`
  (~8.4 GB), so the 3B language model runs on the CPU in fp32 and only the folding
  trunk on the GPU (~2.8 GB). Against the CPU fp32 folds, mean pLDDT agrees to 0.01
  and Δ local pLDDT to 0.02 (WT, R780W, L881V).

**Reproducibility guarantees**

- Each prediction records a settings fingerprint (`model | recycles | precision`, where
  precision is `fp32` on CPU and `fp16-lm/tf32` or `fp32/tf32` on CUDA). Structures are
  stored per fingerprint (`structures/<fingerprint-hash>/`), so runs never overwrite or
  get paired with each other.
  Analysis compares a variant only with a wild type folded under the *same*
  fingerprint and reports anything else as excluded. CPU (fp32) test runs and GPU
  (fp16 language model) production runs are therefore never mixed. After switching to
  the GPU, `varfold fold` refolds everything automatically.
- Raw API responses are frozen in `raw/` and hashed in `dataset.meta.json`; re-curate
  from them, or pass `--refresh` to take a new snapshot.
- Weights are loaded from the model repository's main branch, in the format it
  publishes. transformers' automatic download of an unreviewed safetensors
  conversion PR is disabled.

### Outputs for the paper

| File | Contents |
|---|---|
| `dataset.csv`, `dataset.meta.json` | The curated variant set, labels, ClinVar/dbSNP ids, gnomAD carrier counts, provenance |
| `structures/<fp>/*.pdb`, `pae/<fp>/*.npz` | Every predicted structure (B-factor = pLDDT) and its PAE matrix, per settings fingerprint |
| `predictions.jsonl` | Per-residue pLDDT, pTM, runtime, device, versions per structure |
| `analysis/metrics.csv` | Per-variant ΔpLDDT (local/site/global), Cα RMSD, lDDT-Cα, LLR, breaking-point q-values |
| `analysis/positions.csv` | Per-residue aggregates along the domain |
| `analysis/stats.json`, `analysis/report.md` | All tests in machine-readable form + a narrative summary with tables |
| `analysis/figures/fig1–6.{pdf,png}` | WT pLDDT profile, ΔpLDDT by class, ROC, breaking-point map, structure-vs-sequence, saturation heatmap |
| `validation.json` | WT prediction vs crystal structure (RMSD, lDDT, pLDDT calibration) |

### Methodology in brief

- **Primary metric:** Δ local pLDDT, the mean change in Cα pLDDT within ±5 residues of
  the mutation.
- **Breaking points:** ESMFold is deterministic, so there are no replicate runs to
  estimate noise from. Instead, the same windowed statistic is computed at residues
  windows lying entirely more than 15 Å from each mutation, where a substitution should
  have no direct effect.
  That gives an empirical null; variants beyond it at BH-FDR < 0.05 are breaking points.
- **Discrimination:** pathogenic vs benign AUC with stratified bootstrap 95% CIs,
  Mann–Whitney U, Cliff's δ, and BH q-values across metrics. A sensitivity analysis
  repeats it on high-confidence labels only.
- **Baselines a structural metric must beat:** the ESM-2 log-likelihood ratio
  (sequence-only zero-shot predictor), BLOSUM62 (substitution chemistry alone; this
  matters because the benign set is dominated by conservative swaps such as I→V and
  D→E), and site priors (WT pLDDT and burial at the site, which ignore which residue
  is substituted). "Does structure add anything?" is answered by the paired ΔAUC on the
  same variants: a DeLong 95% CI, with the p-value from a paired permutation test.
  DeLong's own p is liberal with only 21 benign variants (7–8% type-I error at nominal
  5% in simulation; the permutation test stays near 5%).
- **AlphaMissense** (the published state-of-the-art predictor) is the strongest baseline.
  It is not downloaded automatically: open the `amAnnotationsUrl` listed at
  <https://alphafold.ebi.ac.uk/api/prediction/P10275> in a browser and save the file as
  `data/ar_lbd/raw/AF-P10275-F1-aa-substitutions.csv`. Until it is there, analysis skips
  that baseline with a warning; once it is, its SHA-256 goes into `stats.json`.
- **Label classes beyond P/B/VUS.** A pathogenic variant whose diseases are all cancers
  becomes `cancer`: for AR these are prostate-cancer resistance mutations (T878A, L702H,
  H875Y…), gain-of-function changes that keep the receptor folded. They test the method
  as a *negative control*: germline pathogenic variants should be breaking points more
  often than they are (one-sided Fisher). A clinically unclassified variant carried by at
  least one hemizygous XY person in gnomAD v4 becomes `population` (config `population:`).
  Benign + population is a larger, noisier negative set for a secondary analysis; a
  clinical label always wins, because mild androgen insensitivity is compatible with a
  male phenotype and 13 ClinVar-pathogenic LBD variants do have hemizygous carriers.
- **Enrichment:** whether pathogenic variants become breaking points more often than
  benign ones is a variant-level one-sided Fisher test. A residue-level version would be
  confounded by how many variants each residue carries.
- **Mechanism map** (`mechanism:` in the config, `analysis/mechanism.csv`): every residue
  is tagged from human AR crystal structures as a testosterone contact (2AM9), coactivator
  groove (TIF2 peptide, 2Q7J; AR N-terminal FQNLF peptide, 2Q7I), dimer interface (5JJM,
  chain A and its symmetry copy in biological assembly 1), buried core (relative
  accessibility < 0.20 in 2AM9) or surface; 4 Å heavy-atom contacts, SIFTS numbering,
  first matching class wins. ESMFold folds one chain with no hormone or partner, so the
  test is within the pathogenic class: core variants should be breaking points more often
  than interaction-site variants (one-sided Fisher), with per-class AUCs alongside.
  Pathogenic variants are mostly buried (151 core vs 84 surface) while the negatives are
  mostly on the surface, so a pooled AUC partly measures burial; read it per class.
- **Expect a sober result.** ESMFold was not trained to predict mutation effects, and
  single substitutions often leave predictions nearly unchanged. "pLDDT tracks
  pathogenicity only at buried core positions" or "structure adds little over ESM-2"
  are legitimate findings. The pipeline is built to measure that honestly, not to
  manufacture a signal. With 21 benign variants, CIs are wide; say so in the paper.
- **Numbering:** everything is UniProt canonical numbering. AR literature often uses an
  older numbering (e.g. "T877A" = T878A here).

### Changing the target

Copy `configs/ar_lbd.yaml` and set `uniprot`, `construct.start/end` (keep it under
~400 aa for 8 GB VRAM) and optionally `reference_structure.pdb_id`. Then:

```bash
varfold --config configs/my_target.yaml curate
```

Set `variants.scan: alanine | saturation` to add systematic in-silico mutagenesis.

---

## 3. Showcase web app

### Local development

```bash
make dev-api     # FastAPI on :8000 serving data/ar_lbd/web/varfold.sqlite
make dev-web     # Vite on :5173 with /api proxied to :8000
```

Pages: **Overview** (headline numbers, WT structure, breaking-point map,
discrimination table), **Variants** (filter/sort/search, URL-shareable state),
**Variant detail** (3D mutant vs WT ghost, pLDDT or ΔpLDDT colouring, synchronized
per-residue charts, ClinVar links, PDB download), **Live fold**, and **Methods**
(generated from the dataset's own provenance). The JSON API is documented at
`/api/docs`.

### Publishing to the demo site

The public site is the `demo` branch; it never folds anything and reads only the bundle
exported here. After `varfold export`:

```bash
cd ../varfold-demo
./scripts/publish.sh ../varfold-active/data/ar_lbd/web/varfold.sqlite
```

Live folding is off on the public site. The worker (`varfold serve`,
[deploy/varfold-worker.service](deploy/varfold-worker.service)) stays here because it needs
the GPU; `make live` uses it to fold on this machine only.

---

## Repository layout

```
configs/            target definitions (YAML)
engine/varfold/     research engine: curate, folding, llr, validate, metrics, stats, figures, analyze, export, worker, cli
engine/tests/       unit + end-to-end tests (synthetic run with planted signal; no model needed)
web/backend/app/    FastAPI: read-only dataset API, live-fold proxy, SPA hosting
web/backend/tests/  API tests against a bundle built by the real exporter
web/frontend/       React + TypeScript + Vite; 3Dmol.js viewer, SVG charts
deploy/             systemd unit for the live-fold worker (server files are on the demo branch)
data/<target>/      generated (git-ignored)
```

## Credits

ESMFold & ESM-2 (Meta AI, MIT licence) · UniProt · EBI Proteins API · ClinVar ·
PDBe SIFTS · RCSB PDB · 3Dmol.js (BSD). Research use only. Nothing here is a
clinical interpretation.
