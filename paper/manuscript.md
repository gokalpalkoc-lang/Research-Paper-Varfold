# Does predicted-structure confidence add information beyond burial and AlphaMissense? A case study of androgen receptor ligand-binding-domain variants

> **DRAFT v0.1 (2026-11-08).** Every number below is copied from `data/ar_lbd/analysis/report.md` and `stats.json`, regenerated on 2026-11-08. Items in [square brackets] are unfinished: citations to verify, author details, or data-availability links. Nothing here has been peer reviewed.

## Abstract

**Background.** Protein structure predictors such as ESMFold report a per-residue confidence (pLDDT). It has been suggested that a drop in confidence when a single amino acid is substituted could flag damaging missense variants. Whether this adds information beyond how buried the site is, or beyond dedicated variant-effect predictors, is unclear.

**Methods.** We folded the androgen receptor ligand-binding domain (UniProt P10275, residues 669–919) with ESMFold for the wild type and 441 missense variants (294 pathogenic/likely pathogenic, 21 benign/likely benign, 36 gnomAD carrier variants, 79 of uncertain significance, 11 cancer-context). We compared pathogenic variants with benign and with "tolerated" (benign plus gnomAD carrier) variants, using five structural change metrics, burial (relative solvent accessibility of the wild-type site), ESM-2, BLOSUM62 and AlphaMissense. We tested whether results survive trimming the construct ends, and whether structure adds to burial by cross-validated modelling.

**Results.** The wild-type prediction matched crystal structure 2AM9 (Cα RMSD 0.90 Å). Structural change separated pathogenic from tolerated variants (AUC 0.70–0.76) and the signal persisted after dropping 30 residues from each end (AUC 0.72). However, burial alone performed as well or better (cross-validated AUC 0.77 versus 0.70–0.75 for structure), and adding structure to burial changed AUC by +0.021 (95% CI −0.021 to +0.063). AlphaMissense (AUC 0.82) outperformed structural metrics against tolerated variants (ΔAUC −0.12 to −0.06), and burial improved AlphaMissense (+0.059, 95% CI +0.003 to +0.115).

**Conclusions.** In this domain, mutant-versus-wild-type confidence change carries a real but modest signal that is largely accounted for by burial. Refolding each variant did not add measurable information beyond a single wild-type structure plus AlphaMissense.

## 1. Introduction

Interpreting missense variants is a central bottleneck in clinical genetics. Dedicated predictors such as AlphaMissense [CITE Cheng et al. 2023] score every possible substitution, but report a number without a reason. Structure predictors such as AlphaFold and ESMFold [CITE Lin et al. 2023] are not trained to predict mutation effects, and several authors argue they are largely insensitive to single substitutions [CITE Buel & Walters 2022], while others report that structural change in predicted structures tracks mutation effects [CITE McBride et al. 2023]. A natural, cheap diagnostic is the change in the model's own confidence between mutant and wild type.

Two issues make this hard to interpret. First, confidence at a position depends on how buried it is, and burial is already a strong predictor of pathogenicity; any apparent signal might be burial in disguise [CITE structure-feature predictors, e.g. AlphScore]. Second, monomer predictors cannot see partners, so variants that act by disrupting interactions rather than folding may be invisible to them [CITE Livesey & Marsh 2022].

We test both on one well-characterised domain, the androgen receptor ligand-binding domain (AR LBD). Loss-of-function variants cause androgen insensitivity syndrome, the domain has crystal structures with ligand, coactivator peptides and a dimer, and it has many labelled variants. We ask: (i) does the confidence change separate pathogenic from tolerated variants, and is that robust to artefacts of truncating the protein; (ii) does it add information beyond burial and beyond AlphaMissense; and (iii) is it weaker at sites that act through interactions?

## 2. Methods

**Variants and labels.** Missense variants of AR (UniProt P10275, canonical numbering) were taken from UniProt and the EBI Proteins API variation endpoint, with population counts from gnomAD v4 [CITE]. Variants outside residues 669–919 (n = 2,477), with a mismatched reference residue (n = 214) or non-canonical/nonsense consequences (n = 51) were dropped. Classes: *pathogenic* (pathogenic/likely pathogenic; 294 variants, of which 158 have ≥1 ClinVar review star or UniProt curation and 136 do not), *benign* (21; 2 with ≥1 star), *cancer* (pathogenic variants whose reported diseases are all cancers, assumed mostly gain-of-function; 11), *population* (clinically unclassified variants with hemizygous carriers in gnomAD; 36) and VUS (79). *Tolerated* = benign + population (57). Raw API responses are stored with SHA-256 hashes.

**Structure prediction.** The wild type and every variant of the 251-residue construct were folded with ESMFold v1 (`facebook/esmfold_v1`, 4 recycles; language model ESM-2 3B on CPU in fp32, folding trunk on an NVIDIA RTX 5070 Laptop GPU with TF32). Settings were identical for all structures. The wild type was compared with PDB 2AM9 chain A through SIFTS residue mapping (Cα RMSD, lDDT-Cα).

**Structural metrics.** For each variant relative to the wild type: Δ local pLDDT (mean change within ±5 residues of the substitution, the primary metric), Δ site pLDDT, Δ mean pLDDT, Δ pTM, local lDDT-Cα, and Cα RMSD after superposition. A *breaking point* is a variant whose Δ local pLDDT falls below an empirical noise floor, taken from the same window statistic at residues more than 15 Å from the substitution (n = 67,858 windows; 1st percentile −5.28), at Benjamini–Hochberg FDR < 0.05.

**Baselines.** ESM-2 650M masked-marginal log-likelihood ratio, BLOSUM62, AlphaMissense (17,480 scores from the AlphaFold DB file `AF-P10275-F1-aa-substitutions.csv`), wild-type site pLDDT, and two burial measures from the wild-type structure: Cα neighbours within 10 Å, and relative solvent accessibility (rSASA).

**Mechanism classes.** Each position was assigned one class from experimental structures: hormone contact (2AM9), coactivator groove (2Q7J, 2Q7I), dimer interface (5JJM assembly 1), else buried core or surface by rSASA.

**Statistics.** AUC (pathogenic as positive) with stratified bootstrap 95% CIs (2,000 resamples); paired AUC differences on the same variants with DeLong 95% CI and a paired permutation test (5,000 permutations; the permutation p is primary because DeLong is liberal with few negatives); one-sided Fisher exact tests for breaking-point rates. Robustness checks (added after a first look at the AR results, so exploratory): (a) repeating the comparison after dropping 0, 10, 20 and 30 residues from each end; (b) five-fold cross-validated AUC, repeated 30 times, of L2-regularised logistic models on feature sets (burial; structure; AlphaMissense; combinations), with features standardised within training folds and gains tested on pooled out-of-fold predictions. Complete cases only (294 pathogenic, 55 tolerated). The random seed is fixed in the configuration. [CODE-AVAILABILITY: repository URL and commit hash.]

## 3. Results

**3.1 The wild-type prediction is accurate.** Mean pLDDT was 90.0 (pTM 0.913); 70% of residues scored above 90. Against 2AM9 (249 aligned residues, 99% coverage) Cα RMSD was 0.90 Å and lDDT-Cα 0.932. Per-residue pLDDT correlated modestly with per-residue lDDT (Spearman ρ = 0.36).

**3.2 Pathogenic variants are more often breaking points.** 96 variants at 58 positions were breaking points: 68 of 294 pathogenic (23%), 1 of 21 benign (5%), 1 of 36 carrier variants (3%), 25 of 79 VUS (32%), 1 of 11 cancer-context. Pathogenic versus benign: odds ratio 6.0, one-sided Fisher p = 0.034; pathogenic versus tolerated (2 of 57): odds ratio 8.3, p = 1.5 × 10⁻⁴. The cancer-context negative control was inconclusive (odds ratio 3.0, p = 0.25; 11 variants).

**3.3 Discrimination and baselines (Figure 3).**

| Metric | AUC vs benign (294/21) | AUC vs tolerated (294/57) |
|---|---|---|
| Δ local pLDDT | 0.724 [0.620–0.824] | 0.700 [0.627–0.766] |
| Δ mean pLDDT | 0.769 [0.676–0.854] | 0.724 [0.662–0.785] |
| Cα RMSD | 0.718 [0.593–0.833] | 0.759 [0.686–0.828] |
| Burial (Cα within 10 Å) | 0.757 | 0.773 |
| ESM-2 LLR | 0.614 [0.529–0.700] | 0.682 [0.615–0.746] |
| BLOSUM62 | 0.724 [0.610–0.831] | 0.642 [0.562–0.721] |
| AlphaMissense | 0.777 [0.682–0.861] | 0.821 [0.760–0.874] |

Against benign variants no structural metric differed significantly from AlphaMissense (e.g. Δ local pLDDT ΔAUC −0.052, p = 0.49) or from BLOSUM62 (+0.001, p = 0.99). Δ mean pLDDT and Δ pTM exceeded ESM-2 (ΔAUC +0.156, p = 0.011; +0.157, p = 0.009), but not against the larger tolerated set (+0.042, p = 0.36). Against tolerated variants AlphaMissense outperformed Δ local pLDDT (ΔAUC −0.122 [−0.214, −0.030], p = 0.011) and Δ mean pLDDT (−0.097, p = 0.022), but not Cα RMSD (−0.063, p = 0.17). With only 2 benign variants carrying ≥1 review star, the high-confidence-label analysis is not informative.

**3.4 The signal is not an artefact of the construct ends (Figure 8a).** 43 of 96 breaking points fell in the outer 25 residues of each end, which hold 94 of 441 variants; 53% of pathogenic variants there were breaking points versus 17% in the middle. The class difference persisted: after dropping 30 residues from each end, AUC against tolerated variants was 0.72 for both Δ local and Δ mean pLDDT (untrimmed 0.70 and 0.72), and the pathogenic-versus-tolerated breaking-point difference remained (33 of 230 versus 0 of 42; p = 0.003). Against benign variants alone the breaking-point test lost significance after trimming (p = 0.11, 0.058, 0.14 at 10, 20, 30 residues), but only 13–17 benign variants remain, so the test cannot distinguish a lost effect from lost power. The headline breaking-point rate (23%) is inflated by the ends; it is 14% after trimming 30 residues.

**3.5 Burial explains most of the structural signal (Figure 8b).**

| Feature set (294 pathogenic, 55 tolerated) | Cross-validated AUC |
|---|---|
| AlphaMissense + burial | 0.857 |
| AlphaMissense + burial + structure | 0.844 |
| AlphaMissense | 0.803 |
| Burial + structure | 0.790 |
| Burial | 0.771 |
| Structure (Cα RMSD) | 0.753 |
| Structure (all three metrics) | 0.738 |
| Structure (Δ mean pLDDT) | 0.721 |
| Structure (Δ local pLDDT) | 0.701 |

Adding structure to burial gained +0.021 AUC (95% CI −0.021 to +0.063; permutation p = 0.37). Adding structure to AlphaMissense plus burial changed AUC by −0.015 (−0.044 to +0.015; p = 0.43). Adding burial to AlphaMissense gained +0.059 (+0.003 to +0.115; p = 0.046; one of four gains examined, p-values not adjusted). Structural change was not purely burial: within strata split at the median rSASA (0.14), Δ local pLDDT still separated classes (AUC 0.68 buried, 0.70 exposed), though only 10 tolerated variants are buried.

**3.6 Mechanism (Figure 7).** Among pathogenic variants, breaking-point rates were 25% at buried-core sites (38 of 151), 29% at surface sites (24 of 84), 12% at ligand-contact sites (4 of 32), 11% at dimer-interface sites (2 of 18) and 0% at coactivator-groove sites (0 of 9). Core versus the three interaction classes pooled: 38 of 151 versus 6 of 59, odds ratio 3.0, one-sided Fisher p = 0.011; median Δ local pLDDT −1.98 versus −0.82 (one-sided Mann–Whitney p = 0.013). The omnibus test across the five classes was not significant (Kruskal–Wallis p = 0.17). Per-class AUCs are not interpretable: tolerated variants numbered 2 (ligand), 3 (dimer), 6 (coactivator), 9 (core) and 35 (surface).

## 4. Discussion

In this domain, ESMFold confidence change is a real but modest correlate of pathogenicity: it survives trimming the construct ends and separates pathogenic from tolerated variants at AUC 0.70–0.76. But it adds little to what is available without refolding. Burial from a single wild-type structure matches or exceeds it, adding structure to burial improved AUC by only 0.021 (the CI does not exclude a gain of up to 0.063), and AlphaMissense is stronger and best used together with burial. Practically, for the AR LBD, folding every variant (about 12 s each on a laptop GPU after setup) does not appear to be worth the compute compared with computing burial once.

The mechanism analysis is suggestive, not conclusive. Core variants lose more confidence than interaction-site variants, as expected for a predictor that sees one chain without hormone or partners, but surface variants had the highest breaking-point rate (29%), which a pure fold-stability explanation does not predict and which may reflect residual end effects or variants at flexible loops. With 2–9 tolerated variants in most classes, we cannot say whether interaction-site variants escape detection; the data are compatible with that.

**Limitations.** One protein, one folding model and one fixed setting. Only 21 benign variants (2 with ≥1 review star), so benign-based comparisons have wide intervals; the larger tolerated set includes gnomAD carrier variants that are clinically unclassified and may include mildly affected people. 136 of the 294 pathogenic labels come from literature curation without ClinVar review stars. Labels in ClinVar are partly assigned using computational predictors, which can inflate agreement with them. AlphaMissense itself uses structural context and population frequency, so it is not independent of the information tested here. The robustness checks were chosen after seeing the data and are exploratory; p-values are not corrected across them. Burial and mechanism classes come from ligand- and peptide-bound crystal structures. The construct truncates the full-length protein, and the dimer class is taken from a single crystal assembly. Deterministic prediction means variants have no replicate runs; the empirical noise floor is a proxy.

**What would strengthen this.** Repeating the analysis across other nuclear receptors would give many more benign variants and test whether the pattern generalises, which is the main route to a stronger paper. Deep mutational scanning data, where available, would give quantitative labels without ClinVar circularity.

## 5. Data and code availability

[Repository URL, commit hash, DOI for the data bundle: to be completed after the repository is published. `analysis/stats.json` and `analysis/report.md` contain every number above. Raw API snapshots are stored with SHA-256 hashes.]

## 6. Use of AI tools

[YOU MUST CONFIRM AND EDIT THIS TO BE TRUE.] The analysis code, the statistical analyses and a first draft of this manuscript were produced with the assistance of Claude (Anthropic) in Claude Code, working from the authors' project design. The authors reviewed the code, the outputs and the text, verified the numbers against the pipeline's output files, and take full responsibility for the content. AI tools are not listed as authors. [bioRxiv's FAQ says generative-AI use "should be detailed in manuscripts" and that AI tools "should not be listed as authors"; check the target journal's own rule as well.]

## References (see status key)

Status key: **[checked]** = a search on 2026-11-08 confirmed the title, journal and year; **[from memory]** = not checked, look up before submitting.

1. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379(6637):1123–1130, 2023. doi:10.1126/science.ade2574. [checked; this paper covers both ESMFold and ESM-2]
2. Cheng J, Novati G, Pan J, et al. Accurate proteome-wide missense variant effect prediction with AlphaMissense. *Science* 381(6664):eadg7492, 2023. [checked; sources disagree on one author's name and on the date (20 vs 22 Sep 2023), so copy the final author list from the journal page]
3. Buel GR, Walters KJ. Can AlphaFold2 predict the impact of missense mutations on structure? *Nat Struct Mol Biol*, online 19 Jan 2022. doi:10.1038/s41594-021-00714-2. [checked; volume and pages not confirmed]
4. McBride JM, Polev K, Abdirasulov A, Reinharz V, Grzybowski BA, Tlusty T. AlphaFold2 can predict single-mutation effects. *Phys Rev Lett* 131:218401, 2023. doi:10.1103/PhysRevLett.131.218401. [checked; **the title in the first draft was wrong**, it was the 2022 preprint's wording]
5. Livesey BJ, Marsh JA. The properties of human disease mutations at protein interfaces. *PLoS Comput Biol* 18(2):e1009858, 2022. doi:10.1371/journal.pcbi.1009858. [checked]
6. Schmidt A, Röner S, Mai K, Klinkhammer H, Kircher M, Ludwig KU. Predicting the pathogenicity of missense variants using features derived from AlphaFold2. *Bioinformatics* 39(5):btad280, 2023. doi:10.1093/bioinformatics/btad280. [checked; issue number from memory]
7. Cagiada M, Jonsson N, Lindorff-Larsen K, et al. Decoding molecular mechanisms for loss-of-function variants in the human proteome. *eLife* reviewed preprint 108160 (first posted to bioRxiv 2024). https://elifesciences.org/reviewed-preprints/108160 [checked that it exists; the author list and whether a final version has appeared were NOT confirmed]
8. Henikoff S, Henikoff JG. Amino acid substitution matrices from protein blocks. *Proc Natl Acad Sci USA* 89(22):10915–10919, 1992. [from memory]
9. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* 44(3):837–845, 1988. [from memory]
10. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc B* 57(1):289–300, 1995. [from memory]
11. [CITE, not yet looked up] ClinVar; UniProt; EBI Proteins API; gnomAD v4; PDB entries 2AM9, 2Q7J, 2Q7I, 5JJM; SIFTS.
