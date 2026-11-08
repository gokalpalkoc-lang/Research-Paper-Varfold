# Analysis plan and status (written 2026-11-08)

## Main question
Does the change in ESMFold confidence between a mutant and the wild type tell harmful AR ligand-binding-domain variants from tolerated ones, and does it add anything beyond burial and AlphaMissense?

## Honest status of each analysis
| Analysis | Status | Where |
|---|---|---|
| Wild-type prediction vs crystal structure | Planned in the original design | report.md, "Wild-type prediction" |
| Pathogenic vs benign/tolerated AUCs, breaking-point enrichment | Planned in the original design | report.md |
| Comparison with ESM-2, BLOSUM62, AlphaMissense, site pLDDT | Planned in the original design | report.md |
| Mechanism classes (core vs interaction sites) | Specified in code before the full run, from the README's prediction that interaction sites escape ESMFold | report.md, "By mechanism" |
| **Edge-trim check** | **Added after a first look at the data. Exploratory.** | report.md, "Robustness checks" |
| **Burial head-to-head (cross-validated)** | **Added after a first look at the data. Exploratory.** | report.md, "Robustness checks" |

The two exploratory checks are in code (`engine/varfold/robustness.py`, 13 tests), so any other protein is analysed with the same choices. p-values are not corrected for multiple testing across them. Say this in the paper (it is in the draft).

## Pre-committed rules, to avoid fooling ourselves
1. The primary claim is the one in the manuscript abstract, whatever the numbers show. We do not swap in a better-looking metric afterwards.
2. All numbers in the paper come from `stats.json`. Nothing is typed from memory.
3. If we add genes, the same code runs unchanged and the combined result is reported even if it is negative.

## Figure map (file → paper figure)
| File | Paper figure |
|---|---|
| fig1_wt_plddt_profile | Fig 1: wild-type confidence profile |
| fig2_delta_plddt_by_class | Fig 2: confidence change by clinical class |
| fig3_roc | Fig 3: ROC curves vs baselines |
| fig4_position_profile | Supplement: per-position profile |
| fig5_structure_vs_sequence | Supplement: structure vs sequence scores |
| fig7_delta_by_mechanism | Fig 4: by mechanism |
| fig8_robustness | Fig 5: edge trim and burial head-to-head |

The text currently says "Figure 3", "Figure 7" and "Figure 8" using file numbers. Renumber when the figures are final.

## Still to do before submission
- [ ] Verify every reference (the list is marked); add the missing ones.
- [ ] Push the repository, add its URL and commit hash to the data-availability section.
- [ ] Check the pathogenic labels: 136 of 294 have no ClinVar review stars. Consider a sensitivity analysis restricted to the 158 well-supported ones (only 2 benign remain, so it has little power).
- [ ] Decide whether to add more nuclear receptors (the main route to a stronger paper).
- [ ] Read the target venue's rules on disclosing AI-assisted writing and analysis. Many journals and bioRxiv expect it to be stated.
- [ ] Have at least one person with variant-interpretation experience read the draft if you can find one.
