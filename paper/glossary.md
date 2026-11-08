# Plain-language glossary

For you, not for the paper. Each entry: what it means, then why it matters here.

**Protein** — a chain of amino acids that folds into a 3D shape. The shape decides what it does.
**Androgen receptor (AR)** — the protein that responds to testosterone. Its hormone-binding end is the "ligand-binding domain" (LBD), amino acids 669–919.
**Missense variant** — a one-letter change in the gene that swaps one amino acid for another. "R780W" means amino acid 780 changed from R (arginine) to W (tryptophan).
**Pathogenic / benign / VUS** — doctors' labels for variants: disease-causing, harmless, or "uncertain significance" (not enough evidence yet).
**ClinVar** — the public database where labs and doctors submit these labels. **Review stars** show how well-checked a label is (0–4). Most of our harmless labels have none.
**gnomAD** — a database of variants seen in healthy-looking people. We use it for extra "probably tolerated" examples, but those people could be mildly affected, so they are a noisier harmless set.
**ESMFold** — an AI that predicts a protein's 3D shape from its amino-acid letters alone. Meta built it.
**pLDDT** — ESMFold's confidence for each amino acid, 0–100. Above 90 = very sure.
**Δ local pLDDT** — how much that confidence falls around the changed amino acid, mutant minus normal. Our main measurement.
**Breaking point** — a variant whose confidence drop is bigger than random wobble would explain. We measure the wobble at amino acids far from the change, where it should have no effect.
**RMSD** — how far apart two structures are, in ångströms (Å, a tenth of a billionth of a metre). Our prediction vs the real crystal: 0.90 Å, which is very close.
**Crystal structure (PDB 2AM9)** — a real, experimentally measured shape of the protein, used to check the AI.
**AUC** — a score for how well a measurement separates two groups. 0.5 = coin flip, 1.0 = perfect. 0.7 = fair, 0.8 = good.
**Confidence interval (CI)** — the range the true value plausibly lies in. Wide = unsure. "[0.62–0.82]" means somewhere from 0.62 to 0.82.
**p-value** — the chance of seeing a result this strong by luck alone if nothing were really going on. Under 0.05 is the usual bar, but a small p does not mean a big effect.
**Cross-validation** — testing a model on data it was not trained on, repeated in rotation, so the score is not flattering.
**Burial (rSASA)** — how hidden an amino acid is inside the protein. Buried ones usually hold the shape together, so changing them tends to hurt.
**AlphaMissense** — Google DeepMind's AI that scores how harmful each possible variant is. The tool to beat.
**ESM-2 LLR** — a language-model score: how "surprising" the change looks given millions of natural proteins.
**BLOSUM62** — a lookup table of how similar two amino acids are. A very simple baseline.
**Baseline** — a simple or existing method your new idea must beat to be worth using.
**Mechanism classes** — what the changed amino acid does: touches the hormone (ligand), touches a partner protein (coactivator), joins two copies (dimer), is buried in the core, or sits on the surface.
**Preprint** — a paper posted publicly before peer review (bioRxiv). It gets you a citable record fast and needs no co-author.
**Exploratory vs confirmatory** — if you pick a test after seeing the data, it is exploratory. Say so; do not claim it was planned.
