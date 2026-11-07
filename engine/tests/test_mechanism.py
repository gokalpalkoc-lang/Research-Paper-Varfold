"""Mechanism map: atom parsing, contacts, solvent accessibility, class precedence."""

import numpy as np
import pandas as pd
import pytest

from varfold.mechanism import atom_sasa, classify, contact_residues, protein, read_atoms, relative_sasa


def _atom(record, serial, name, resname, chain, resnum, xyz, element):
    return (f"{record:<6}{serial:5d} {name:<4} {resname:>3} {chain}{resnum:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00          {element:>2}")


TOY = "\n".join([
    _atom("ATOM", 1, "CA", "LEU", "A", 10, (0, 0, 0), "C"),
    _atom("ATOM", 2, "CA", "ALA", "A", 11, (20, 0, 0), "C"),
    _atom("ATOM", 3, "H", "ALA", "A", 11, (20, 1, 0), "H"),       # hydrogens are dropped
    _atom("HETATM", 4, "C1", "TES", "A", 900, (3.5, 0, 0), "C"),  # ligand next to residue 10
    _atom("HETATM", 5, "O", "HOH", "A", 901, (20, 3, 0), "O"),    # water is dropped
    _atom("ATOM", 6, "CA", "GLY", "B", 1, (20, 3.8, 0), "C"),     # partner chain next to residue 11
    "END",
])


def test_read_atoms_and_contacts():
    atoms = read_atoms(TOY)
    assert len(atoms.xyz) == 4 and "H" not in set(atoms.element) and "HOH" not in set(atoms.resname)
    chain_a = protein(atoms, "A")
    ligand = atoms.select(atoms.hetero & (atoms.resname == "TES"))
    assert contact_residues(chain_a, ligand, 4.0) == {10}
    assert contact_residues(chain_a, protein(atoms, "B"), 4.0) == {11}
    assert contact_residues(chain_a, protein(atoms, "B"), 3.0) == set()


def test_assembly_models_are_read_separately():
    text = "\n".join(["MODEL        1", _atom("ATOM", 1, "CA", "LEU", "A", 10, (0, 0, 0), "C"), "ENDMDL",
                      "MODEL        2", _atom("ATOM", 1, "CA", "LEU", "A", 10, (3, 0, 0), "C"), "ENDMDL", "END"])
    one, two = read_atoms(text, model=1), read_atoms(text, model=2)
    assert one.xyz[0, 0] == 0 and two.xyz[0, 0] == 3
    assert read_atoms(text).xyz.shape == (1, 3)  # default: first model only
    assert contact_residues(protein(one, "A"), protein(two, "A"), 4.0) == {10}


def test_sasa_of_isolated_and_touching_atoms():
    atoms = read_atoms(_atom("ATOM", 1, "CA", "GLY", "A", 1, (0, 0, 0), "C"))
    assert atom_sasa(atoms)[0] == pytest.approx(4 * np.pi * (1.7 + 1.4) ** 2, rel=1e-6)
    pair = read_atoms("\n".join([_atom("ATOM", 1, "CA", "GLY", "A", 1, (0, 0, 0), "C"),
                                 _atom("ATOM", 2, "CA", "GLY", "A", 2, (3, 0, 0), "C")]))
    area = atom_sasa(pair)
    assert np.all(area < 4 * np.pi * 3.1 ** 2) and area[0] == pytest.approx(area[1], rel=0.02)
    assert set(relative_sasa(pair)) == {1, 2}


def test_classify_precedence():
    sets = {"ligand": {5}, "coactivator": {5, 6}, "dimer": {7}}
    rsasa = {5: 0.0, 6: 0.0, 7: 0.0, 8: 0.05, 9: 0.6}
    got = {p: classify(p, sets, rsasa, 0.2) for p in range(5, 11)}
    assert got == {5: "ligand", 6: "coactivator", 7: "dimer", 8: "core", 9: "surface", 10: "unresolved"}


def test_mechanism_summary_and_figure(synthetic_run):
    from varfold.analyze import mechanism_summary, run_analysis
    from varfold import figures

    cfg = synthetic_run
    run_analysis(cfg)
    metrics = pd.read_csv(cfg.paths.analysis / "metrics.csv")
    # Planted signal is on pathogenic variants; put half of them at "interaction sites".
    path = metrics.index[metrics["label"] == "pathogenic"]
    metrics["mechanism"] = "surface"
    metrics.loc[path[:8], "mechanism"] = "core"
    metrics.loc[path[8:], "mechanism"] = "ligand"
    metrics.loc[metrics["label"] == "benign", "mechanism"] = ["core"] * 5 + ["surface"] * 5
    summary = mechanism_summary(metrics, n_boot=50, seed=1)
    rows = {r["mechanism"]: r for r in summary["classes"]}
    assert rows["core"]["n_pathogenic"] == 8 and rows["core"]["n_tolerated"] == 5
    assert rows["ligand"]["n_tolerated"] == 0 and rows["ligand"]["auc_delta_local_plddt"] is None
    assert rows["core"]["auc_delta_local_plddt"] > 0.9
    tests = summary["tests"]
    assert sum(tests["breaking_core_vs_interaction"]["table"][0]) == 8
    assert tests["delta_local_core_vs_interaction"]["n_interaction"] == 8
    assert "kruskal_delta_local_pathogenic" in tests
    assert figures.delta_by_mechanism(metrics, -5.0, cfg.paths.figures, 1) == [
        "fig7_delta_by_mechanism.pdf", "fig7_delta_by_mechanism.png"]
