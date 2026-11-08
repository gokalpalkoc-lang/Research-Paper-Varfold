"""Edge-trim and burial checks: planted signals are recovered, noise is not, bookkeeping is right."""

import numpy as np
import pandas as pd
import pytest

from varfold import robustness as rb
from varfold.analyze import render_robustness
from varfold.stats import auc

START, END = 100, 349


def make_metrics(n_path=120, n_tol=60, seed=0, burial_signal=2.0, structure_signal=0.0, edge_only=False):
    """Pathogenic variants sit at more buried sites; structure carries ``structure_signal`` on top."""
    rng = np.random.default_rng(seed)
    n = n_path + n_tol
    path = np.r_[np.ones(n_path, bool), np.zeros(n_tol, bool)]
    pos = rng.integers(START, END + 1, n)
    rsasa = np.clip(rng.normal(0.35 - 0.12 * burial_signal * path / 2, 0.12, n), 0, 1)
    delta_local = rng.normal(0, 3, n) - structure_signal * path - (8 * (pos < START + 10) if edge_only else 0)
    df = pd.DataFrame({
        "position": pos, "label": np.where(path, "pathogenic", np.where(rng.random(n) < 0.5, "benign", "population")),
        "rsasa": rsasa, "delta_local_plddt": delta_local,
        "delta_mean_plddt": delta_local / 3 + rng.normal(0, 0.3, n), "rmsd_ca": rng.gamma(2, 0.3, n),
        "alphamissense": rng.random(n) * 0.5 + 0.4 * path,
    })
    df["breaking_point"] = df["delta_local_plddt"] < -6
    return df


def test_direction_orients_burial_and_metrics():
    assert rb.direction("rsasa") == -1          # smaller rSASA = more buried = more damaging
    assert rb.direction("delta_local_plddt") == -1
    assert rb.direction("rmsd_ca") == +1
    assert rb.direction("alphamissense") == +1


def test_cross_validated_separates_signal_from_noise():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(80, bool), np.zeros(80, bool)]
    signal = np.where(y, 1.5, 0.0) + rng.normal(0, 1, 160)
    noise = rng.normal(0, 1, 160)
    good, _, oof = rb.cross_validated(signal[:, None], y, reps=5, folds=5, seed=1)
    bad, _, _ = rb.cross_validated(noise[:, None], y, reps=5, folds=5, seed=1)
    assert good > 0.8 and 0.35 < bad < 0.65
    assert auc(oof, y) > 0.8 and len(oof) == 160


def test_cross_validated_is_reproducible():
    y = np.r_[np.ones(40, bool), np.zeros(40, bool)]
    X = np.random.default_rng(0).normal(size=(80, 2)) + y[:, None]
    a = rb.cross_validated(X, y, reps=3, folds=5, seed=7)
    b = rb.cross_validated(X, y, reps=3, folds=5, seed=7)
    assert a[0] == b[0] and np.array_equal(a[2], b[2])


def test_edge_trim_drops_edge_variants_and_keeps_signal():
    df = make_metrics(structure_signal=4.0)
    rows = rb.edge_trim(df, START, END, rb.TOLERATED, n_boot=50, seed=1)
    assert [r["trim"] for r in rows] == [0, 10, 20, 30]
    sizes = [r["n_variants"] for r in rows]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] == len(df) and sizes[-1] < sizes[0]
    kept = df[(df.position >= START + 30) & (df.position <= END - 30)]
    assert rows[-1]["n_variants"] == len(kept)
    assert all(r["auc"]["delta_local_plddt"]["auc"] > 0.55 for r in rows)


def test_edge_trim_exposes_an_edge_artefact():
    """When only variants near the start look damaged, trimming removes the apparent signal."""
    df = make_metrics(structure_signal=0.0, edge_only=True, seed=5)
    # Make the artefact class-blind-but-uneven: pathogenic variants more often at the edge.
    rng = np.random.default_rng(5)
    path = df.label == "pathogenic"
    df.loc[path & (rng.random(len(df)) < 0.3), "position"] = START + 2
    df["delta_local_plddt"] -= 8 * (df.position < START + 10)
    full, trimmed = rb.edge_trim(df, START, END, rb.TOLERATED, n_boot=50, seed=1, trims=(0, 20))
    assert full["auc"]["delta_local_plddt"]["auc"] > trimmed["auc"]["delta_local_plddt"]["auc"] + 0.05


def test_edge_share_counts():
    df = make_metrics()
    es = rb.edge_share(df, START, END, width=25)
    edge = (df.position < START + 25) | (df.position > END - 25)
    assert es["variants_in_edges"] == int(edge.sum()) and es["variants"] == len(df)
    assert es["breaking_points_in_edges"] == int((df.breaking_point & edge).sum())
    assert set(es["by_label"]) == set(df.label)


def test_burial_beats_structure_when_only_burial_is_informative():
    df = make_metrics(burial_signal=3.0, structure_signal=0.0)
    out = rb.feature_sets(df, rb.TOLERATED, reps=5, seed=1, n_perm=200)
    cv = {s["name"]: s["cv_auc"] for s in out["sets"]}
    assert cv["burial"] > cv["structure (Δ local pLDDT)"] + 0.1
    gain = next(g for g in out["gains"] if g["new"] == "burial + structure" and g["base"] == "burial")
    assert abs(gain["difference"]) < 0.1 and gain["p"] > 0.05   # structure adds nothing here
    assert out["n_pathogenic"] + out["n_negative"] == len(df)


def test_structure_adds_when_it_carries_independent_signal():
    df = make_metrics(burial_signal=1.0, structure_signal=6.0, seed=2)
    out = rb.feature_sets(df, rb.TOLERATED, reps=5, seed=1, n_perm=200)
    gain = next(g for g in out["gains"] if g["new"] == "burial + structure" and g["base"] == "burial")
    assert gain["difference"] > 0.05


def test_missing_values_use_complete_cases_and_missing_columns_are_skipped():
    df = make_metrics()
    df.loc[df.index[:10], "rmsd_ca"] = np.nan
    out = rb.feature_sets(df, rb.TOLERATED, reps=2, seed=1, n_perm=50)
    assert out["n_pathogenic"] + out["n_negative"] == len(df) - 10
    no_am = rb.feature_sets(df.drop(columns="alphamissense"), rb.TOLERATED, reps=2, seed=1, n_perm=50)
    assert "AlphaMissense" not in {s["name"] for s in no_am["sets"]}


def test_too_few_variants_returns_none():
    df = make_metrics(n_path=6, n_tol=4)
    assert rb.feature_sets(df, rb.TOLERATED, reps=2, seed=1, n_perm=50) is None


def test_run_needs_burial():
    df = make_metrics().drop(columns="rsasa")
    assert rb.run(df, START, END, n_boot=20, n_perm=50, seed=1) is None


def test_run_report_and_figure(tmp_path):
    from varfold.figures import robustness_figure

    df = make_metrics(seed=4)
    out = rb.run(df, START, END, n_boot=30, n_perm=100, seed=1, reps=3)
    assert out["edge_trim_tolerated"] and out["feature_sets"]["strata"]["groups"]
    text = "\n".join(render_robustness(out))
    assert "Robustness checks" in text and "Burial head-to-head" in text and "burial + structure" in text
    files = robustness_figure(out, tmp_path)
    assert sorted(files) == ["fig8_robustness.pdf", "fig8_robustness.png"]
    assert all((tmp_path / f).stat().st_size > 0 for f in files)


def test_run_without_population_uses_benign_only():
    df = make_metrics(seed=6)
    df["label"] = df["label"].replace("population", "benign")
    out = rb.run(df, START, END, n_boot=20, n_perm=50, seed=1, reps=2)
    assert out["edge_trim_tolerated"] == [] and out["feature_sets"]["negatives"] == ["benign"]
