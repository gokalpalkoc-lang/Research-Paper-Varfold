"""run_batch with a stub predictor: resume logic and isolation between settings."""

import json

import numpy as np
import pandas as pd
import pytest

# run_batch records the torch/transformers versions with every prediction, even with the stub.
pytest.importorskip("torch")

from synthetic import make_pdb, random_chain  # noqa: E402
from varfold import folding  # noqa: E402
from varfold.folding import Prediction, read_predictions, structure_file  # noqa: E402
from varfold.metrics import compute_metrics  # noqa: E402


def stub_predictor(precision: str, shift: float):
    """Deterministic fake ESMFold: geometry depends on the sequence; ``shift`` marks the run."""

    class Stub:
        def __init__(self, options, device=None):
            self.options = options
            self.precision = precision
            self.device_name = f"stub-{precision}"
            self.fingerprint = f"{options.model}|recycles={options.num_recycles}|{precision}"
            self.calls = []

        def predict(self, seq):
            self.calls.append(seq)
            rng = np.random.default_rng(abs(hash(seq)) % 2**32)
            ca = random_chain(len(seq), rng) + shift
            plddt = np.full(len(seq), 80.0 + shift)
            return Prediction(seq, make_pdb(seq, ca, plddt), plddt, float(plddt.mean()), float(plddt.mean()),
                              0.8, None, 0.01, None)

    return Stub


@pytest.fixture
def run_cfg(synthetic_run):
    cfg = synthetic_run
    cfg.paths.predictions.unlink()  # start from an empty run on the synthetic dataset
    cfg.folding.save_pae = False
    return cfg


def test_resume_and_settings_isolation(run_cfg, monkeypatch):
    cfg = run_cfg
    n = len(pd.read_csv(cfg.paths.dataset))

    monkeypatch.setattr(folding, "ESMFoldPredictor", stub_predictor("fp16-lm/tf32", 0.0))
    first = folding.run_batch(cfg)
    assert first == {**first, "folded": n, "remaining": 0, "interrupted": False}
    again = folding.run_batch(cfg)  # resume: nothing left to do
    assert again["folded"] == 0 and again["remaining"] == 0

    gpu = read_predictions(cfg.paths.predictions, first["fingerprint"])
    gpu_files = {v: structure_file(cfg.paths, r).read_text() for v, r in gpu.items()}

    # A partial run under other settings (e.g. CPU fallback) must not touch the first run.
    monkeypatch.setattr(folding, "ESMFoldPredictor", stub_predictor("fp32", 50.0))
    second = folding.run_batch(cfg, limit=5)
    assert second["folded"] == 5 and second["remaining"] == n - 5
    assert {v: structure_file(cfg.paths, r).read_text() for v, r in gpu.items()} == gpu_files

    # Analysis follows the latest WT record's settings and only pairs files from that run.
    metrics, _, info = compute_metrics(cfg)
    assert info["fingerprint"] == second["fingerprint"]
    assert len(metrics) == 4 and info["n_other_settings"] == n - 5
    assert (metrics["rmsd_ca"] > 1.0).all()  # stub geometry differs per sequence; no silent mixing of runs


def test_limit_reports_true_remaining(run_cfg, monkeypatch):
    monkeypatch.setattr(folding, "ESMFoldPredictor", stub_predictor("fp32", 0.0))
    n = len(pd.read_csv(run_cfg.paths.dataset))
    res = folding.run_batch(run_cfg, limit=3)
    assert res["folded"] == 3 and res["remaining"] == n - 3
    recs = [json.loads(l) for l in run_cfg.paths.predictions.read_text().splitlines()]
    assert recs[0]["variant_id"] == "WT" and all("structure" in r for r in recs)
