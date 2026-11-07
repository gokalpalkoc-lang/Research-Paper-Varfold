"""Integration tests against the real weights. Slow; run with ``pytest -m model``."""

import numpy as np
import pytest

from varfold.config import FoldingOptions
from varfold.structure import read_ca

pytestmark = pytest.mark.model

# Trp-cage (20 aa): small, well-folded, fast even on CPU.
TRP_CAGE = "NLYIQWLKDGGPSSGRPPPS"


@pytest.fixture(scope="module")
def predictor():
    from varfold.folding import ESMFoldPredictor

    # cpu-lm fits any CUDA card with ~3 GB free and is plain fp32 on CPU-only machines.
    return ESMFoldPredictor(FoldingOptions(num_recycles=1, chunk_size=None, precision="cpu-lm"), device="auto")


def test_prediction_is_on_0_100_scale_and_consistent_with_pdb(predictor):
    pred = predictor.predict(TRP_CAGE)
    assert pred.plddt.shape == (len(TRP_CAGE),)
    assert 1.0 < pred.mean_plddt <= 100.0, "pLDDT must be rescaled to 0-100"
    assert 0.0 <= pred.ptm <= 1.0
    ca = read_ca(pred.pdb)
    assert len(ca.coords) == len(TRP_CAGE)
    # B-factors in the PDB carry the same per-residue pLDDT we report.
    assert np.allclose(ca.bfactor, pred.plddt, atol=0.01)
    assert pred.pae is not None and pred.pae.shape == (len(TRP_CAGE), len(TRP_CAGE))
    # Consecutive Cα atoms of a real backbone are ~3.8 Å apart.
    steps = np.linalg.norm(np.diff(ca.coords, axis=0), axis=1)
    assert np.all((steps > 3.0) & (steps < 4.5))


def test_loader_uses_published_weight_format():
    from varfold.folding import _publishes_safetensors

    assert _publishes_safetensors("facebook/esmfold_v1") is False      # .bin only on main
    assert _publishes_safetensors("facebook/esm2_t33_650M_UR50D") is True
