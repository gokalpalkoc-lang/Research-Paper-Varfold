"""ESMFold inference (Hugging Face ``EsmForProteinFolding``) and the resumable batch runner."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import FoldingOptions, TargetConfig
from .sequences import sequence_hash

log = logging.getLogger(__name__)

CA_INDEX = 1  # atom37 ordering: N, CA, C, ...


@dataclass
class Prediction:
    sequence: str
    pdb: str
    plddt: np.ndarray          # per residue (CA), 0-100
    mean_plddt: float          # mean of per-residue pLDDT
    mean_plddt_atoms: float    # ESMFold's own atom-averaged pLDDT
    ptm: float
    pae: np.ndarray | None     # (L, L) predicted aligned error, Angstrom
    runtime_s: float
    chunk_size: int | None


def _publishes_safetensors(name: str) -> bool:
    if Path(name).is_dir():
        return any(Path(name).glob("*.safetensors"))
    from huggingface_hub import file_exists, try_to_load_from_cache

    for fname in ("model.safetensors", "model.safetensors.index.json"):
        if isinstance(try_to_load_from_cache(name, fname), str):
            return True
    try:
        return file_exists(name, "model.safetensors") or file_exists(name, "model.safetensors.index.json")
    except Exception:  # offline: fall back to whatever the cache/hub resolves
        return False


def load_pretrained(cls, name: str):
    """Load the weights the repository's main branch actually publishes.

    ``facebook/esmfold_v1`` only publishes ``pytorch_model.bin``. With the default
    ``use_safetensors=None`` transformers then starts a background download of an
    8 GB safetensors conversion from an unmerged ``refs/pr/*`` branch — wasteful,
    and not the reviewed artifact a paper should cite. So be explicit: safetensors
    when main has them (e.g. ESM-2), otherwise the .bin (loaded with ``weights_only``).
    """
    return cls.from_pretrained(name, use_safetensors=_publishes_safetensors(name))


def configure_cpu_threads() -> None:
    """Honour VARFOLD_THREADS for CPU inference.

    PyTorch defaults to one thread per physical core. Using every logical CPU is
    ~2x slower on hybrid CPUs (measured on a Core Ultra 9 185H, 128 aa: 22 threads
    22 s, 16 threads 12 s, 6 threads 9.6 s), so only override on request.
    """
    import torch

    if threads := os.environ.get("VARFOLD_THREADS"):
        torch.set_num_threads(int(threads))


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (is the NVIDIA driver installed? see README)")
    return torch.device(requested)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _set_bfactors(pdb: str, values) -> str:
    """Overwrite the B-factor column (cols 61-66) of ATOM records, in file order."""
    lines = pdb.splitlines()
    atoms = [k for k, line in enumerate(lines) if line.startswith("ATOM") and len(line) >= 66]
    if len(atoms) != len(values):
        raise RuntimeError(f"PDB has {len(atoms)} atoms but {len(values)} pLDDT values")
    for k, v in zip(atoms, values):
        lines[k] = f"{lines[k][:60]}{float(v):6.2f}{lines[k][66:]}"
    return "\n".join(lines) + "\n"


class ESMFoldPredictor:
    """Loads ESMFold once and folds single sequences.

    On CUDA with ``precision: fp16`` the 3B-parameter ESM-2 language model runs in
    half precision (required to fit 8 GB cards) while the folding trunk stays fp32
    (TF32 matmuls). On CPU everything runs in fp32. The effective precision is part
    of :attr:`fingerprint`, so predictions made under different numerics are never mixed.
    """

    def __init__(self, options: FoldingOptions, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.options = options
        self.device = resolve_device(device or options.device)
        if self.device.type == "cuda":
            # TF32 matmuls are enabled on CUDA, so CUDA "fp32" differs numerically from CPU fp32.
            self.precision = "fp16-lm/tf32" if options.precision == "fp16" else "fp32/tf32"
        else:
            self.precision = "fp32"
        self.chunk_size = options.chunk_size
        self._torch = torch
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(options.model)
        model = load_pretrained(EsmForProteinFolding, options.model)
        model.eval()
        if self.precision.startswith("fp16-lm"):
            model.esm = model.esm.half()
        self.model = model.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            configure_cpu_threads()
        self._set_chunk(self.chunk_size)
        log.info("ESMFold loaded on %s (%s) in %.0fs", self.device_name, self.precision, time.perf_counter() - t0)

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return self._torch.cuda.get_device_name(self.device)
        return f"cpu ({_cpu_model()})"

    @property
    def fingerprint(self) -> str:
        return f"{self.options.model}|recycles={self.options.num_recycles}|{self.precision}"

    def _set_chunk(self, size: int | None) -> None:
        self.model.trunk.set_chunk_size(size)
        self.chunk_size = size

    def _forward(self, sequence: str):
        torch = self._torch
        inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] != len(sequence):
            raise RuntimeError("unexpected ESM tokenisation: expected one token per residue")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        torch.manual_seed(0)
        with torch.inference_mode():
            return self.model(**inputs, num_recycles=self.options.num_recycles)

    def predict(self, sequence: str) -> Prediction:
        torch = self._torch
        t0 = time.perf_counter()
        while True:
            try:
                out = self._forward(sequence)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                current = self.chunk_size or 128
                if current <= 4:
                    raise
                log.warning("CUDA OOM at chunk_size=%s for L=%d; retrying with %d",
                            self.chunk_size, len(sequence), current // 2)
                self._set_chunk(current // 2)
        runtime = time.perf_counter() - t0

        plddt_atoms = out["plddt"][0].float().cpu().numpy()            # (L, 37)
        atom_mask = out["atom37_atom_exists"][0].float().cpu().numpy()  # (L, 37)
        # transformers reports pLDDT on a 0-1 scale (the original ESM code used 0-100).
        scale = 100.0 if plddt_atoms.max() <= 1.0 + 1e-6 else 1.0
        plddt_atoms = plddt_atoms * scale
        per_residue = plddt_atoms[:, CA_INDEX]
        mean_atoms = float((plddt_atoms * atom_mask).sum() / atom_mask.sum())
        # output_to_pdb rounds B-factors to 2 decimals on the 0-1 scale (i.e. whole pLDDT
        # units), so write the full-precision values ourselves, in its atom37 order.
        pdb = _set_bfactors(self.model.output_to_pdb(out)[0], plddt_atoms[atom_mask > 0.5])
        pae = None
        if self.options.save_pae and "predicted_aligned_error" in out:
            pae = out["predicted_aligned_error"][0].float().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return Prediction(
            sequence=sequence, pdb=pdb, plddt=per_residue, mean_plddt=float(per_residue.mean()),
            mean_plddt_atoms=mean_atoms, ptm=float(out["ptm"].float().item()), pae=pae,
            runtime_s=runtime, chunk_size=self.chunk_size,
        )


# --------------------------------------------------------------------------- batch


def read_predictions(path: Path, fingerprint: str | None = None) -> dict[str, dict]:
    """Latest prediction record per variant (optionally only for one fingerprint)."""
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with path.open() as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # e.g. a write torn by power loss; that variant is refolded
                log.warning("%s:%d is not valid JSON; ignoring it", path.name, n)
                continue
            if fingerprint is None or rec.get("fingerprint") == fingerprint:
                latest[rec["variant_id"]] = rec
    return latest


def fingerprint_dir(fingerprint: str) -> str:
    """Short stable directory name for one settings fingerprint."""
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]


def structure_file(paths, record: dict) -> Path:
    """The PDB belonging to a prediction record.

    Files live under ``structures/<fingerprint-dir>/`` so predictions made under
    different settings can never overwrite or be paired with each other. Records
    written before that layout (no ``structure`` key) point at ``structures/<id>.pdb``.
    """
    return paths.structures / record.get("structure", f"{record['variant_id']}.pdb")


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _versions() -> dict:
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version()}


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def run_batch(cfg: TargetConfig, ids: list[str] | None = None, limit: int | None = None,
              device: str | None = None, force: bool = False) -> dict:
    """Fold every dataset sequence that has no prediction under the current settings.

    Results are appended (and fsync'ed) one variant at a time, so the run can be
    interrupted at any point and resumed by invoking it again.
    """
    from .curate import load_dataset

    paths = cfg.paths
    df = load_dataset(cfg)
    if ids:
        missing = set(ids) - set(df["variant_id"])
        if missing:
            raise ValueError(f"not in dataset: {sorted(missing)}")
        df = df[df["variant_id"].isin(set(ids) | {"WT"})]

    predictor = ESMFoldPredictor(cfg.folding, device=device)
    fp = predictor.fingerprint
    done = {} if force else read_predictions(paths.predictions, fp)
    todo = [r for r in df.itertuples(index=False)
            if not (r.variant_id in done and done[r.variant_id]["sequence_hash"] == sequence_hash(r.sequence)
                    and structure_file(paths, done[r.variant_id]).exists())]
    # The wild type anchors every comparison, so it always goes first.
    todo.sort(key=lambda r: r.variant_id != "WT")
    n_done, n_todo_total = len(df) - len(todo), len(todo)
    if limit is not None:
        todo = todo[:limit]

    fp_dir = fingerprint_dir(fp)
    (paths.structures / fp_dir).mkdir(parents=True, exist_ok=True)
    if cfg.folding.save_pae:
        (paths.pae / fp_dir).mkdir(parents=True, exist_ok=True)
    versions = _versions()
    log.info("%d to fold this run (%d of %d already done under %s) on %s",
             len(todo), n_done, len(df), fp, predictor.device_name)

    stop = {"flag": False}

    def _graceful(signum, frame):  # finish the current structure, then exit
        log.warning("signal %s received; stopping after the current structure", signum)
        stop["flag"] = True

    previous = signal.signal(signal.SIGTERM, _graceful)
    n_ok = n_fail = 0
    interrupted = False
    t_start = time.perf_counter()
    try:
        for i, row in enumerate(todo, 1):
            if stop["flag"]:
                interrupted = True
                break
            if not _SAFE_ID.match(row.variant_id):
                raise ValueError(f"unsafe variant id {row.variant_id!r}")
            try:
                pred = predictor.predict(row.sequence)
            except Exception as exc:  # keep the overnight run alive; failures are logged
                n_fail += 1
                log.exception("failed %s", row.variant_id)
                _append_jsonl(paths.failures, {"variant_id": row.variant_id, "error": repr(exc),
                                               "fingerprint": fp,
                                               "timestamp": datetime.now(timezone.utc).isoformat()})
                if row.variant_id == "WT":
                    raise
                continue
            structure = f"{fp_dir}/{row.variant_id}.pdb"
            (paths.structures / structure).write_text(pred.pdb)
            pae_file = None
            if pred.pae is not None:
                pae_file = f"{fp_dir}/{row.variant_id}.npz"
                np.savez_compressed(paths.pae / pae_file, pae=pred.pae.astype(np.float16))
            _append_jsonl(paths.predictions, {
                "variant_id": row.variant_id,
                "structure": structure,
                "pae": pae_file,
                "sequence_hash": sequence_hash(row.sequence),
                "length": len(row.sequence),
                "mean_plddt": round(pred.mean_plddt, 3),
                "mean_plddt_atoms": round(pred.mean_plddt_atoms, 3),
                "ptm": round(pred.ptm, 4),
                "plddt": [round(float(x), 2) for x in pred.plddt],
                "runtime_s": round(pred.runtime_s, 2),
                "chunk_size": pred.chunk_size,
                "device": predictor.device_name,
                "fingerprint": fp,
                "model": cfg.folding.model,
                "num_recycles": cfg.folding.num_recycles,
                "precision": predictor.precision,
                "versions": versions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            n_ok += 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (len(todo) - i)
            log.info("[%d/%d] %-8s pLDDT %5.1f  pTM %.3f  %5.1fs  (ETA %s)", i, len(todo), row.variant_id,
                     pred.mean_plddt, pred.ptm, pred.runtime_s, _fmt_duration(eta))
    except KeyboardInterrupt:
        interrupted = True
        log.warning("interrupted — progress is saved; re-run the same command to resume")
    finally:
        signal.signal(signal.SIGTERM, previous)
    return {"folded": n_ok, "failed": n_fail, "remaining": n_todo_total - n_ok - n_fail,
            "interrupted": interrupted, "fingerprint": fp}


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
