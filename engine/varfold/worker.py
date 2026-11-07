"""Live inference worker: folds single sequences on the local GPU for the showcase app.

Bind it to the machine's Tailscale address so only the tailnet (i.e. the Hetzner
server) can reach it, and require the shared bearer token on every request.
"""

from __future__ import annotations

import asyncio
import gzip
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .sequences import SequenceError, clean_sequence, sequence_hash

log = logging.getLogger(__name__)


class Predictor(Protocol):
    fingerprint: str
    device_name: str

    def predict(self, sequence: str): ...


class FoldRequest(BaseModel):
    sequence: str = Field(min_length=1, max_length=5000)


class WorkerState:
    def __init__(self, predictor: Predictor, max_length: int, max_queue: int, cache_dir: Path | None):
        self.predictor = predictor
        self.max_length = max_length
        self.max_queue = max_queue
        self.cache_dir = cache_dir
        self.gpu = asyncio.Lock()
        self.waiting = 0
        self.served = 0
        self.started = time.monotonic()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.json.gz" if self.cache_dir else None

    def cache_get(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if path and path.exists():
            return json.loads(gzip.decompress(path.read_bytes()))
        return None

    def cache_put(self, key: str, value: dict) -> None:
        path = self._cache_path(key)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(json.dumps(value).encode()))
            tmp.replace(path)


def create_app(predictor_factory, token: str, max_length: int = 400, max_queue: int = 3,
               cache_dir: Path | None = None) -> FastAPI:
    if not token or len(token) < 16:
        raise ValueError("a worker token of at least 16 characters is required")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        predictor = await asyncio.to_thread(predictor_factory)
        app.state.worker = WorkerState(predictor, max_length, max_queue, cache_dir)
        log.info("worker ready on %s (%s)", predictor.device_name, predictor.fingerprint)
        yield

    app = FastAPI(title="varfold worker", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authorise(request: Request) -> None:
        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="unauthorised")

    @app.get("/health", dependencies=[Depends(authorise)])
    async def health(request: Request):
        w: WorkerState = request.app.state.worker
        return {"status": "ok", "device": w.predictor.device_name, "fingerprint": w.predictor.fingerprint,
                "busy": w.gpu.locked(), "queue": w.waiting, "max_queue": w.max_queue,
                "max_length": w.max_length, "served": w.served, "uptime_s": round(time.monotonic() - w.started)}

    @app.post("/fold", dependencies=[Depends(authorise)])
    async def fold(body: FoldRequest, request: Request):
        w: WorkerState = request.app.state.worker
        try:
            seq = clean_sequence(body.sequence, max_length=w.max_length)
        except SequenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        key = sequence_hash(f"{w.predictor.fingerprint}|{seq}")
        if (hit := w.cache_get(key)) is not None:
            return {**hit, "cached": True}
        # max_queue = how many requests may wait *behind* the one on the GPU.
        if w.gpu.locked() and w.waiting >= w.max_queue:
            raise HTTPException(status_code=503, detail="worker busy — try again shortly",
                                headers={"Retry-After": "30"})
        w.waiting += 1
        try:
            await w.gpu.acquire()
        finally:
            w.waiting -= 1
        if (hit := w.cache_get(key)) is not None:  # folded while we queued
            w.gpu.release()
            return {**hit, "cached": True}
        # Release the GPU only when the inference thread really finishes: if this request is
        # cancelled (client gone, shutdown) the thread keeps running and must keep the lock.
        job = asyncio.ensure_future(asyncio.to_thread(w.predictor.predict, seq))
        job.add_done_callback(lambda _: w.gpu.release())
        pred = await asyncio.shield(job)
        result = {
            "sequence_hash": sequence_hash(seq), "length": len(seq),
            "pdb": pred.pdb, "plddt": [round(float(x), 2) for x in pred.plddt],
            "mean_plddt": round(pred.mean_plddt, 2), "ptm": round(pred.ptm, 4),
            "runtime_s": round(pred.runtime_s, 2), "device": w.predictor.device_name,
            "fingerprint": w.predictor.fingerprint,
        }
        w.cache_put(key, result)
        w.served += 1
        log.info("folded L=%d in %.1fs (pLDDT %.1f)", len(seq), pred.runtime_s, pred.mean_plddt)
        return {**result, "cached": False}

    return app
