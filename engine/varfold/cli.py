"""Command line: ``varfold <step> --config configs/ar_lbd.yaml``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import REPO_ROOT, load_config

DEFAULT_CONFIG = REPO_ROOT / "configs" / "ar_lbd.yaml"


def _setup_logging(cfg_root: Path | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if cfg_root is not None:
        cfg_root.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg_root / "varfold.log"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "httpx", "httpx2", "filelock", "huggingface_hub", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_curate(cfg, args):
    from .curate import build_dataset

    df = build_dataset(cfg, refresh=args.refresh)
    print(df["label"].value_counts().to_string())


def cmd_fold(cfg, args) -> dict:
    from .folding import run_batch

    result = run_batch(cfg, ids=args.ids, limit=args.limit, device=args.device, force=args.force)
    print(json.dumps(result))
    return result


def cmd_llr(cfg, args):
    from .llr import compute_llr

    compute_llr(cfg, device=args.device)


def cmd_validate(cfg, args):
    from .validate import validate_wild_type

    res = validate_wild_type(cfg, refresh=args.refresh)
    if res:
        print(json.dumps({k: v for k, v in res.items() if k != "per_residue"}, indent=2))


def cmd_analyze(cfg, args):
    from .analyze import run_analysis

    run_analysis(cfg)
    print((cfg.paths.analysis / "report.md").read_text())


def cmd_export(cfg, args):
    from .export import export_web_db

    print(export_web_db(cfg))


def cmd_status(cfg, args):
    from .curate import load_dataset
    from .folding import read_predictions

    from .sequences import sequence_hash

    df = load_dataset(cfg)
    latest = read_predictions(cfg.paths.predictions)
    fp = latest.get("WT", {}).get("fingerprint")
    recs = read_predictions(cfg.paths.predictions, fp) if fp else {}
    # Only predictions for the dataset's current sequence count as done.
    done = {v for v, s in zip(df["variant_id"], df["sequence"])
            if v in recs and recs[v]["sequence_hash"] == sequence_hash(s)}
    by_label = df.assign(done=df["variant_id"].isin(done)).groupby("label")["done"].agg(["sum", "count"])
    print(f"settings: {fp or '(WT not folded yet)'}")
    print(by_label.rename(columns={"sum": "folded", "count": "total"}).to_string())
    # Runs timed with the wall clock before it was replaced can carry negative runtimes.
    runtimes = [recs[v]["runtime_s"] for v in done if recs[v]["runtime_s"] > 0]
    if runtimes:
        mean = sum(runtimes) / len(runtimes)
        remaining = len(df) - len(done)
        print(f"mean {mean:.1f}s/structure → ~{remaining * mean / 3600:.1f} h for the remaining {remaining}")


def cmd_all(cfg, args):
    cmd_curate(cfg, args)
    result = cmd_fold(cfg, args)
    if result["interrupted"] or result["remaining"] or result["failed"]:
        # Don't overwrite analysis/ and the web bundle with a partial dataset.
        raise SystemExit(f"folding incomplete ({result['remaining']} remaining, {result['failed']} failed"
                         f"{', interrupted' if result['interrupted'] else ''}); re-run `varfold all` to resume")
    for step in (cmd_llr, cmd_validate, cmd_analyze, cmd_export):
        step(cfg, args)


def cmd_serve(cfg, args):
    import uvicorn

    from .folding import ESMFoldPredictor
    from .worker import create_app

    token = os.environ.get("VARFOLD_WORKER_TOKEN", "")
    app = create_app(lambda: ESMFoldPredictor(cfg.folding, device=args.device), token=token,
                     max_length=args.max_length, max_queue=args.max_queue,
                     cache_dir=cfg.paths.root / "live_cache")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="varfold", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="target YAML (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("curate", help="download sequence + variants and build dataset.csv")
    p.add_argument("--refresh", action="store_true", help="re-download instead of using raw/ snapshots")
    p.set_defaults(func=cmd_curate)

    p = sub.add_parser("fold", help="batch ESMFold inference (resumable)")
    p.add_argument("--ids", nargs="+", help="only these variant ids (WT is always included)")
    p.add_argument("--limit", type=int, help="fold at most N structures this run")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.add_argument("--force", action="store_true", help="refold even if a prediction exists")
    p.set_defaults(func=cmd_fold)

    p = sub.add_parser("llr", help="ESM-2 masked-marginal baseline scores")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.set_defaults(func=cmd_llr)

    p = sub.add_parser("validate", help="compare WT prediction with the experimental structure")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_validate)

    sub.add_parser("analyze", help="metrics, statistics, figures, report").set_defaults(func=cmd_analyze)
    sub.add_parser("export", help="build the SQLite bundle for the web app").set_defaults(func=cmd_export)
    sub.add_parser("status", help="folding progress").set_defaults(func=cmd_status)

    p = sub.add_parser("all", help="curate → fold → llr → validate → analyze → export")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.set_defaults(func=cmd_all, ids=None, limit=None, force=False)

    p = sub.add_parser("serve", help="live inference worker for the web app (bind to your Tailscale IP)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-length", type=int, default=400)
    p.add_argument("--max-queue", type=int, default=3)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _setup_logging(cfg.paths.root, args.verbose)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
