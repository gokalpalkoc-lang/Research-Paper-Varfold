# VarFold demo

**The public showcase of VarFold, live at <https://varfold.devtga.com>.**

VarFold folds every clinically classified missense variant in the androgen receptor
ligand-binding domain with ESMFold, measures where the model loses confidence relative to
the wild type, and tests whether those "structural breaking points" track pathogenicity.

This branch (`demo`) is only the website. The research itself (curation, folding,
statistics, figures for the paper) lives on the **`active`** branch (local copy:
`~/Desktop/VarFold/varfold-active`). The two share history, so a fix made on one can be
moved to the other with `git cherry-pick`.

| | `active` (paper) | `demo` (this branch) |
|---|---|---|
| Runs on | Your GPU workstation | The shared Hetzner server behind devtga.com |
| Does | Curate → fold → LLR → validate → analyze → **export** a bundle | Serves one exported bundle, read-only |
| Needs | PyTorch, ESMFold weights (~11 GB) | FastAPI + a ~0.5–20 MB SQLite file. No PyTorch, no model |

![Overview page](docs/overview.png)

## Look at it locally

You need Python 3.10+, Node 20+ and `make`.

```bash
make setup                 # venv (no PyTorch) + npm ci
make demo                  # http://127.0.0.1:8000, exactly what the server runs
make demo DB=../varfold-active/data/ar_lbd/web/varfold.sqlite   # preview a fresh export
make test                  # engine core + API tests + frontend type-check
```

For frontend work: `make dev-api` and `make dev-web` (Vite on :5173, `/api` proxied to :8000).

## Deploying to devtga.com

The server (`YZ@178.105.242.183`) also runs scout, the portfolio (devtga.com) and Operon.
**Scout's Caddy owns ports 80/443**, so the demo has no Caddy of its own:

- the app is its own Compose project (`-p varfold`) in `/home/YZ/varfold`, and joins scout's
  `deploy_default` network as `varfold-app` ([deploy/docker-compose.yml](deploy/docker-compose.yml));
- its site block ([deploy/Caddyfile.varfold](deploy/Caddyfile.varfold)) is appended to scout's
  Caddyfile, with a backup, validation inside the running Caddy, a reload rather than a
  restart, and an automatic rollback if any other site's response changes.

Editing scout's Caddyfile and checking the other sites are shared with every project on
the server, so they live in **`~/Desktop/scripts`** (`devtga-add-site.sh`,
`devtga-status.sh`; see its README) rather than in this repo. Set `DEVTGA_SCRIPTS` if
that folder moves.

**First time**

1. In Cloudflare, add `varfold` → CNAME → `devtga.com`, **DNS only** (Caddy needs it to
   get a certificate).
2. Commit, then run `./scripts/setup-server.sh` (or `make setup-server`) **from this
   folder**. It creates the server's `.env` (live folding off), uploads the demo bundle,
   deploys, adds the Caddy block and checks every site. It is safe to re-run; if the DNS
   record isn't there yet it stops before touching Caddy, and the next run carries on.

**Updates**

```bash
./scripts/deploy.sh                                   # code: ships the committed HEAD
./scripts/publish.sh path/to/varfold.sqlite           # data: replaces the bundle
```

`publish.sh` checks the bundle before uploading, verifies the upload's SHA-256, swaps it in
and restarts the app; if the app doesn't come back healthy, the previous bundle goes back.
Typical flow after a research run: `varfold export` in `varfold-active`, then
`./scripts/publish.sh ../varfold-active/data/ar_lbd/web/varfold.sqlite`.

### Load on the server

The server never folds anything. It runs one uvicorn process over a read-only SQLite file
(measured: **76 MB RAM** serving every page of the site), with API responses cacheable for
5 minutes and structures for a day. Compose caps it at 384 MB, half a CPU core and 128
processes, and rotates its logs (3 × 10 MB). The heaviest moment is a deploy, which builds
the frontend once on the server, as the other projects' deploys do.

### Live folding stays off

`VARFOLD_WORKER_URL` is empty in the server's `.env`, so the *Live fold* page answers only
sequences that are already in the dataset (instantly, from the bundle) and says live folding
is not enabled for anything else. Turning it on would forward folds to the GPU workstation
over Tailscale (the worker and its setup live on the `active` branch), so the public could
queue work on that machine and the page would show "offline" whenever it sleeps. Leave it
off for the public demo.

### Preliminary data

While the bundle is incomplete (some variants not yet folded), the Overview page shows a
"Preliminary" notice above the headline numbers. The bundled demo dataset has 9 of 405
variants, so its statistics are not meaningful; publish the full export before linking the
demo from the paper.

## Pages

**Overview** (headline numbers, WT structure, breaking-point map, discrimination table),
**Variants** (filter/sort/search, URL-shareable state), **Variant detail** (3D mutant vs WT
ghost, pLDDT or ΔpLDDT colouring, per-residue charts, ClinVar links, PDB download),
**Live fold**, and **Methods** (generated from the dataset's own provenance). The JSON API is
documented at `/api/docs`.

## Repository layout

```
web/backend/app/    FastAPI: read-only dataset API, live-fold proxy (off), SPA hosting
web/backend/tests/  API tests against a bundle built by the real exporter
web/frontend/       React + TypeScript + Vite; 3Dmol.js viewer, SVG charts
deploy/             Compose file + Caddy site block for the shared server, .env template
scripts/            setup-server.sh (once), deploy.sh (code), publish.sh (data); they use ~/Desktop/scripts
demo/               bundled demo dataset (WT + 9 variants)
engine/             the research engine's core, kept for the API tests' exporter
```

## Credits

ESMFold & ESM-2 (Meta AI, MIT licence) · UniProt · EBI Proteins API · ClinVar ·
PDBe SIFTS · RCSB PDB · 3Dmol.js (BSD). Research use only. Nothing here is a
clinical interpretation.
