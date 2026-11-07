.PHONY: demo setup test test-engine test-web typecheck build dev-api dev-web deploy publish setup-server

PY := .venv/bin/python
DB ?= demo/varfold-demo.sqlite

setup:            ## venv with the web backend + the engine's light core (API tests need its exporter; no PyTorch), and npm ci
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e "engine[dev]" -r web/backend/requirements.txt
	cd web/frontend && npm ci

test: test-engine test-web typecheck

test-engine:
	cd engine && ../$(PY) -m pytest -q

test-web:
	cd web/backend && ../../$(PY) -m pytest -q

typecheck:
	cd web/frontend && npx tsc --noEmit

build:            ## production frontend bundle
	cd web/frontend && npm run build

dev-api:          ## API on :8000 against $(DB)
	cd web/backend && VARFOLD_DB=$(abspath $(DB)) ../../.venv/bin/uvicorn app.main:app_factory --factory --reload --port 8000

dev-web:          ## Vite dev server on :5173 (proxies /api to :8000)
	cd web/frontend && npx vite

demo: build       ## the site exactly as the server runs it, from $(DB) → http://127.0.0.1:8000
	cd web/backend && VARFOLD_DB=$(abspath $(DB)) VARFOLD_STATIC_DIR=$(abspath web/frontend/dist) \
	  ../../.venv/bin/uvicorn app.main:app_factory --factory --port 8000

setup-server:     ## one-time: varfold.devtga.com on the shared server (see README)
	./scripts/setup-server.sh $(DB)

deploy:           ## ship the committed HEAD to varfold.devtga.com
	./scripts/deploy.sh

publish:          ## replace the server's dataset: make publish DB=path/to/varfold.sqlite
	./scripts/publish.sh $(DB)
