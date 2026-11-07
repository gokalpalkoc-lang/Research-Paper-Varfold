#!/usr/bin/env bash
# One-time setup of the VarFold demo on the shared Hetzner server. Safe to re-run:
# each step checks what is already there and skips it.
#   ./scripts/setup-server.sh [path/to/varfold.sqlite]   default bundle: demo/varfold-demo.sqlite
#
# Uses the shared server tools in ~/Desktop/scripts (override with DEVTGA_SCRIPTS):
# devtga-add-site.sh edits scout's Caddyfile safely, devtga-status.sh checks every site.
#
#   1. Creates /home/YZ/varfold/.env from deploy/.env.example (live folding off).
#   2. Uploads a dataset bundle if the server has none (scripts/publish.sh).
#   3. Deploys the app (scripts/deploy.sh), unless it is already running.
#   4. Adds deploy/Caddyfile.varfold to scout's Caddyfile with devtga-add-site.sh: it
#      checks varfold.devtga.com's DNS, validates, backs up, reloads, and puts the old
#      file back if any other site's response changes.
#   5. Checks the demo and every other site.
#
# Later updates: scripts/deploy.sh ships code, scripts/publish.sh ships a bundle.
set -euo pipefail

HOST="${VARFOLD_HOST:-YZ@178.105.242.183}"
DIR=/home/YZ/varfold
DOMAIN=varfold.devtga.com
CADDY=deploy-caddy-1
APP=varfold-varfold-1
SCRIPTS="${DEVTGA_SCRIPTS:-$HOME/Desktop/scripts}"
export DEVTGA_HOST="$HOST"

cd "$(dirname "$0")/.."
for tool in devtga-add-site.sh devtga-status.sh; do
  [[ -x "$SCRIPTS/$tool" ]] ||
    { echo "Needs $SCRIPTS/$tool (the shared server tools; set DEVTGA_SCRIPTS if they live elsewhere)." >&2; exit 1; }
done

echo "1/5 Server settings"
if ssh "$HOST" "test -f $DIR/.env"; then
  echo "    $DIR/.env already exists; keeping it."
else
  ssh "$HOST" "mkdir -p $DIR/data && umask 077 && cat > $DIR/.env" < deploy/.env.example
  echo "    Created $DIR/.env from deploy/.env.example (live folding off)."
fi

echo "2/5 Dataset bundle"
if ssh "$HOST" "test -f $DIR/data/varfold.sqlite"; then
  echo "    The server already has one; scripts/publish.sh replaces it."
else
  ./scripts/publish.sh "${1:-demo/varfold-demo.sqlite}"
fi

echo "3/5 Deploy"
if ssh "$HOST" "docker ps --format '{{.Names}}' | grep -qx $APP"; then
  echo "    Already running; use scripts/deploy.sh to ship updates."
else
  ./scripts/deploy.sh
fi
echo "    Waiting for the app to answer on scout's network…"
for _ in $(seq 1 30); do
  ssh "$HOST" "docker exec $CADDY wget -qO /dev/null http://varfold-app:8000/api/health" 2>/dev/null && break
  sleep 2
done
ssh "$HOST" "docker exec $CADDY wget -qO /dev/null http://varfold-app:8000/api/health"

echo "4/5 Caddy: $DOMAIN -> varfold-app"
"$SCRIPTS/devtga-add-site.sh" deploy/Caddyfile.varfold

echo "5/5 Checks"
# A new certificate can take a little while on the first run.
for _ in $(seq 1 15); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "https://$DOMAIN/api/health")" = 200 ] && break
  sleep 4
done
"$SCRIPTS/devtga-status.sh"
if [ "$(curl -s -o /dev/null -w '%{http_code}' "https://$DOMAIN/api/health")" = 200 ]; then
  echo "Done: https://$DOMAIN"
else
  echo "The demo does not answer yet at https://$DOMAIN/api/health" >&2
  exit 1
fi
