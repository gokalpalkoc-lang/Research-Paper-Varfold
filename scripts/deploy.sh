#!/usr/bin/env bash
# Deploys the committed HEAD to varfold.devtga.com: ships the code with
# `git archive`, then builds and restarts the container on the server.
# Uncommitted changes are not deployed. The dataset bundle is separate
# (scripts/publish.sh) and a deploy never touches it.
set -euo pipefail

HOST="${VARFOLD_HOST:-YZ@178.105.242.183}"
DIR=/home/YZ/varfold
cd "$(dirname "$0")/.."
REV="$(git rev-parse --short HEAD)"

if ! git diff --quiet HEAD; then
  echo "Note: uncommitted changes are not deployed; deploying $REV." >&2
fi
# The app refuses to start without a bundle, so don't replace a running app before there is one.
ssh "$HOST" "test -f $DIR/data/varfold.sqlite" ||
  { echo "No dataset bundle on the server yet: run scripts/setup-server.sh first." >&2; exit 1; }

echo "Shipping $REV to $HOST:$DIR/app"
git archive --format=tar HEAD | ssh "$HOST" "
  set -e
  rm -rf $DIR/app.new && mkdir -p $DIR/app.new
  tar -x -C $DIR/app.new
  echo $REV > $DIR/app.new/REVISION
  rm -rf $DIR/app && mv $DIR/app.new $DIR/app
"

echo "Building and restarting"
ssh "$HOST" "
  set -e
  cd $DIR
  docker compose -p varfold -f app/deploy/docker-compose.yml --env-file .env up -d --build
  docker image prune -f >/dev/null
  docker compose -p varfold -f app/deploy/docker-compose.yml --env-file .env ps
"

echo "Deployed $REV: https://varfold.devtga.com"
