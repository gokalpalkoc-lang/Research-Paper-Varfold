#!/usr/bin/env bash
# Publishes a dataset bundle to the demo. Bundles are made by `varfold export`
# in the varfold-active copy (data/ar_lbd/web/varfold.sqlite).
#   ./scripts/publish.sh [path/to/varfold.sqlite]      default: demo/varfold-demo.sqlite
#
# Checks the bundle locally, uploads it, verifies the upload's SHA-256, swaps it in
# and restarts the app (it opens the file once, read-only, at start-up). If the app
# doesn't come back healthy, the previous bundle is put back.
set -euo pipefail

HOST="${VARFOLD_HOST:-YZ@178.105.242.183}"
DIR=/home/YZ/varfold
cd "$(dirname "$0")/.."
DB="${1:-demo/varfold-demo.sqlite}"
[[ -f "$DB" ]] || { echo "No bundle at $DB" >&2; exit 1; }

# A bundle the app can't open would take the site down, so check it before it leaves this machine.
python3 - "$DB" <<'PY'
import json, sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}
assert meta.get("schema_version") == 1, f"bundle schema {meta.get('schema_version')} != 1 (what the app reads)"
assert con.execute("SELECT 1 FROM variants WHERE variant_id = 'WT'").fetchone(), "bundle has no wild type"
ds = meta["stats"]["dataset"]
print(f"Bundle: {meta['target']['title']}; {ds['n_folded']} variants folded, {ds['n_missing']} not yet; "
      f"exported {meta['exported_at']}")
PY

sum="$(sha256sum "$DB" | cut -d' ' -f1)"
echo "Uploading $(du -h "$DB" | cut -f1) to $HOST:$DIR/data"
ssh "$HOST" "mkdir -p $DIR/data && cat > $DIR/data/varfold.sqlite.new" < "$DB"

ssh "$HOST" bash -s -- "$DIR" "$sum" <<'REMOTE'
set -euo pipefail
cd "$1"
compose() { docker compose -p varfold -f app/deploy/docker-compose.yml --env-file .env "$@"; }
if [ "$(sha256sum data/varfold.sqlite.new | cut -d' ' -f1)" != "$2" ]; then
  rm -f data/varfold.sqlite.new
  echo "The upload arrived corrupted; nothing was changed." >&2
  exit 1
fi
if [ -f data/varfold.sqlite ]; then cp data/varfold.sqlite data/varfold.sqlite.prev; fi
mv data/varfold.sqlite.new data/varfold.sqlite
if [ -z "$(compose ps -q varfold 2>/dev/null)" ]; then
  echo "The app isn't running yet; the bundle is in place for the first deploy."
  exit 0
fi
compose restart varfold
for _ in $(seq 1 20); do
  if docker exec deploy-caddy-1 wget -qO /dev/null http://varfold-app:8000/api/health 2>/dev/null; then
    echo "The app is serving the new bundle."
    exit 0
  fi
  sleep 2
done
echo "The app did not come back with the new bundle." >&2
if [ -f data/varfold.sqlite.prev ]; then
  mv data/varfold.sqlite.prev data/varfold.sqlite
  compose restart varfold
  echo "Put the previous bundle back." >&2
fi
exit 1
REMOTE
echo "Published $DB"
