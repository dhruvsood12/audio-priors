#!/usr/bin/env bash
# Applies the pip + notebook CI to GitHub without using `git push` for workflow files.
# Uses the GitHub REST API (same auth rules: classic PAT needs "repo" + "workflow").
#
# Usage:
#   export GITHUB_TOKEN=ghp_xxxxxxxx   # classic PAT: enable repo + workflow
#   ./scripts/sync_ci_workflow_to_github.sh
#
# Optional: GITHUB_REPO=owner/repo GITHUB_BRANCH=main

set -euo pipefail

REPO="${GITHUB_REPO:-dhruvsood12/SongAddiction}"
BRANCH="${GITHUB_BRANCH:-main}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CI_FILE="$ROOT/.github/workflows/ci.yml"
CONDA_PATH=".github/workflows/python-package-conda.yml"
CI_PATH=".github/workflows/ci.yml"

TOKEN="${GITHUB_TOKEN:-${1:-}}"
if [[ -z "$TOKEN" ]]; then
  echo "Set GITHUB_TOKEN to a classic PAT with repo + workflow scopes, then re-run."
  echo "Create: GitHub → Settings → Developer settings → Personal access tokens"
  exit 1
fi

if [[ ! -f "$CI_FILE" ]]; then
  REF="$ROOT/scripts/gh_actions_ci_reference.yml"
  if [[ -f "$REF" ]]; then
    mkdir -p "$(dirname "$CI_FILE")"
    tail -n +4 "$REF" > "$CI_FILE"
    echo "Wrote $CI_FILE from $REF (no header comments)."
  else
    echo "Missing $CI_FILE and $REF"
    exit 1
  fi
fi

api_get_sha() {
  local path="$1" resp
  resp=$(curl -sS -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/contents/${path}?ref=$BRANCH" 2>/dev/null) || true
  echo "$resp" | python3 -c "import sys,json
try:
  d=json.load(sys.stdin)
  print(d.get('sha','') or '')
except Exception:
  print('')
"
}

api_delete() {
  local path="$1" sha="$2" msg="$3"
  local data
  data=$(python3 -c "import json,sys; print(json.dumps({'message': sys.argv[1], 'sha': sys.argv[2], 'branch': sys.argv[3]}))" "$msg" "$sha" "$BRANCH")
  curl -fsS -X DELETE \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "$data" \
    "https://api.github.com/repos/$REPO/contents/$path"
}

api_put_ci() {
  local existing_sha="$1"
  python3 << PY
import json, base64, pathlib
path = pathlib.Path("$CI_FILE")
content = base64.b64encode(path.read_bytes()).decode()
body = {
    "message": "ci: add pip + notebook GitHub Actions workflow",
    "content": content,
    "branch": "$BRANCH",
}
sha = "$existing_sha".strip()
if sha:
    body["sha"] = sha
print(json.dumps(body))
PY
}

echo "→ Checking $CONDA_PATH on $REPO@$BRANCH ..."
SHA_CONDA=$(api_get_sha "$CONDA_PATH")
if [[ -n "$SHA_CONDA" ]]; then
  echo "→ Deleting Conda template workflow ..."
  api_delete "$CONDA_PATH" "$SHA_CONDA" "ci: remove conda template (no environment.yml in repo)"
  echo "   Deleted."
else
  echo "   (already absent)"
fi

echo "→ Uploading $CI_PATH ..."
SHA_CI=$(api_get_sha "$CI_PATH")
BODY=$(api_put_ci "$SHA_CI")
curl -fsS -X PUT \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d "$BODY" \
  "https://api.github.com/repos/$REPO/contents/$CI_PATH"
echo ""
echo "Done. Sync your clone:"
echo "  cd \"$ROOT\" && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH"
