#!/usr/bin/env bash
# Writes .github/workflows/ci.yml from scripts/gh_actions_ci_reference.yml (skips the 3-line how-to header).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.github/workflows"
tail -n +4 "$ROOT/scripts/gh_actions_ci_reference.yml" > "$ROOT/.github/workflows/ci.yml"
echo "Wrote $ROOT/.github/workflows/ci.yml"
echo ""
echo "Then run:"
echo "  git add .github/workflows/ci.yml && git commit -m 'ci: add GitHub Actions workflow' && git push"
echo "If push fails: use SSH, or a PAT with 'workflow' scope, or paste this file on github.com → Add file."
