#!/usr/bin/env bash
# Commit and push the ./registry checkout back to the mlflow-registry branch.
# Retries with a rebase in case another writer slipped in (workflows are already
# serialized by an Actions concurrency group; this is belt-and-braces).
#
# Usage: registry_push.sh "commit message"
set -euo pipefail
cd "$(dirname "$0")/../registry"

MSG=${1:?commit message required}
BRANCH=mlflow-registry

git add -A
if git diff --cached --quiet; then
  echo "no registry changes to push"
  exit 0
fi
git -c user.name="github-actions[bot]" \
  -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
  commit -m "$MSG"

for attempt in 1 2 3; do
  if git push origin "HEAD:$BRANCH"; then
    echo "pushed registry state"
    exit 0
  fi
  echo "push rejected (attempt $attempt); rebasing on remote"
  git fetch origin "$BRANCH"
  git rebase "origin/$BRANCH" || { git rebase --abort; break; }
  sleep 2
done
echo "failed to push registry state" >&2
exit 1
