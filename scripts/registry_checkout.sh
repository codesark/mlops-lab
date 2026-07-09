#!/usr/bin/env bash
# Check out the shared mlflow-registry branch into ./registry (fresh shallow clone).
#
# Usage: registry_checkout.sh [repo-url] [--read-only]
#   repo-url    defaults to `git remote get-url origin`; CI passes a token URL
#   --read-only if the branch doesn't exist, start with an empty local registry
#               instead of bootstrapping and pushing the branch (used on PRs)
set -euo pipefail
cd "$(dirname "$0")/.."

URL=${1:-$(git remote get-url origin)}
READ_ONLY=${2:-}
BRANCH=mlflow-registry

rm -rf registry
if git clone --depth 1 --branch "$BRANCH" "$URL" registry 2>/dev/null; then
  echo "checked out $BRANCH into ./registry"
  exit 0
fi

if [[ "$READ_ONLY" == "--read-only" ]]; then
  echo "$BRANCH branch missing; starting with empty local registry (read-only mode)"
  mkdir -p registry/deployments
  : > registry/deployments/history.ndjson
  exit 0
fi

echo "$BRANCH branch missing; bootstrapping it"
mkdir -p registry
git -C registry init -b "$BRANCH"
git -C registry remote add origin "$URL"
mkdir -p registry/deployments
: > registry/deployments/history.ndjson
cat > registry/README.md <<'EOF'
# mlflow-registry branch

Shared MLflow tracking + model registry state for this repo's MLOps lab:
sqlite backend store (`mlflow.db`), proxied artifacts (`mlruns/`), and
deployment manifests (`deployments/`). Written only by CI workflows,
serialized via a shared Actions concurrency group. A real org would use a
hosted tracking server + object storage; this branch is the lab stand-in.

Inspect locally:  make registry-pull && make mlflow-up
EOF
git -C registry add -A
git -C registry -c user.name="github-actions[bot]" \
  -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
  commit -m "bootstrap empty registry"
git -C registry push origin "$BRANCH"
echo "bootstrapped and pushed $BRANCH"
