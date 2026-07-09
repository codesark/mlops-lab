#!/usr/bin/env bash
# Post or update a sticky PR comment holding one candidate's metrics report.
# Requires GH_TOKEN and GITHUB_REPOSITORY in the environment (set by Actions).
#
# Usage: pr_comment.sh <config-name> <report.md> <pr-number>
set -euo pipefail

CONFIG=${1:?config name}
REPORT=${2:?report file}
PR=${3:?pr number}
MARKER="<!-- mlops-metrics:$CONFIG -->"

BODY="$MARKER
$(cat "$REPORT")"

COMMENT_ID=$(gh api "repos/$GITHUB_REPOSITORY/issues/$PR/comments" --paginate \
  --jq ".[] | select(.body | startswith(\"$MARKER\")) | .id" | head -1)

if [[ -n "$COMMENT_ID" ]]; then
  gh api -X PATCH "repos/$GITHUB_REPOSITORY/issues/comments/$COMMENT_ID" \
    -f body="$BODY" >/dev/null
  echo "updated metrics comment for $CONFIG"
else
  gh api -X POST "repos/$GITHUB_REPOSITORY/issues/$PR/comments" \
    -f body="$BODY" >/dev/null
  echo "created metrics comment for $CONFIG"
fi
