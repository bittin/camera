#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Merged dates survive the July 2026 LFS history rewrite, unlike merge SHAs.
# Intentionally retain the existing unbounded history and predecessor selection.
set -euo pipefail

TAG=${1:?release tag}
base=${2:?PR base branch}
repo_url=${3:?repository URL}
output=${4:?output path}

# No matches are valid; drain the pipeline so pipefail still exposes git errors.
PREVIOUS_TAG=$(
  git tag --sort=-v:refname |
    { grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || [ "$?" -eq 1 ]; } |
    { grep -v "^${TAG}$" || [ "$?" -eq 1 ]; } |
    sed -n '1p'
)

if [ "${5:-}" = --fallback ]; then
  echo "Reconstructing notes from $PREVIOUS_TAG to HEAD for tag $TAG"
fi

{
  echo "## What's Changed"
  if [ -n "$PREVIOUS_TAG" ]; then
    PREV_DATE=$(git log -1 --format=%cI "$PREVIOUS_TAG")
    gh pr list --state merged --base "$base" --limit 1000 \
      --json number,title,author,mergedAt \
      --jq "[.[] | select(.mergedAt > \"$PREV_DATE\")] | sort_by(.mergedAt) | .[] | \"* \(.title) by @\(if (.author.login | startswith(\"app/\")) then (.author.login[4:] + \"[bot]\") else .author.login end) in ${repo_url}/pull/\(.number)\""
    echo ""
    echo "**Full Changelog**: ${repo_url}/compare/${PREVIOUS_TAG}...${TAG}"
  else
    gh pr list --state merged --base "$base" --limit 1000 \
      --json number,title,author,mergedAt \
      --jq "sort_by(.mergedAt) | .[] | \"* \(.title) by @\(if (.author.login | startswith(\"app/\")) then (.author.login[4:] + \"[bot]\") else .author.login end) in ${repo_url}/pull/\(.number)\""
  fi
} > "$output"

# Preparation uses a fallback; final publication historically does not.
if [ "${5:-}" = --fallback ] && { [ ! -s "$output" ] || ! grep -q '^\* ' "$output"; }; then
  echo "No PRs found, using fallback"
  echo "Bug fixes and improvements." > "$output"
fi
