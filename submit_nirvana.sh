#!/usr/bin/env bash
# submit_nirvana.sh — run this after setting a valid softmax token.
# Usage:
#   cogames auth set-token <FRESH_TOKEN>
#   bash submit_nirvana.sh
#
# Or override the season:
#   SEASON=among-them bash submit_nirvana.sh
set -euo pipefail

VENV=/home/user/venv
COGAMES=$VENV/bin/cogames
REPO=/home/user/softmax-among-them
POLICY_NAME="lessandro-nirvana"
BUNDLE="$REPO/lessandro_nirvana_bundle.zip"
SEASON="${SEASON:-among-them}"

echo "=== Auth status ==="
$COGAMES auth status

echo ""
echo "=== Available seasons ==="
$COGAMES season list 2>/dev/null || true

echo ""
echo "=== Uploading lessandro-nirvana ==="
$COGAMES upload \
  -p "$BUNDLE" \
  -n "$POLICY_NAME" \
  --no-submit \
  --skip-validation

echo ""
echo "=== Submitting to season: $SEASON ==="
$COGAMES submit "$POLICY_NAME" --season "$SEASON"

echo ""
echo "=== Submission complete ==="
echo "Check status with: $COGAMES submissions"
echo "Check matches with: $COGAMES matches"
