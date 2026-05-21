#!/bin/bash
# Build and submit lessandro-nirvana to the Among Them Daily league.
# Run this after auth is fixed with a fresh softmax token.
# Usage: ./build_and_submit.sh [league_id]

set -euo pipefail

POLICY_NAME="lessandro-nirvana"
IMAGE_TAG="lessandro-nirvana:latest"
OPENROUTER_KEY="${OPENROUTER_API_KEY:-<SET_OPENROUTER_API_KEY>}"
OPENROUTER_MODEL="anthropic/claude-haiku-4-5"
LEAGUE_ID="${1:-}"

# 1. Verify auth
echo "=== Checking auth ==="
uv run softmax status

# 2. Find league if not provided
if [ -z "$LEAGUE_ID" ]; then
  echo "=== Listing leagues ==="
  uv run coworld leagues
  echo ""
  echo "Set LEAGUE_ID: export LEAGUE_ID=league_..."
  exit 1
fi

# 3. Build Docker image from source (requires HTTPS access)
echo "=== Building Docker image ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker build \
  --platform=linux/amd64 \
  -f "$SCRIPT_DIR/Dockerfile.from_source" \
  -t "$IMAGE_TAG" \
  "$SCRIPT_DIR"

# 4. Upload policy with secrets
echo "=== Uploading policy ==="
uv run coworld upload-policy "$IMAGE_TAG" \
  --name "$POLICY_NAME" \
  --secret-env "OPENROUTER_API_KEY=$OPENROUTER_KEY" \
  --secret-env "OPENROUTER_MODEL=$OPENROUTER_MODEL"

# 5. Submit to league
echo "=== Submitting to league $LEAGUE_ID ==="
uv run coworld submit "$POLICY_NAME" --league "$LEAGUE_ID"

# 6. Check status
echo "=== Submission status ==="
uv run coworld submissions --mine --league "$LEAGUE_ID"

echo ""
echo "Done! Watch placement at: https://softmax.com/observatory/v2 → Leagues → Among Them Daily"
