#!/usr/bin/env bash
set -euo pipefail

# CPU-only web UI. No model, simulator, credentials, or cluster job commands.
REVIEW_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROLLOUT_REVIEW_BASE="${ROLLOUT_REVIEW_BASE:-/mnt/rollout/robodojo_mixed_control}"
ROLLOUT_REVIEW_RESULTS="${ROLLOUT_REVIEW_RESULTS:-${ROLLOUT_REVIEW_BASE}/results}"
ROLLOUT_REVIEW_ANNOTATIONS="${ROLLOUT_REVIEW_ANNOTATIONS:-${ROLLOUT_REVIEW_BASE}/annotations/video_review_v1}"
ROLLOUT_REVIEW_LOG_ROOT="${ROLLOUT_REVIEW_LOG_ROOT:-${ROLLOUT_REVIEW_BASE}/dashboard/video_review_v1}"
ROLLOUT_REVIEW_PORT="${ROLLOUT_REVIEW_PORT:-8767}"
ROLLOUT_REVIEW_HOST="${ROLLOUT_REVIEW_HOST:-127.0.0.1}"
ROLLOUT_REVIEW_PYTHON="${ROLLOUT_REVIEW_PYTHON:-python3}"

umask 077
mkdir -p -- "$ROLLOUT_REVIEW_LOG_ROOT"
REVIEW_LOG="${ROLLOUT_REVIEW_LOG_ROOT}/server_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
cd -- "$REVIEW_REPO_ROOT"
printf 'Video review bind: http://%s:%s\nAnnotations: %s\nLog: %s\n' \
  "$ROLLOUT_REVIEW_HOST" "$ROLLOUT_REVIEW_PORT" "$ROLLOUT_REVIEW_ANNOTATIONS" "$REVIEW_LOG"
exec "$ROLLOUT_REVIEW_PYTHON" -u -m hybrid_rollout.annotation_dashboard \
  --results-root "$ROLLOUT_REVIEW_RESULTS" --annotation-root "$ROLLOUT_REVIEW_ANNOTATIONS" \
  --host "$ROLLOUT_REVIEW_HOST" --port "$ROLLOUT_REVIEW_PORT" >>"$REVIEW_LOG" 2>&1
