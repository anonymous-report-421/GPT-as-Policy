#!/usr/bin/env bash
# This entire script is inlined into ACP's startup_script for inspection.
set -euo pipefail
umask 077
: "${CODE_ROOT:?Shared immutable source snapshot is required}"
: "${ROLLOUT_EXPERIMENT_ID:?}"
: "${ROBODOJO_TASK:?}"
: "${ROLLOUT_REPLICA_ID:?}"
: "${ROLLOUT_COUNT:?}"
: "${ROLLOUT_EVAL_MANIFEST:?}"
: "${ROLLOUT_EVAL_MANIFEST_SHA256:?}"
case "${ROLLOUT_AUTH_PROFILE:-galbot}" in
    codex_[abc]|codex_[abc]_[2-5]) ;;
    *) : "${ROLLOUT_OPENAI_API_KEY_FILE:?Shared private credential file is required}" ;;
esac
: "${ROLLOUT_RUN_UID:?}"
: "${ROLLOUT_RUN_GID:?}"
if [[ "$(id -u)" == 0 && "$ROLLOUT_RUN_UID" != 0 ]]; then
    exec setpriv --reuid="$ROLLOUT_RUN_UID" --regid="$ROLLOUT_RUN_GID" --clear-groups \
        bash "$CODE_ROOT/hybrid_rollout/robodojo/acp_startup.sh"
fi
[[ "$(id -u)" == "$ROLLOUT_RUN_UID" ]] || { echo 'Unexpected shared-storage user identity' >&2; exit 2; }
export ROLLOUT_SCHEDULER_JOB_ID="${JOB_NAME:-${HOSTNAME:-unknown}}"
export NO_PROXY="127.0.0.1,localhost,${ROLLOUT_GATEWAY_NO_PROXY-gateway.example.invalid}"
export no_proxy="$NO_PROXY"
# Keep the supervisor alive to forward SIGTERM to batch.py. A plain pipeline
# may lose this signal on container stop, leaving batch.json at "running".
exec 3>&1 4>&2
exec > >(tee -a "${ROLLOUT_BOOTSTRAP_LOG:?Shared bootstrap log path is required}") 2>&1
log_pid=$!
if [[ -n "${ROLLOUT_HTTPS_PROXY:-}${ROLLOUT_HTTPS_PROXY_FILE:-}" ]]; then
    case "${ROLLOUT_AUTH_PROFILE:-galbot}" in
        codex_[abc]|codex_[abc]_[2-5]) ;;
        *) echo 'Explicit subscription proxy is not allowed for API gateway slots' >&2; exit 2 ;;
    esac
    set +x
    if [[ -n "${ROLLOUT_HTTPS_PROXY_FILE:-}" ]]; then
        [[ -z "${ROLLOUT_HTTPS_PROXY:-}" ]] || { echo 'Choose exactly one proxy source' >&2; exit 2; }
        env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.proxy_config validate --file "$ROLLOUT_HTTPS_PROXY_FILE"
        IFS= read -r rollout_proxy < "$ROLLOUT_HTTPS_PROXY_FILE"
    else
        rollout_proxy="$ROLLOUT_HTTPS_PROXY"
    fi
    export HTTP_PROXY="$rollout_proxy" HTTPS_PROXY="$rollout_proxy"
    export http_proxy="$rollout_proxy" https_proxy="$rollout_proxy"
    unset rollout_proxy
    unset ALL_PROXY all_proxy
    # Verify TLS/HTTP egress without sending credentials or spending model tokens.
    # A reachable endpoint is not a successful subscription/model authorization.
    env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.proxy_config probe
fi
batch_pid=''
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$batch_pid" ]] && kill -0 "$batch_pid" 2>/dev/null; then
        kill -TERM "$batch_pid" 2>/dev/null || true
        wait "$batch_pid" 2>/dev/null || true
    fi
    exec 1>&3 2>&4 3>&- 4>&-
    wait "$log_pid" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
bash "$CODE_ROOT/hybrid_rollout/robodojo/cluster_entrypoint.sh" &
batch_pid=$!
wait "$batch_pid"
batch_pid=''
