#!/usr/bin/env bash
# One fresh process tree and Codex home per episode, called by batch.py.
set -euo pipefail

ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd "$ENTRY_ROOT/../.." && pwd)}"
ROLLOUT_EXPERIMENT_ID="${ROLLOUT_EXPERIMENT_ID:?Set ROLLOUT_EXPERIMENT_ID}"
ROBODOJO_TASK="${ROBODOJO_TASK:?Set ROBODOJO_TASK}"
ROLLOUT_REPLICA_ID="${ROLLOUT_REPLICA_ID:?Set ROLLOUT_REPLICA_ID}"
ROLLOUT_LAYOUT_ID="${ROLLOUT_LAYOUT_ID:?Set ROLLOUT_LAYOUT_ID from the evaluation manifest}"
ROLLOUT_EVAL_SEED="${ROLLOUT_EVAL_SEED:?Set ROLLOUT_EVAL_SEED from the evaluation manifest}"
: "${ROLLOUT_CASE_FILE:?}"
: "${ROLLOUT_ARCHIVE:?}"
ROLLOUT_ATTEMPT="${ROLLOUT_ATTEMPT:-0}"
ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-/mnt/rollout/robodojo_mixed_control}"
[[ "$ROLLOUT_SHARED_ROOT" == /mnt/* ]] || { echo 'ROLLOUT_SHARED_ROOT must be below /mnt' >&2; exit 2; }

for identifier in "$ROLLOUT_EXPERIMENT_ID" "$ROBODOJO_TASK" "$ROLLOUT_REPLICA_ID" "$ROLLOUT_ATTEMPT"; do
    [[ "$identifier" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid rollout identifier: $identifier" >&2; exit 2; }
done
for seed in "$ROLLOUT_LAYOUT_ID" "$ROLLOUT_EVAL_SEED"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo 'Layout ID and eval seed must be non-negative integers' >&2; exit 2; }
done

umask 077
private_parent="$ROLLOUT_SHARED_ROOT/private_runtime/$ROLLOUT_EXPERIMENT_ID/replica_$ROLLOUT_REPLICA_ID/attempt_$ROLLOUT_ATTEMPT"
mkdir -p "$private_parent"
private_root="$(mktemp -d "$private_parent/${ROBODOJO_TASK}_g${ROLLOUT_EVAL_SEED}_l${ROLLOUT_LAYOUT_ID}.XXXXXX")"
chmod 700 "$private_root"
launcher_pid=''
credential_file="$private_root/openai_api_key"
cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
        kill -TERM "$launcher_pid" 2>/dev/null || true
        wait "$launcher_pid" 2>/dev/null || true
    fi
    # Preserve Codex DB/logs, caches and crash evidence outside public archives.
    # Only this episode's copied credential is removed on an orderly exit.
    rm -f -- "$credential_file"
    printf '%s\n' "$status" > "$private_root/episode_exit_status.txt"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

codex_home_dir="$private_root/codex_home"
mkdir -p "$codex_home_dir"
env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" \
    -m hybrid_rollout.robodojo.codex_backend.profiles config --output "$codex_home_dir/config.toml"
chmod 600 "$codex_home_dir/config.toml"
export ROLLOUT_CODEX_STATE_DIR="$codex_home_dir"
case "${ROLLOUT_AUTH_PROFILE:-galbot}" in
codex_[abc]|codex_[abc]_[2-5])
    # Only credentials/config persist. The app-server uses a fresh thread and
    # per-episode sqlite/log directories; no copying rotating OAuth tokens.
    credential_source='chatgpt_managed_file'
    codex_home_dir="$ROLLOUT_SHARED_ROOT/private/auth_profiles/$ROLLOUT_AUTH_PROFILE/codex_home"
    [[ -s "$codex_home_dir/auth.json" ]] || { echo 'Complete isolated account login first' >&2; exit 2; }
    unset OPENAI_API_KEY ROLLOUT_OPENAI_API_KEY_FILE
    ;;
*)
if [[ -n "${ROLLOUT_OPENAI_API_KEY_FILE:-}" ]]; then
    [[ -f "$ROLLOUT_OPENAI_API_KEY_FILE" ]] || { echo 'Credential file does not exist' >&2; exit 2; }
    IFS= read -r api_key < "$ROLLOUT_OPENAI_API_KEY_FILE" || [[ -n "${api_key:-}" ]]
    credential_source='file'
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
    api_key="$OPENAI_API_KEY"
    credential_source='environment'
else
    echo 'Inject OPENAI_API_KEY or ROLLOUT_OPENAI_API_KEY_FILE at job submission' >&2
    exit 2
fi
[[ "$api_key" == sk-* ]] || { echo 'Injected API key has an unexpected format' >&2; exit 2; }
printf '%s' "$api_key" > "$credential_file"
chmod 600 "$credential_file"
unset api_key OPENAI_API_KEY
export ROLLOUT_OPENAI_API_KEY_FILE="$credential_file"
;;
esac

export CODE_ROOT ROLLOUT_SHARED_ROOT ROLLOUT_EXPERIMENT_ID ROLLOUT_REPLICA_ID ROLLOUT_ATTEMPT
export TASK="$ROBODOJO_TASK" SEED="$ROLLOUT_LAYOUT_ID" EVAL_SEED="$ROLLOUT_EVAL_SEED"
export RESULTS_ROOT="$(dirname "$ROLLOUT_ARCHIVE")"
export RUN_ID="attempt_$ROLLOUT_ATTEMPT"
export TASK_ROOT="$private_root"
export TMPDIR="$private_root/tmp"
export XDG_CONFIG_HOME="$private_root/config"
export XDG_DATA_HOME="$private_root/data"
export XDG_STATE_HOME="$private_root/state"
export XDG_CACHE_HOME="$private_root/cache"
export CUDA_CACHE_PATH="$private_root/cuda_cache"
export TORCH_EXTENSIONS_DIR="$private_root/torch_extensions"
export MPLCONFIGDIR="$private_root/matplotlib"
# Native eval_result/ and any relative side outputs are unique and shared too.
export ROBODOJO_RUN_ID="${ROLLOUT_EXPERIMENT_ID}_r${ROLLOUT_REPLICA_ID}_i${ROLLOUT_INDEX}_a${ROLLOUT_ATTEMPT}"
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"
export ROLLOUT_CODEX_HOME_DIR="$codex_home_dir"
export ROLLOUT_CODEX_CONFIG="$ROLLOUT_CODEX_STATE_DIR/config.toml"
export ROLLOUT_CREDENTIAL_SOURCE="$credential_source"

set +e
"$CODE_ROOT/hybrid_rollout/robodojo/run_local.sh" &
launcher_pid=$!
wait "$launcher_pid"
status=$?
launcher_pid=''
set -e
exit "$status"
