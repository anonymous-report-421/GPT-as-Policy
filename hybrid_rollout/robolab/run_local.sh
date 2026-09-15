#!/usr/bin/env bash
set -euo pipefail

# Independent three-part Codex + pi05 + RoboLab rollout; no old pipeline imports.

CODE_ROOT="${CODE_ROOT:-/workspace/eval-of-gpt-6-astra-as-policy}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/rollout/openpi_mailbox_repro}"
LEGACY_RUNTIME_ROOT="${LEGACY_RUNTIME_ROOT:-/workspace/eval-of-gpt-6-astra-as-policy}"
RUN_ID="${RUN_ID:?Set RUN_ID to a unique experiment identifier}"
POLICY_PORT="${POLICY_PORT:-18820}"
SIM_PORT="${SIM_PORT:-19103}"
CODEX_BIN="${CODEX_BIN:-}"

if [[ -z "$CODEX_BIN" ]]; then
    CODEX_BIN="$(command -v codex || true)"
fi
if [[ -z "$CODEX_BIN" || ! -x "$CODEX_BIN" ]]; then
    printf 'Codex CLI is required for the GPT-6-Astra action reviewer\n' >&2
    exit 2
fi

CHECKPOINT="$RUNTIME_ROOT/checkpoints/pi05_droid_jointpos"
OPENPI_SOURCE="$LEGACY_RUNTIME_ROOT/src/openpi"
ROBOLAB_SOURCE="$LEGACY_RUNTIME_ROOT/src/RoboLab"
OPENPI_PYTHON="$RUNTIME_ROOT/openpi-venv/bin/python"
ROBOLAB_PYTHON="$LEGACY_RUNTIME_ROOT/envs/robolab/bin/python"
CONTROL_ROOT="$CODE_ROOT/runtime/$RUN_ID"
ARCHIVE_ROOT="$RUNTIME_ROOT/results/$RUN_ID"
CONTROLLER_OUTPUT="$ARCHIVE_ROOT/controller"
SOURCE_SNAPSHOT="$ARCHIVE_ROOT/source_snapshot"
LOG_ROOT="$ARCHIVE_ROOT/logs"

for required in \
    "$CHECKPOINT/params" \
    "$CHECKPOINT/assets/droid/norm_stats.json" \
    "$OPENPI_SOURCE/src/openpi" \
    "$ROBOLAB_SOURCE/robolab" \
    "$OPENPI_PYTHON" \
    "$ROBOLAB_PYTHON"; do
    if [[ ! -e "$required" ]]; then
        printf 'Missing required reproduction input: %s\n' "$required" >&2
        exit 2
    fi
done

if [[ -e "$CONTROL_ROOT" || -e "$ARCHIVE_ROOT" ]]; then
    printf 'Refusing to reuse an existing run path: %s or %s\n' "$CONTROL_ROOT" "$ARCHIVE_ROOT" >&2
    exit 2
fi

mkdir -p "$CONTROL_ROOT" "$ARCHIVE_ROOT" "$SOURCE_SNAPSHOT" "$LOG_ROOT" "$ARCHIVE_ROOT/sim" "$RUNTIME_ROOT/tmp"
cp -a "$CODE_ROOT/hybrid_rollout" "$SOURCE_SNAPSHOT/hybrid_rollout"
git -C "$CODE_ROOT" rev-parse HEAD > "$ARCHIVE_ROOT/repository_head.txt"
git -C "$CODE_ROOT" status --short > "$ARCHIVE_ROOT/repository_status.txt"
git -C "$CODE_ROOT" diff -- hybrid_rollout > "$ARCHIVE_ROOT/source.patch"
git -C "$OPENPI_SOURCE" rev-parse HEAD > "$ARCHIVE_ROOT/openpi_head.txt"
git -C "$ROBOLAB_SOURCE" rev-parse HEAD > "$ARCHIVE_ROOT/robolab_head.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader \
    > "$ARCHIVE_ROOT/gpu_inventory.csv"

POLICY_PID=""
SIM_PID=""
CONTROLLER_PID=""

stop_owned_pid() {
    local pid="$1"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return
        fi
        sleep 0.5
    done
    kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    set +e
    stop_owned_pid "$CONTROLLER_PID"
    stop_owned_pid "$SIM_PID"
    stop_owned_pid "$POLICY_PID"
    printf '%s\n' "$status" > "$ARCHIVE_ROOT/job_exit_status.txt"
}
trap cleanup EXIT

export TMPDIR="$RUNTIME_ROOT/tmp"
export OPENPI_DATA_HOME="$RUNTIME_ROOT/openpi-cache"
export PYTHONUNBUFFERED=1

(
    cd "$SOURCE_SNAPSHOT"
    exec env \
        CUDA_VISIBLE_DEVICES=0 \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH="$SOURCE_SNAPSHOT:$OPENPI_SOURCE/src:$OPENPI_SOURCE/packages/openpi-client/src" \
        "$OPENPI_PYTHON" -m hybrid_rollout.robolab.pi05_server.server \
        --checkpoint "$CHECKPOINT" \
        --port "$POLICY_PORT" \
        --identity-output "$ARCHIVE_ROOT/policy_identity.json"
) > "$LOG_ROOT/policy.log" 2>&1 &
POLICY_PID=$!
printf '%s\n' "$POLICY_PID" > "$ARCHIVE_ROOT/policy.pid"

(
    export TASK_ROOT="$LEGACY_RUNTIME_ROOT"
    export CUDA_VISIBLE_DEVICES=1
    export PYTHONPATH="$SOURCE_SNAPSHOT:$ROBOLAB_SOURCE:$OPENPI_SOURCE/packages/openpi-client/src"
    # shellcheck disable=SC1091
    source "$SOURCE_SNAPSHOT/hybrid_rollout/robolab/robolab_server/runtime.sh"
    cd "$ROBOLAB_SOURCE"
    exec "$ROBOLAB_PYTHON" -m hybrid_rollout.robolab.robolab_server.server \
        --task BlockStackingSpecifiedOrderTask \
        --output "$ARCHIVE_ROOT/sim" \
        --port "$SIM_PORT" \
        --seed 0 \
        --instruction-type default
) > "$LOG_ROOT/sim.log" 2>&1 &
SIM_PID=$!
printf '%s\n' "$SIM_PID" > "$ARCHIVE_ROOT/sim.pid"

deadline=$((SECONDS + 900))
until grep -q '"event": "ready"' "$LOG_ROOT/sim.log"; do
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        printf 'RoboLab server exited before readiness; see %s\n' "$LOG_ROOT/sim.log" >&2
        exit 3
    fi
    if (( SECONDS >= deadline )); then
        printf 'Timed out waiting for RoboLab readiness; see %s\n' "$LOG_ROOT/sim.log" >&2
        exit 3
    fi
    sleep 2
done

deadline=$((SECONDS + 900))
until [[ -f "$ARCHIVE_ROOT/policy_identity.json" ]]; do
    if ! kill -0 "$POLICY_PID" 2>/dev/null; then
        printf 'OpenPI server exited before identity validation; see %s\n' "$LOG_ROOT/policy.log" >&2
        exit 4
    fi
    if (( SECONDS >= deadline )); then
        printf 'Timed out waiting for OpenPI policy; see %s\n' "$LOG_ROOT/policy.log" >&2
        exit 4
    fi
    sleep 2
done

(
    cd "$SOURCE_SNAPSHOT"
    exec env \
        NO_PROXY=127.0.0.1,localhost \
        no_proxy=127.0.0.1,localhost \
        PYTHONPATH="$SOURCE_SNAPSHOT:$OPENPI_SOURCE/packages/openpi-client/src" \
        "$ROBOLAB_PYTHON" -m hybrid_rollout.robolab.skill.run \
        --output "$CONTROLLER_OUTPUT" \
        --sim-port "$SIM_PORT" \
        --student-port "$POLICY_PORT" \
        --max-decisions 180 \
        --checkpoint "$CHECKPOINT" \
        --seed 0 \
        --task BlockStackingSpecifiedOrderTask \
        --codex "$CODEX_BIN"
) > "$LOG_ROOT/controller.log" 2>&1 &
CONTROLLER_PID=$!
printf '%s\n' "$CONTROLLER_PID" > "$ARCHIVE_ROOT/controller.pid"
set +e
wait "$CONTROLLER_PID"
controller_status=$?
set -e
CONTROLLER_PID=""

if [[ ! -f "$CONTROLLER_OUTPUT/result.json" ]]; then
    printf 'Controller exited with status %s and without a result artifact\n' "$controller_status" >&2
    if [[ "$controller_status" == 0 ]]; then exit 5; fi
    exit "$controller_status"
fi
exit "$controller_status"
