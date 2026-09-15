#!/usr/bin/env bash
# RoboDojo deployment inside hybrid_rollout; no external test-code dependencies.
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/workspace/eval-of-gpt-6-astra-as-policy}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/rollout/robodojo_mixed_control}"
ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-$RUNTIME_ROOT/src/RoboDojo}"
OPENPI_SOURCE="${OPENPI_SOURCE:-$ROBODOJO_SOURCE/XPolicyLab/policy/Pi_05/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/mnt/rollout/openpi_mailbox_repro/openpi-venv/bin/python}"
ROBODOJO_PYTHON="${ROBODOJO_PYTHON:?Set an explicitly verified RoboDojo Isaac Sim 5.1 Python environment}"
CODEX_BIN="${CODEX_BIN:?Set the Codex executable}"
ROLLOUT_CODEX_HOME_DIR="${ROLLOUT_CODEX_HOME_DIR:?Run through cluster_entrypoint.sh with an isolated Codex home}"
ROLLOUT_CODEX_CONFIG="${ROLLOUT_CODEX_CONFIG:?Set the isolated rollout Codex config path}"
ROLLOUT_OPENAI_API_KEY_FILE="${ROLLOUT_OPENAI_API_KEY_FILE:-}"
ROLLOUT_AUTH_PROFILE="${ROLLOUT_AUTH_PROFILE:-galbot}"
export ROLLOUT_EVALUATION_METHOD="${ROLLOUT_EVALUATION_METHOD:-pi05_plus_gpt}"
[[ "$ROLLOUT_EVALUATION_METHOD" == pi05_plus_gpt || "$ROLLOUT_EVALUATION_METHOD" == gpt_only ]] || { echo 'Invalid evaluation method' >&2; exit 2; }
RUN_ID="${RUN_ID:?Set a fresh run identifier}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
TASK="${TASK:-build_tower}"
SEED="${SEED:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
MAX_DECISIONS="${MAX_DECISIONS:-100}"
POLICY_PORT="${POLICY_PORT:-18830}"
SIM_PORT="${SIM_PORT:-19113}"
CHECKPOINT="${CHECKPOINT:-$RUNTIME_ROOT/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUNTIME_ROOT/results}"
ARCHIVE="$RESULTS_ROOT/$RUN_ID"
SNAPSHOT="$ARCHIVE/source_snapshot"
for input in "$ROBODOJO_PYTHON" "$CODEX_BIN" \
    "$ROLLOUT_CODEX_HOME_DIR" "$ROLLOUT_CODEX_CONFIG" \
    "$ROBODOJO_SOURCE/Assets"; do
    [[ -e "$input" ]] || { echo "Missing input: $input" >&2; exit 2; }
done
if [[ "$ROLLOUT_EVALUATION_METHOD" == pi05_plus_gpt ]]; then
    for input in "$OPENPI_PYTHON" "$CHECKPOINT/params" "$CHECKPOINT/assets/arx_x5_sim/norm_stats.json" "$OPENPI_SOURCE/src/openpi"; do
        [[ -e "$input" ]] || { echo "Missing policy input: $input" >&2; exit 2; }
    done
fi
case "$ROLLOUT_AUTH_PROFILE" in
    codex_[abc]|codex_[abc]_[2-5])
        [[ -s "$ROLLOUT_CODEX_HOME_DIR/auth.json" && -n "${ROLLOUT_ACCOUNT_LOCK_FD:-}" ]] || {
            echo 'Managed account login and exclusive lease required' >&2; exit 2;
        } ;;
    *) [[ -s "$ROLLOUT_OPENAI_API_KEY_FILE" ]] || { echo 'Rollout API key file is empty' >&2; exit 2; } ;;
esac
unset OPENAI_API_KEY OPENAI_BASE_URL CODEX_API_KEY
PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.codex_backend.validate \
    "$ROLLOUT_CODEX_CONFIG" --home-config "$ROLLOUT_CODEX_HOME_DIR/config.toml" \
    --shared-root "${ROLLOUT_SHARED_ROOT:-$RUNTIME_ROOT}"
[[ "$RESULTS_ROOT" == /mnt/* ]] || { echo 'RESULTS_ROOT must be an absolute path below /mnt' >&2; exit 2; }
"$ROBODOJO_PYTHON" -c 'import importlib.metadata as m; assert m.version("isaacsim").startswith("5.1."), "RoboDojo checkout requires Isaac Sim 5.1; do not silently reuse RoboLab 5.0"'
"$ROBODOJO_PYTHON" -c 'import socket,sys; ports=[int(p) for p in sys.argv[1:]]; assert len(set(ports))==len(ports); sockets=[socket.socket() for _ in ports]; [s.bind(("127.0.0.1",p)) for s,p in zip(sockets,ports)]' "$POLICY_PORT" "$SIM_PORT"
mkdir -p "$RESULTS_ROOT"
mkdir "$ARCHIVE" || { echo "Refusing existing/reserved archive: $ARCHIVE" >&2; exit 2; }
mkdir -p "$SNAPSHOT" "$ARCHIVE/logs" "$ARCHIVE/sim"
policy_pid=''; sim_pid=''; controller_pid=''
stop_child() {
    local child="$1"
    [[ -n "$child" ]] || return 0
    if kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$child" 2>/dev/null || break
            sleep .5
        done
        kill -KILL "$child" 2>/dev/null || true
    fi
    wait "$child" 2>/dev/null || true
}
cleanup() {
    local status=$?
    trap - EXIT
    set +e
    stop_child "$controller_pid"; stop_child "$sim_pid"; stop_child "$policy_pid"
    printf '%s\n' "$status" > "$ARCHIVE/job_exit_status.txt"
    manifest_status=0
    if [[ -f "$SNAPSHOT/hybrid_rollout/robodojo/artifact_manifest.py" ]]; then
        manifest_pythonpath="$SNAPSHOT"
    elif [[ -f "$CODE_ROOT/hybrid_rollout/robodojo/artifact_manifest.py" ]]; then
        manifest_pythonpath="$CODE_ROOT"
    else
        manifest_pythonpath=''
    fi
    if [[ -n "$manifest_pythonpath" ]]; then
        manifest_args=(--run-root "$ARCHIVE")
        (( status == 0 )) && manifest_args+=(--require-complete)
        env PYTHONPATH="$manifest_pythonpath" "$ROBODOJO_PYTHON" \
            -m hybrid_rollout.robodojo.artifact_manifest "${manifest_args[@]}" \
            > "$ARCHIVE/logs/artifact_manifest.log" 2>&1
        manifest_status=$?
    fi
    if (( status == 0 && manifest_status != 0 )); then
        status=5
        printf '%s\n' "$status" > "$ARCHIVE/job_exit_status.txt"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp -a "$CODE_ROOT/hybrid_rollout" "$SNAPSHOT/hybrid_rollout"
if [[ -f "$CODE_ROOT/source_manifest.json" ]]; then
    cp "$CODE_ROOT/source_manifest.json" "$ARCHIVE/submitted_source_manifest.json"
fi
cp "$ROLLOUT_CODEX_CONFIG" "$ARCHIVE/codex_config.toml"
if [[ -n "${ROLLOUT_EVAL_MANIFEST:-}" ]]; then
    cp "$ROLLOUT_EVAL_MANIFEST" "$ARCHIVE/evaluation_manifest.json"
    cp "${ROLLOUT_CASE_FILE:?}" "$ARCHIVE/evaluation_case.json"
fi
git -C "$ROBODOJO_SOURCE" rev-parse HEAD > "$ARCHIVE/robodojo_head.txt"
if [[ "$ROLLOUT_EVALUATION_METHOD" == pi05_plus_gpt ]]; then
    git -C "$OPENPI_SOURCE" rev-parse HEAD > "$ARCHIVE/openpi_head.txt"
fi
export PYTHONUNBUFFERED=1
export TASK_ROOT="${TASK_ROOT:-$RUNTIME_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/rollout/openpi_mailbox_repro/openpi-cache}"
export CODE_ROOT RUNTIME_ROOT RESULTS_ROOT ROBODOJO_SOURCE OPENPI_SOURCE OPENPI_PYTHON
export ROBODOJO_PYTHON CODEX_BIN RUN_ID TASK SEED EVAL_SEED POLICY_PORT SIM_PORT CHECKPOINT
export POLICY_GPU="${POLICY_GPU:-0}" SIM_GPU="${SIM_GPU:-1}" MAX_DECISIONS
env PYTHONPATH="$SNAPSHOT" "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.runtime_manifest \
    --archive "$ARCHIVE" --config "$ROLLOUT_CODEX_CONFIG" --output "$ARCHIVE/launch_manifest.json"
if [[ "$ROLLOUT_EVALUATION_METHOD" == pi05_plus_gpt ]]; then
(
    cd "$SNAPSHOT"
    exec env CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH="$SNAPSHOT:$OPENPI_SOURCE/src:$OPENPI_SOURCE/packages/openpi-client/src" \
        "$OPENPI_PYTHON" -m hybrid_rollout.robodojo.pi05_server.server \
        --checkpoint "$CHECKPOINT" --port "$POLICY_PORT" --identity-output "$ARCHIVE/policy_identity.json"
) > "$ARCHIVE/logs/policy.log" 2>&1 &
policy_pid=$!
fi
(
    export CUDA_VISIBLE_DEVICES="$SIM_GPU"
    export PYTHONPATH="$SNAPSHOT:$ROBODOJO_SOURCE:$ROBODOJO_SOURCE/XPolicyLab:$ROBODOJO_SOURCE/third_party/curobo${SIM_EXTRA_PYTHONPATH:+:$SIM_EXTRA_PYTHONPATH}"
    source "$SNAPSHOT/hybrid_rollout/robodojo/robodojo_server/runtime.sh"
    mkdir -p "$ARCHIVE/sim/native_runtime"
    cd "$ARCHIVE/sim/native_runtime"
    evaluation_args=(--eval-seed "$EVAL_SEED")
    if [[ -n "${ROLLOUT_EVAL_MANIFEST:-}" ]]; then
        evaluation_args+=(--eval-manifest "$ARCHIVE/evaluation_manifest.json"
            --case-file "$ARCHIVE/evaluation_case.json")
    fi
    exec "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.robodojo_server.server \
        --task "$TASK" --output "$ARCHIVE/sim" --port "$SIM_PORT" "${evaluation_args[@]}"
) > "$ARCHIVE/logs/sim.log" 2>&1 &
sim_pid=$!
[[ -z "$policy_pid" ]] || printf '%s\n' "$policy_pid" > "$ARCHIVE/policy.pid"
printf '%s\n' "$sim_pid" > "$ARCHIVE/sim.pid"
deadline=$((SECONDS+900))
until { [[ "$ROLLOUT_EVALUATION_METHOD" == gpt_only || -f "$ARCHIVE/policy_identity.json" ]]; } && grep -q '"event": "ready"' "$ARCHIVE/logs/sim.log"; do
    if [[ -n "$policy_pid" ]]; then
        kill -0 "$policy_pid" || { echo "Policy exited: $ARCHIVE/logs" >&2; exit 3; }
    fi
    kill -0 "$sim_pid" || { echo "Simulator exited: $ARCHIVE/logs" >&2; exit 3; }
    (( SECONDS < deadline )) || { echo 'Service startup timeout' >&2; exit 3; }
    sleep 2
done
(
    cd "$SNAPSHOT"
    auth_environment=()
    controller_args=(--evaluation-method "$ROLLOUT_EVALUATION_METHOD")
    controller_pythonpath="$SNAPSHOT"
    if [[ "$ROLLOUT_EVALUATION_METHOD" == pi05_plus_gpt ]]; then
        controller_args+=(--checkpoint "$CHECKPOINT" --student-port "$POLICY_PORT")
        controller_pythonpath="$SNAPSHOT:$OPENPI_SOURCE/packages/openpi-client/src"
    fi
    if [[ "$ROLLOUT_AUTH_PROFILE" == galbot || "$ROLLOUT_AUTH_PROFILE" == koozhan ]]; then
        codex_api_key="$(<"$ROLLOUT_OPENAI_API_KEY_FILE")"
        auth_environment+=(OPENAI_API_KEY="$codex_api_key")
    fi
    exec env CODEX_HOME="$ROLLOUT_CODEX_HOME_DIR" "${auth_environment[@]}" \
        NO_PROXY="127.0.0.1,localhost,${ROLLOUT_GATEWAY_NO_PROXY-gateway.example.invalid}" \
        no_proxy="127.0.0.1,localhost,${ROLLOUT_GATEWAY_NO_PROXY-gateway.example.invalid}" \
        PYTHONPATH="$controller_pythonpath" \
        "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.skill.run \
        --output "$ARCHIVE/controller" --task "$TASK" "${controller_args[@]}" \
        --sim-port "$SIM_PORT" --codex "$CODEX_BIN" \
        --seed "$SEED" --max-decisions "$MAX_DECISIONS"
) > "$ARCHIVE/logs/controller.log" 2>&1 &
controller_pid=$!
printf '%s\n' "$controller_pid" > "$ARCHIVE/controller.pid"
wait "$controller_pid"
controller_pid=''
[[ -f "$ARCHIVE/controller/result.json" ]] || { echo 'No rollout result' >&2; exit 4; }
