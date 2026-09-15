#!/usr/bin/env bash
# Reuse the supplied image and shared environments. No image build is required.
set -euo pipefail
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CODE_ROOT="${CODE_ROOT:-$(cd "$ENTRY_ROOT/../.." && pwd)}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/rollout/robodojo_mixed_control}"
export ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-$RUNTIME_ROOT}"
export ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-$RUNTIME_ROOT/src/RoboDojo}"
export OPENPI_SOURCE="${OPENPI_SOURCE:-$ROBODOJO_SOURCE/XPolicyLab/policy/Pi_05/openpi}"
export ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-$RUNTIME_ROOT/sim-venv/bin/python}"
export OPENPI_PYTHON="${OPENPI_PYTHON:-/mnt/rollout/openpi_mailbox_repro/openpi-venv/bin/python}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/rollout/openpi_mailbox_repro/openpi-cache}"
export CHECKPOINT="${CHECKPOINT:-$RUNTIME_ROOT/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999}"
export CODEX_BIN="${CODEX_BIN:?Set CODEX_BIN to the pinned shared Codex 0.153.4 executable}"
export ROLLOUT_CODEX_VERSION="${ROLLOUT_CODEX_VERSION:-0.153.4}"
export POLICY_GPU="${POLICY_GPU:-0}" SIM_GPU="${SIM_GPU:-1}"
export POLICY_PORT="${POLICY_PORT:-18830}" SIM_PORT="${SIM_PORT:-19113}"
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-1}"
: "${ROLLOUT_EVAL_MANIFEST:?Set the frozen paired evaluation manifest}"
: "${ROLLOUT_EVAL_MANIFEST_SHA256:?Set the frozen panel identity}"
export MAX_DECISIONS="${MAX_DECISIONS:-100}" CODEX_MAX_TOTAL_TOKENS="${CODEX_MAX_TOTAL_TOKENS:-0}"
export CODEX_IMAGE_MAX_EDGE="${CODEX_IMAGE_MAX_EDGE:-480}"
[[ "$CODEX_IMAGE_MAX_EDGE" =~ ^[0-9]+$ ]] || { echo 'CODEX_IMAGE_MAX_EDGE must be non-negative' >&2; exit 2; }
export ROLLOUT_MAX_SECONDS="${ROLLOUT_MAX_SECONDS:-7200}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

export ROLLOUT_AUTH_PROFILE="${ROLLOUT_AUTH_PROFILE:-galbot}"
case "$ROLLOUT_AUTH_PROFILE" in
    codex_[abc]|codex_[abc]_[2-5])
        if [[ -z "${ROLLOUT_ACCOUNT_LOCK_FD:-}" ]]; then
            exec env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" \
                -m hybrid_rollout.robodojo.codex_backend.profiles lease -- bash "$ENTRY_ROOT/cluster_entrypoint.sh"
        fi
        [[ "$ROLLOUT_ACCOUNT_LOCK_FD" =~ ^[0-9]+$ && -e "/proc/$$/fd/$ROLLOUT_ACCOUNT_LOCK_FD" ]] || {
            echo 'Managed account requires a live exclusive lease' >&2; exit 2;
        }
        ;;
    galbot|koozhan) ;;
    *) echo 'Unknown auth profile; no fallback' >&2; exit 2 ;;
esac

for executable in "$ROBODOJO_PYTHON" "$CODEX_BIN"; do
    [[ -x "$executable" ]] || { echo "Missing shared executable: $executable" >&2; exit 2; }
done
if [[ "${ROLLOUT_EVALUATION_METHOD:-pi05_plus_gpt}" == pi05_plus_gpt ]]; then
    [[ -x "$OPENPI_PYTHON" ]] || { echo "Missing shared policy executable: $OPENPI_PYTHON" >&2; exit 2; }
fi
source "$ENTRY_ROOT/tool_environment.sh"
umask 077
[[ "$ROLLOUT_SHARED_ROOT" == /mnt/rollout/* ]] || {
    echo 'Rollout output must be below /mnt/rollout' >&2; exit 2;
}
for identifier in "${ROLLOUT_EXPERIMENT_ID:?}" "${ROLLOUT_REPLICA_ID:?}" "${ROLLOUT_ATTEMPT:-0}"; do
    [[ "$identifier" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid rollout identifier' >&2; exit 2; }
done
bootstrap_private="$ROLLOUT_SHARED_ROOT/private_runtime/${ROLLOUT_EXPERIMENT_ID:?}/replica_${ROLLOUT_REPLICA_ID:?}/attempt_${ROLLOUT_ATTEMPT:-0}"
mkdir -p "$bootstrap_private"
version_probe_dir="$(mktemp -d "$bootstrap_private/bootstrap.XXXXXX")"
export TMPDIR="$version_probe_dir/tmp"
mkdir "$TMPDIR"
actual_codex_version="$(env CODEX_HOME="$version_probe_dir" "$CODEX_BIN" --version)"
# Keep startup logs/state on shared storage even if the container disappears.
[[ "$actual_codex_version" == "codex-cli $ROLLOUT_CODEX_VERSION" ]] || {
    echo "Unexpected Codex version: $actual_codex_version" >&2; exit 2;
}
for executable in bash grep git; do
    command -v "$executable" >/dev/null || { echo "Missing image prerequisite: $executable" >&2; exit 2; }
done
# Cluster containers may have HOME=/root, so pin the shared graphics paths.
export DAGGER_GRAPHICS_ROOT="${DAGGER_GRAPHICS_ROOT:-/home/runner/.local/share/robolab-runtime}"
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
export DAGGER_NVIDIA_RUNTIME="${DAGGER_NVIDIA_RUNTIME:-$DAGGER_GRAPHICS_ROOT/NVIDIA-Linux-x86_64-$driver_version}"
export DAGGER_SYSROOT="${DAGGER_SYSROOT:-$DAGGER_GRAPHICS_ROOT/sysroot}"
[[ -f "$DAGGER_NVIDIA_RUNTIME/libGLX_nvidia.so.$driver_version" ]] || {
    echo "Missing shared NVIDIA graphics runtime for driver $driver_version" >&2; exit 2;
}
env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" \
    -m hybrid_rollout.robodojo.codex_backend.profiles config --output "$version_probe_dir/config.toml"
env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" \
    -m hybrid_rollout.robodojo.codex_backend.validate \
    "$version_probe_dir/config.toml"
env CODEX_HOME="$version_probe_dir" PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" \
    -m hybrid_rollout.robodojo.tool_preflight --codex "$CODEX_BIN" \
    --output "$version_probe_dir/tool_preflight"
exec env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.batch
