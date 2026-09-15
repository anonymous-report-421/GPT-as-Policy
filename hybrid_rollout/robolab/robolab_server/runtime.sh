#!/usr/bin/env bash
# Source this file in the RoboLab process; it does not modify system libraries.
export TASK_ROOT="${TASK_ROOT:-$HOME/Projects/gpt-gated-dagger}"
DAGGER_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
export DAGGER_NVIDIA_RUNTIME="${DAGGER_NVIDIA_RUNTIME:-$HOME/.local/share/robolab-runtime/NVIDIA-Linux-x86_64-$DAGGER_DRIVER_VERSION}"
export DAGGER_SYSROOT="${DAGGER_SYSROOT:-$HOME/.local/share/robolab-runtime/sysroot}"
if [[ ! -f "$DAGGER_NVIDIA_RUNTIME/libGLX_nvidia.so.$DAGGER_DRIVER_VERSION" ]]; then
  printf 'Missing NVIDIA userspace graphics libraries matching driver %s\n' "$DAGGER_DRIVER_VERSION" >&2
  return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH="$DAGGER_NVIDIA_RUNTIME:$DAGGER_SYSROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$DAGGER_NVIDIA_RUNTIME/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$DAGGER_NVIDIA_RUNTIME/10_nvidia.json"
export XDG_RUNTIME_DIR="$TASK_ROOT/runtime/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export OMNI_KIT_ACCEPT_EULA=Y
