#!/usr/bin/env bash
# Interactive, resumable isolated logins only. Never starts a rollout.
set -euo pipefail
case "${1:-}" in
  a|b|c|check) group="$1" ;;
  *) echo "Usage: bash hybrid_rollout/robodojo/login_pool15.sh {a|b|c|check}"; exit 2 ;;
esac
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
shared_root="${ROLLOUT_SHARED_ROOT:-/mnt/rollout/robodojo_mixed_control}"
if [[ "$group" == check ]]; then
  exec python3 -m hybrid_rollout.robodojo.codex_backend.profiles --shared-root "$shared_root" check-pool15
fi
codex_bin="${ROLLOUT_CODEX_BIN:-/home/runner/.vscode-server/extensions/openai.chatgpt-26.903.71938-linux-x64/bin/linux-x86_64/codex}"
proxy_file="${ROLLOUT_HTTPS_PROXY_FILE:-$shared_root/private/company_https_proxy.url}"
[[ -x "$codex_bin" ]] || { echo "Set ROLLOUT_CODEX_BIN to the installed Codex executable."; exit 2; }
[[ -f "$proxy_file" ]] || { echo "Set ROLLOUT_HTTPS_PROXY_FILE to the private company proxy file."; exit 2; }
exec python3 -m hybrid_rollout.robodojo.codex_backend.profiles --shared-root "$shared_root" \
  login-group "$group" --codex "$codex_bin" --https-proxy-file "$proxy_file"
