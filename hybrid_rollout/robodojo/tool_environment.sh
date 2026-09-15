#!/usr/bin/env bash
# Source once: inherit the image's tools and add installation directories,
# never an executable allowlist. Child shells inherit this PATH.
export PATH="$(dirname "${CODEX_BIN:?}"):$(dirname "${ROBODOJO_PYTHON:?}"):${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
