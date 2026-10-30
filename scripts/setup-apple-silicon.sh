#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Requires native ARM64 macOS (no Rosetta).' >&2; exit 1; }
accelerator=mps
guide=apple-silicon
source "$(dirname "$0")/setup-common.sh"
