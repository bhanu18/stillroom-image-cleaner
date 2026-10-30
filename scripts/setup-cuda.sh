#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Requires Linux x86-64.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/container GPU access is required.' >&2; exit 1; }
nvidia-smi
accelerator=cuda
guide=nvidia-cuda
source "$(dirname "$0")/setup-common.sh"
