#!/usr/bin/env bash
# Sourced by the platform-specific setup scripts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'Install uv before running setup.' >&2; exit 1; }
command -v node >/dev/null || { echo 'Install Node 22 before running setup.' >&2; exit 1; }
[[ "$(node -p 'process.versions.node.split(".")[0]')" == 22 ]] || { echo 'Node 22 is required.' >&2; exit 1; }
UV_PROJECT_ENVIRONMENT=.runtime uv sync --python 3.12 --frozen --extra models --extra dev
.runtime/bin/python scripts/check-accelerator.py "$accelerator"
npm --prefix apps/web ci
npm --prefix apps/web run build
printf 'Dependencies and %s checks passed. Provision and qualify models separately; see docs/runbooks/%s.md\n' "$accelerator" "$guide"
