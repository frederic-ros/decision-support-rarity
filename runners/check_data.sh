#!/usr/bin/env bash
# Read-only inventory of data/ (images, rare27, standard20, domain). Changes nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/check_data.py "$@"
