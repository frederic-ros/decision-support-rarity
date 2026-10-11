#!/usr/bin/env bash
# Build every table and figure of the manuscript from the results. A failing
# section is logged and skipped, so a partial run still produces a partial report.
# Pass a v13 archive path to overlay provenance (v13 / new / missing).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/eswa/full.json}"
V13="${2:-}"
if [ -n "$V13" ]; then
  rare-al analyze --config "$CONFIG" --v13 "$V13"
else
  rare-al analyze --config "$CONFIG"
fi
