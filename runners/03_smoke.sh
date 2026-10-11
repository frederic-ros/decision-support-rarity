#!/usr/bin/env bash
# Fast end-to-end check (~2 min, CPU, no network): fabricates a tiny image
# embedding cache + a prepared real cache, then runs every experiment path and
# the full analysis on the smoke config.
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/make_smoke_fixtures.py
rare-al run     --config configs/smoke/smoke.json --only synthetic rarity real image
rare-al run     --config configs/smoke/smoke.json --only oracle
rare-al run     --config configs/smoke/smoke.json --only observability
rare-al analyze --config configs/smoke/smoke.json
echo "SMOKE OK -> results/smoke/assets"
