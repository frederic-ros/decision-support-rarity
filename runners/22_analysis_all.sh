#!/usr/bin/env bash
# Full post-campaign analysis: tables/figures + out-of-sample validation.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/eswa/campaign.json}"
rare-al analyze            --config "$CONFIG"
rare-al validate-guidelines --config "$CONFIG"
echo "ANALYSIS DONE -> results (assets/ + validation_*.csv + tab_validation.tex)"
