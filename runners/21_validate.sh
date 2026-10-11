#!/usr/bin/env bash
# Out-of-sample (leave-one-domain-out) validation of the decision procedure.
# Runs on the real-benchmark CSVs AFTER the campaign; writes tab_validation.tex.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/eswa/campaign.json}"
rare-al validate-guidelines --config "$CONFIG"
