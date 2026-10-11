#!/usr/bin/env bash
# Download / cache the tabular + domain datasets and build the manifest.
# The imbalanced-learn moderate stratum downloads itself; the domain datasets
# (finance / industry / medicine / cyber) need their raw files dropped under
# data/raw/<folder> first -- see DATA_README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/eswa/full.json}"
rare-al prepare-data --config "$CONFIG"
