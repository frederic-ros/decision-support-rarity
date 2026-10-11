#!/usr/bin/env bash
# Run every experiment: synthetic grid, rarity stress, real benchmark, image
# benchmark (two embeddings), discovery oracle, observability probe.
# Resumable -- re-running skips completed (state, learner, budget, start, strategy).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/eswa/full.json}"
rare-al run --config "$CONFIG" --only synthetic rarity real image
rare-al run --config "$CONFIG" --only oracle
rare-al run --config "$CONFIG" --only observability
