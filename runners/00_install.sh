#!/usr/bin/env bash
# Install rare_al 2.0 (CPU core). Add image / boosting extras as needed.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e .
echo "Core installed. For image embedding extraction (GPU):  pip install -e '.[images]'"
echo "For LightGBM/XGBoost learners:                         pip install -e '.[boost]'"
