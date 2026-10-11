#!/usr/bin/env bash
# Fetch the cross-domain datasets into data/raw/ (UCI automatic; Kaggle needs the
# kaggle CLI; CIC-IDS2017/ISIC are manual). Prints a checklist at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/fetch_domains.py "$@"
