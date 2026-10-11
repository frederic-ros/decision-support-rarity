#!/usr/bin/env bash
# Extract the TWO embeddings (DINOv2 ViT-S/14 + ResNet-18) for every image set in
# the config. Needs a GPU + network and the image extras (pip install -e '.[images]').
# torchvision sets download automatically; ISIC/PCAM need an ImageFolder under
# data/images_raw/<dataset> (see DATA_README.md).
#
# If you already have a compatible embedding cache (e.g. a forecastability cache
# laid out as <dir>/<dataset>/<backbone>.npz), point embedding_cache_dirs at it in
# the config and skip this step for those datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/eswa/full.json}"
rare-al extract-images --config "$CONFIG"
