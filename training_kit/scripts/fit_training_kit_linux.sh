#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ "$#" -lt 3 ]; then
  echo 'Usage: bash scripts/fit_training_kit_linux.sh IMAGE_ROOT RUN_DIR MAX_HOURS [CONFIG]' >&2
  exit 2
fi
IMAGES="$1"
RUN="$2"
HOURS="$3"
CONFIG="${4:-configs/helmet2020_training_kit_v1.json}"
RESUME=()
if [ -f "$RUN/study.json" ]; then RESUME=(--resume); fi
mkdir -p -- "$(dirname -- "$RUN")"
.venv-kit/bin/python -u scripts/run_helmet2020_training_kit.py fit --manifests manifests --images "$IMAGES" --output "$RUN" --device cuda:0 --max-hours "$HOURS" --weights pretrained --config "$CONFIG" "${RESUME[@]}" 2>&1 | tee -a "$RUN.console.log"
