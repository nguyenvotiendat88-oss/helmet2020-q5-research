#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ "$#" -ne 3 ]; then
  echo 'Usage: bash scripts/test_training_kit_linux.sh IMAGE_ROOT RUN_DIR MAX_HOURS' >&2
  exit 2
fi
IMAGES="$1"
RUN="$2"
HOURS="$3"
SELECTION_SHA="$(.venv-kit/bin/python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$RUN/selection.json")"
.venv-kit/bin/python -u scripts/run_helmet2020_training_kit.py test --manifests manifests --images "$IMAGES" --output "$RUN" --device cuda:0 --max-hours "$HOURS" --selection-sha256 "$SELECTION_SHA" 2>&1 | tee -a "$RUN.test.console.log"
