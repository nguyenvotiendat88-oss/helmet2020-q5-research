#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
WHEEL="${1:-cu126}"
case "$WHEEL" in cu126|cu128|cpu) ;; *) echo 'Use cu126, cu128 or cpu' >&2; exit 2;; esac
python3 -m venv .venv-kit
.venv-kit/bin/python -m pip install --upgrade pip
.venv-kit/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "https://download.pytorch.org/whl/$WHEEL"
.venv-kit/bin/python -m pip install -r requirements-training-kit.txt
.venv-kit/bin/python -m pytest tests/test_kit_data.py tests/test_kit_models.py tests/test_kit_runner.py tests/test_kit_reporting.py tests/test_prepare_helmet2020_trainval_resplit.py -q
