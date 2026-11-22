#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python=.venv/bin/python
export PYTHONPATH=
"$python" -m ruff check .
"$python" -m ruff format --check .
"$python" -m pyright
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$python" -m pytest -q
